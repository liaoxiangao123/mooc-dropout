"""业务服务层：不依赖任何 HTTP 类型，可被脚本、CLI、API 复用。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..evaluation.metrics import risk_level, summarize_risk_bands
from ..features.registry import feature_dictionary
from ..logging_config import get_logger
from ..repository.artifact_repository import ArtifactRepository

logger = get_logger(__name__)


class PredictionService:
    def __init__(self, repository: ArtifactRepository) -> None:
        self.repository = repository

    # ---- 元信息 ----
    def is_ready(self) -> bool:
        return self.repository.is_available

    def model_info(self) -> dict[str, Any]:
        artifacts = self.repository.load()
        meta = artifacts.metadata
        calibration = meta.get("calibration") or {
            "method": "none",
            "label": "未校准",
            "fitted": False,
        }
        return {
            "model_name": artifacts.model_name,
            "model_version": artifacts.version,
            "trained_at": meta.get("trained_at"),
            "threshold": artifacts.threshold,
            "cost_matrix": meta.get("cost_matrix", {}),
            "metrics_test": meta.get("metrics_test", {}),
            "metrics_test_uncalibrated": meta.get("metrics_test_uncalibrated", {}),
            "calibration": calibration,
            "is_calibrated": artifacts.is_calibrated,
            "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"),
            "feature_count": len(artifacts.builder.feature_names),
            "categorical_features": artifacts.builder.categorical_features,
            "top_features": artifacts.explainer.global_importance(top_k=10),
            "features": feature_dictionary(),
        }

    # ---- 推理 ----
    def predict(
        self,
        *,
        enrollment_id: str,
        events: Iterable[dict[str, Any]],
        course_id: str = "unknown",
        username: str = "unknown",
        top_k: int = 5,
    ) -> dict[str, Any]:
        artifacts = self.repository.load()

        log_frame = pd.DataFrame(list(events))
        if log_frame.empty:
            raise ValueError("events 不能为空")
        log_frame["enrollment_id"] = enrollment_id

        meta_frame = pd.DataFrame(
            [
                {
                    "enrollment_id": enrollment_id,
                    "course_id": course_id,
                    "username": username,
                }
            ]
        )

        features = artifacts.builder.transform(log_frame, meta_frame)
        raw_probability = float(np.clip(artifacts.model.predict_proba(features)[0], 0.0, 1.0))
        # 有校准器就用校准后概率决策；两者都返回，便于前端区分「分数」与「概率」
        probability = (
            artifacts.calibrator.transform_one(raw_probability)
            if artifacts.is_calibrated
            else raw_probability
        )
        level = risk_level(probability, artifacts.threshold)
        factors = artifacts.explainer.explain(features, top_k=top_k) if top_k > 0 else []

        return {
            "enrollment_id": enrollment_id,
            "course_id": course_id,
            "risk_probability": round(probability, 4),
            "raw_probability": round(raw_probability, 4),
            "calibrated": artifacts.is_calibrated,
            "risk_level": level,
            "decision_threshold": round(artifacts.threshold, 4),
            "top_factors": factors,
            "model_name": artifacts.model_name,
            "model_version": artifacts.version,
        }

    def predict_batch(
        self,
        records: Iterable[dict[str, Any]],
        *,
        top_k: int = 0,
    ) -> dict[str, Any]:
        items = []
        for record in records:
            items.append(
                self.predict(
                    enrollment_id=record["enrollment_id"],
                    events=record["events"],
                    course_id=record.get("course_id", "unknown"),
                    username=record.get("username", "unknown"),
                    top_k=top_k,
                )
            )

        level_counts = summarize_risk_bands([item["risk_level"] for item in items])
        probabilities = [item["risk_probability"] for item in items]
        summary = {
            "n_records": len(items),
            **level_counts,
            "mean_probability": round(float(np.mean(probabilities)), 4) if probabilities else 0.0,
            # 按当前阈值判定需要干预的人数（= 高风险人数）
            "n_flagged": level_counts.get("high", 0),
        }

        logger.info(
            "批量预测完成",
            extra={"event": "predict.batch", "n_records": len(items)},
        )
        return {"items": items, "summary": summary}

    # ---- 干预名单 ----
    def build_intervention_list(
        self,
        records: Iterable[dict[str, Any]],
        *,
        only_flagged: bool = True,
        min_level: str = "low",
    ) -> list[dict[str, Any]]:
        """把批量预测结果整理成「可直接派单」的名单。

        排序即优先级：先按风险概率降序，同分按顶尖风险因子排序稳定输出。
        只保留需要干预的学生（默认阈值以上），避免教师面对一份几百人的全量名单 ——
        名单的价值在于「谁先打电话」，而不是「把所有概率列一遍」。
        """
        order = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
        floor = order.get(min_level, 0)

        rows: list[dict[str, Any]] = []
        for record in records:
            item = self.predict(
                enrollment_id=record["enrollment_id"],
                events=record["events"],
                course_id=record.get("course_id", "unknown"),
                username=record.get("username", "unknown"),
                top_k=1,
            )
            if only_flagged and item["risk_probability"] < item["decision_threshold"]:
                continue
            if order.get(item["risk_level"], 0) < floor:
                continue

            top = item["top_factors"][0] if item["top_factors"] else {}
            rows.append(
                {
                    "enrollment_id": item["enrollment_id"],
                    "course_id": item["course_id"],
                    "risk_probability": item["risk_probability"],
                    "risk_level": item["risk_level"],
                    "decision_threshold": item["decision_threshold"],
                    "top_factor": top.get("feature", ""),
                    "top_factor_value": top.get("value", ""),
                    "model_version": item["model_version"],
                }
            )

        rows.sort(key=lambda r: (-float(r["risk_probability"]), str(r["enrollment_id"])))
        for rank, row in enumerate(rows, start=1):
            row["priority"] = rank
        logger.info(
            "干预名单生成",
            extra={"event": "predict.intervention_list", "n_records": len(rows)},
        )
        return rows
