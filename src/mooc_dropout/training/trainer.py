"""训练编排：切分 -> 特征 -> 训练 -> OOF 阈值 -> 评估 -> 落盘。

方法学要点：
    1. 分组切分（按 username），避免同一学习者的多条记录跨集；
    2. 特征构建器只在训练集上 fit；
    3. 决策阈值在训练集的 OOF 预测上选择，绝不在测试集上调 —— 这是很多同类毕设的硬伤；
    4. 同时保存 0.5 阈值与代价最优阈值下的指标，用于消融对比；
    5. 若启用概率校准，校准器同样只在 OOF 上拟合，且阈值必须在校准后的尺度上重选
       （否则沿用旧阈值会让代价上升，见 E8）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.schema import validate_enrollments, validate_logs
from ..data.split import group_train_test_split, oof_group_splits
from ..errors import DataQualityError, ValidationFailure
from ..evaluation.cost import CostMatrix, cost_optimal_threshold
from ..evaluation.metrics import classification_metrics
from ..features.builder import FeatureBuilder, FeatureConfig
from ..logging_config import get_logger
from ..models.base import BaseRiskModel
from ..models.calibration import ProbabilityCalibrator
from ..models.registry import build_model

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name: str = "catboost"
    test_size: float = 0.2
    n_splits: int = 5
    seed: int = 42
    cost_fn: float = 1000.0
    cost_fp: float = 100.0
    threshold_strategy: str = "oof_cost"  # oof_cost | elkan | fixed
    fixed_threshold: float = 0.5
    compute_oof: bool = True
    weighting: str = "class_weight"
    model_params: dict[str, Any] = field(default_factory=dict)
    # none | platt | isotonic：校准器在 OOF 上拟合，阈值随之重选
    calibration: str = "none"

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> "TrainConfig":
        base = cls(
            seed=settings.random_seed,
            cost_fn=settings.cost_fn,
            cost_fp=settings.cost_fp,
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base


@dataclass
class PreparedData:
    train_enrollments: pd.DataFrame
    test_enrollments: pd.DataFrame
    train_logs: pd.DataFrame
    test_logs: pd.DataFrame
    builder: FeatureBuilder
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


@dataclass
class TrainResult:
    metrics: dict[str, Any]
    threshold: float
    artifact_dir: Path
    model: BaseRiskModel
    builder: FeatureBuilder
    importance: pd.Series


class Trainer:
    def __init__(self, config: TrainConfig | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = config or TrainConfig.from_settings(self.settings)
        self.cost_matrix = CostMatrix(c_fn=self.config.cost_fn, c_fp=self.config.cost_fp)

    # ------------------------------------------------------------------
    def prepare(self, logs: pd.DataFrame, enrollments: pd.DataFrame) -> PreparedData:
        logs = validate_logs(logs)
        enrollments = validate_enrollments(enrollments)

        known = set(enrollments["enrollment_id"])
        logs = logs.loc[logs["enrollment_id"].isin(known)]
        if logs.empty:
            raise ValidationFailure("过滤后无可用日志，请检查 enrollment_id 是否对齐")

        train_enr, test_enr = group_train_test_split(
            enrollments, test_size=self.config.test_size, seed=self.config.seed
        )
        train_ids = set(train_enr["enrollment_id"])
        test_ids = set(test_enr["enrollment_id"])

        train_logs = logs.loc[logs["enrollment_id"].isin(train_ids)]
        test_logs = logs.loc[logs["enrollment_id"].isin(test_ids)]
        if train_logs.empty or test_logs.empty:
            raise ValidationFailure("切分后某一侧无日志，请增大样本量或调整切分比例")

        builder = FeatureBuilder(FeatureConfig.from_settings(self.settings)).fit(train_logs)
        X_train = builder.transform(train_logs, train_enr)
        X_test = builder.transform(test_logs, test_enr)

        y_train = train_enr.set_index("enrollment_id")["dropout"].reindex(X_train.index).to_numpy()
        y_test = test_enr.set_index("enrollment_id")["dropout"].reindex(X_test.index).to_numpy()

        return PreparedData(
            train_enrollments=train_enr,
            test_enrollments=test_enr,
            train_logs=train_logs,
            test_logs=test_logs,
            builder=builder,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _save_predictions(
        data: PreparedData,
        prob: np.ndarray,
        prob_raw: np.ndarray,
        artifact_dir: Path,
    ) -> None:
        """把测试集逐样本预测写成 CSV，供容量分析与错误复盘使用。"""
        index = data.X_test.index.astype(str)
        side = (
            data.test_enrollments.set_index("enrollment_id")
            .reindex(index)[["course_id"]]
            .reset_index()
        )
        frame = pd.DataFrame(
            {
                "enrollment_id": index,
                "y_true": data.y_test.astype(int),
                "probability": np.round(prob, 6),
                "raw_probability": np.round(prob_raw, 6),
            }
        )
        if "course_id" in side.columns:
            frame.insert(1, "course_id", side["course_id"].to_numpy())
        frame = frame.sort_values("probability", ascending=False, kind="mergesort")
        frame.to_csv(artifact_dir / "predictions_test.csv", index=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _save_oof(data: PreparedData, oof_prob: np.ndarray, artifact_dir: Path) -> None:
        """落盘训练集的 OOF 预测，供代价比敏感性扫描复用（避免反复重训）。

        OOF 与 ``data.y_train`` 同序（均按 X_train.index），因此 ID 也必须取 X_train.index，
        否则落盘文件里会出现「概率与标签错位」——这种错误在离线分析里极难察觉。
        """
        if len(oof_prob) != len(data.X_train):
            return
        frame = pd.DataFrame(
            {
                "enrollment_id": data.X_train.index.astype(str),
                "y_true": data.y_train.astype(int),
                "probability": np.round(oof_prob, 6),
            }
        )
        frame.to_csv(artifact_dir / "predictions_oof.csv", index=False)

    # ------------------------------------------------------------------
    def create_model(self, overrides: dict[str, Any] | None = None) -> BaseRiskModel:
        """按当前配置构建模型；overrides 可覆盖 cost_ratio / weighting 等超参。"""
        params = dict(self.config.model_params)
        params.update(overrides or {})
        cost_ratio = float(params.pop("cost_ratio", self.cost_matrix.ratio))
        weighting = str(params.pop("weighting", self.config.weighting))
        model_name = str(params.pop("model_name", self.config.model_name))
        return build_model(
            model_name,
            cost_ratio=cost_ratio,
            weighting=weighting,
            random_seed=self.config.seed,
            categorical_features=["course_id", "username"],
            **params,
        )

    def _oof_probabilities(self, data: PreparedData) -> np.ndarray:
        """在训练集上做分组 K 折，得到 OOF 预测，用于选择阈值。

        **对齐是这里唯一容易出错的地方**：OOF 数组必须与 ``data.y_train`` 同序，
        而 ``y_train`` 是按 ``X_train.index``（特征表按 enrollment_id 排序后的顺序）
        取的标签。若直接按 ``train_enrollments`` 的行号填充 OOF，在 enrollment_id
        不是等宽补零字符串时（真实数据的 ID 是 "1"、"2"、…、"120542"，字典序与数值序
        不一致）两者顺序不同，阈值搜索会在一组**打乱的标签**上进行 —— 得到的阈值看似
        「模型建议全报警」，实际是标签错位。真实数据上 OOF AUC 只有 0.51、测试集却
        有 0.82，正是这个错位的指纹。

        因此这里统一用 enrollment_id 做键：先按 ID 收集，再 reindex 到 X_train.index。
        """
        oof = pd.Series(
            np.nan,
            index=pd.Index(data.train_enrollments["enrollment_id"].astype(str).to_numpy()),
            dtype="float64",
        )

        for fold, (tr_idx, va_idx) in enumerate(
            oof_group_splits(
                data.train_enrollments, n_splits=self.config.n_splits, seed=self.config.seed
            )
        ):
            sub_train = data.train_enrollments.iloc[tr_idx]
            sub_valid = data.train_enrollments.iloc[va_idx]

            train_mask = data.train_logs["enrollment_id"].isin(set(sub_train["enrollment_id"]))
            valid_mask = data.train_logs["enrollment_id"].isin(set(sub_valid["enrollment_id"]))

            X_tr = data.builder.transform(data.train_logs.loc[train_mask], sub_train)
            X_va = data.builder.transform(data.train_logs.loc[valid_mask], sub_valid)
            y_tr = sub_train.set_index("enrollment_id")["dropout"].reindex(X_tr.index).to_numpy()

            model = self.create_model()
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_va)

            oof.loc[X_va.index.astype(str)] = prob

            logger.info(
                "OOF 折完成",
                extra={"event": "training.oof_fold", "n_records": int(len(X_va))},
            )

        aligned = oof.reindex(pd.Index(data.X_train.index.astype(str)))
        missing = int(aligned.isna().sum())
        if missing:
            raise DataQualityError(
                f"OOF 预测与训练集对齐后有 {missing} 条缺失，无法选阈值",
                details={"n_train": int(len(data.X_train)), "missing": missing},
            )
        return aligned.to_numpy(dtype="float64")

    # ------------------------------------------------------------------
    def run(self, logs: pd.DataFrame, enrollments: pd.DataFrame) -> TrainResult:
        started = time.perf_counter()
        data = self.prepare(logs, enrollments)

        model = self.create_model()
        model.fit(data.X_train, data.y_train)

        # ---- 决策阈值 ----
        oof_prob: np.ndarray | None = None
        if self.config.threshold_strategy == "fixed":
            threshold = self.config.fixed_threshold
        elif self.config.threshold_strategy == "elkan":
            threshold = self.cost_matrix.elkan_threshold
        else:
            if self.config.compute_oof:
                oof_prob = self._oof_probabilities(data)
                threshold = float(
                    cost_optimal_threshold(data.y_train, oof_prob, self.cost_matrix)["threshold"]
                )
            else:
                threshold = self.cost_matrix.elkan_threshold

        # ---- 概率校准（可选）----
        # 关键顺序：先拟合校准器 -> 变换 OOF -> 在新尺度上重选阈值。
        # 若沿用校准前的阈值，等价于用旧尺子量新布，代价会明显上升。
        calibrator = ProbabilityCalibrator(self.config.calibration)
        if self.config.calibration != "none":
            if oof_prob is None:
                logger.warning(
                    "未计算 OOF，无法拟合校准器，已跳过校准",
                    extra={"event": "training.calibration_skipped"},
                )
                calibrator = ProbabilityCalibrator("none")
            else:
                calibrator.fit(oof_prob, data.y_train)
                oof_prob = calibrator.transform(oof_prob)
                threshold = float(
                    cost_optimal_threshold(data.y_train, oof_prob, self.cost_matrix)["threshold"]
                )
                logger.info(
                    "概率校准完成（阈值已重选）",
                    extra={
                        "event": "training.calibrated",
                        "calibration": self.config.calibration,
                        "threshold": round(threshold, 4),
                    },
                )

        # ---- 评估 ----
        prob_test_raw = model.predict_proba(data.X_test)
        prob_test = calibrator.transform(prob_test_raw) if calibrator.method != "none" else prob_test_raw
        metrics_test = classification_metrics(
            data.y_test, prob_test, threshold=threshold, cost_matrix=self.cost_matrix
        )
        metrics_default = classification_metrics(
            data.y_test, prob_test, threshold=0.5, cost_matrix=self.cost_matrix
        )
        metrics_elkan = classification_metrics(
            data.y_test, prob_test, threshold=self.cost_matrix.elkan_threshold, cost_matrix=self.cost_matrix
        )
        metrics_raw = (
            classification_metrics(
                data.y_test, prob_test_raw, threshold=threshold, cost_matrix=self.cost_matrix
            )
            if calibrator.method != "none"
            else {}
        )

        if oof_prob is not None:
            metrics_oof = classification_metrics(
                data.y_train, oof_prob, threshold=threshold, cost_matrix=self.cost_matrix
            )
        else:
            metrics_oof = {}

        importance = model.feature_importance()

        # ---- 落盘 ----
        artifact_dir = self.settings.model_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model.save(artifact_dir / "model.joblib")
        data.builder.save(artifact_dir / "feature_builder.joblib")
        importance.rename("importance").to_csv(artifact_dir / "feature_importance.csv")

        # 测试集逐样本预测：容量分析（固定干预人力下的 top-K）与误差复盘都靠它。
        # OOF 预测一并落盘，代价比敏感性扫描要靠它重选阈值而无需重新训练。
        self._save_predictions(data, prob_test, prob_test_raw, artifact_dir)
        if oof_prob is not None:
            self._save_oof(data, oof_prob, artifact_dir)

        # 校准器产物：启用时落盘；未启用时清理旧文件，避免「重训后仍加载上一版校准器」
        calibrator_path = artifact_dir / "calibrator.joblib"
        if calibrator.method != "none":
            calibrator.save(calibrator_path)
        elif calibrator_path.exists():
            calibrator_path.unlink()

        metadata = {
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model.name,
            "model_description": model.describe(),
            "threshold": round(threshold, 4),
            "threshold_strategy": self.config.threshold_strategy,
            "cost_matrix": self.cost_matrix.to_dict(),
            "feature_names": data.builder.feature_names,
            "categorical_features": data.builder.categorical_features,
            "n_train": int(len(data.X_train)),
            "n_test": int(len(data.X_test)),
            "positive_rate_train": round(float(np.mean(data.y_train)), 4),
            "positive_rate_test": round(float(np.mean(data.y_test)), 4),
            "metrics_test": metrics_test,
            "metrics_test_at_0_5": metrics_default,
            "metrics_test_at_elkan": metrics_elkan,
            "metrics_test_uncalibrated": metrics_raw,
            "metrics_train_oof": metrics_oof,
            "calibration": calibrator.metadata(),
            "top_features": [
                {"feature": str(k), "importance": round(float(v), 4)}
                for k, v in importance.head(20).items()
            ],
            "train_config": asdict(self.config),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        reports_dir = self.settings.resolved_reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "test": metrics_test,
                    "test_at_0_5": metrics_default,
                    "test_at_elkan": metrics_elkan,
                    "test_uncalibrated": metrics_raw,
                    "train_oof": metrics_oof,
                    "calibration": calibrator.metadata(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "训练流水线完成",
            extra={
                "event": "training.completed",
                "model_name": model.name,
                "n_records": int(len(data.X_train) + len(data.X_test)),
                "threshold": round(threshold, 4),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

        return TrainResult(
            metrics=metadata,
            threshold=threshold,
            artifact_dir=artifact_dir,
            model=model,
            builder=data.builder,
            importance=importance,
        )
