"""对比实验与消融实验编排，一键产出台账（CSV + Markdown）。

实验矩阵：
    A. 基线对比：LR / RF / HGBDT / CatBoost（无加权）/ CatBoost（加权）
    B. 阈值消融：0.5 / Elkan 解析阈值 / OOF 代价最优阈值
    C. 代价比敏感性：ρ ∈ {1, 2, 5, 10, 20}
    D. 特征消融：去掉趋势衰减特征组
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.split import oof_group_splits
from ..errors import DataQualityError
from ..evaluation.cost import CostMatrix, cost_optimal_threshold
from ..evaluation.metrics import classification_metrics
from ..models.registry import build_model
from ..logging_config import get_logger
from .trainer import PreparedData, TrainConfig, Trainer

logger = get_logger(__name__)

_DECAY_GROUP = (
    "decayed_events",
    "decayed_active_days",
    "early_events",
    "late_events",
    "late_event_ratio",
    "activity_trend",
    "first_3day_events",
    "last_3day_events",
)


@dataclass
class ExperimentSpec:
    """单个实验配置。

    注意两个代价比是不同概念，混用会让不同实验行的代价不可比：
        cost_ratio  —— 训练时正类的权重比（模型是否做代价敏感训练）
        eval_ratio  —— 评估与决策时使用的业务代价比（决定代价矩阵与最优阈值）
    基线模型：cost_ratio=1（代价中立训练），eval_ratio 仍取业务值 10，这样各行代价才可比。
    """

    name: str
    model_name: str
    weighting: str = "none"
    cost_ratio: float | None = None
    eval_ratio: float | None = None
    drop_features: tuple[str, ...] = ()
    note: str = ""


DEFAULT_SPECS: tuple[ExperimentSpec, ...] = (
    # 基线：训练代价中立（cost_ratio=1），但评估仍按业务代价比 10:1，保证同行可比
    ExperimentSpec("LR (baseline)", "logreg", cost_ratio=1.0, note="代价中立训练，线性基线"),
    ExperimentSpec("RF (baseline)", "random_forest", cost_ratio=1.0, note="代价中立训练，装袋树基线"),
    ExperimentSpec("HGBDT (baseline)", "hist_gbdt", cost_ratio=1.0, note="代价中立训练，梯度提升基线"),
    ExperimentSpec("CatBoost (no cost)", "catboost", weighting="none", cost_ratio=1.0, note="消融：去掉代价敏感训练"),
    ExperimentSpec("CatBoost (cost-sensitive)", "catboost", weighting="class_weight", note="本文方法"),
)


def _params_for(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """按模型族裁剪超参，避免把 CatBoost 的 iterations 传给 sklearn。"""
    if model_name == "catboost":
        return dict(params)
    if model_name == "random_forest":
        return {"n_estimators": int(params.get("iterations", 300))}
    if model_name == "hist_gbdt":
        return {
            "max_iter": int(params.get("iterations", 300)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
        }
    return {}


def _evaluate_at(
    y_true: np.ndarray,
    prob: np.ndarray,
    matrix: CostMatrix,
    threshold: float,
) -> dict[str, Any]:
    return classification_metrics(y_true, prob, threshold=threshold, cost_matrix=matrix)


def run_comparison(
    logs: pd.DataFrame,
    enrollments: pd.DataFrame,
    *,
    settings: Settings | None = None,
    specs: Iterable[ExperimentSpec] = DEFAULT_SPECS,
    model_params: dict[str, Any] | None = None,
    threshold_thresholds: int | None = None,
) -> pd.DataFrame:
    """在同一数据切分上跑完整对比实验，返回逐实验一行的结果表。

    可比起见，所有实验共享同一份特征矩阵与训练/测试切分；
    阈值一律在训练集 3 折分组 OOF 预测上选择（不接触测试集），
    评估统一使用业务代价矩阵（eval_ratio，默认 10:1）。
    threshold_thresholds 默认 None 表示走精确阈值搜索（O(n log n)），
    仅在调试时传网格点数。
    """
    settings = settings or get_settings()
    base_config = TrainConfig.from_settings(
        settings, model_params=model_params or {}, n_splits=3
    )
    trainer = Trainer(base_config, settings)
    data = trainer.prepare(logs, enrollments)

    matrix = trainer.cost_matrix
    model_config_params = dict(model_params or {})
    label_map = data.train_enrollments.set_index("enrollment_id")["dropout"]
    folds = oof_group_splits(data.train_enrollments, n_splits=3, seed=settings.random_seed)

    rows: list[dict[str, Any]] = []

    for spec in specs:
        started = time.perf_counter()
        # 训练权重比与评估代价比分离
        train_ratio = spec.cost_ratio if spec.cost_ratio is not None else matrix.ratio
        eval_ratio = spec.eval_ratio if spec.eval_ratio is not None else matrix.ratio
        effective_matrix = CostMatrix(
            c_fn=eval_ratio * matrix.c_fp,
            c_fp=matrix.c_fp,
            c_tp=matrix.c_tp,
            c_tn=matrix.c_tn,
        )

        X_train, X_test = data.X_train, data.X_test
        if spec.drop_features:
            keep = [c for c in X_train.columns if c not in spec.drop_features]
            X_train, X_test = X_train[keep], X_test[keep]

        def _make_model(_spec=spec, _train_ratio=train_ratio):
            return build_model(
                _spec.model_name,
                cost_ratio=_train_ratio,
                weighting=_spec.weighting,
                random_seed=settings.random_seed,
                categorical_features=["course_id", "username"],
                **_params_for(_spec.model_name, model_config_params),
            )

        # ---- 1) 训练集 K 折 OOF，用于选阈值 ----
        # 对齐方式与 Trainer._oof_probabilities 一致：以 enrollment_id 为键收集，
        # 最后 reindex 到 X_train.index。若按 va_idx 直接写数组，在真实数据
        # （ID 为 "1","2",…，"10" 排在 "2" 之前）上会与 y_train 错位，
        # 阈值搜索就会在打乱的标签上进行——症状是 OOF AUC≈0.5 而测试集 AUC 很高。
        oof_series = pd.Series(
            np.nan,
            index=pd.Index(data.train_enrollments["enrollment_id"].astype(str).to_numpy()),
            dtype="float64",
        )
        for tr_idx, va_idx in folds:
            sub_train = data.train_enrollments.iloc[tr_idx]
            sub_valid = data.train_enrollments.iloc[va_idx]
            mask_tr = X_train.index.isin(set(sub_train["enrollment_id"]))
            mask_va = X_train.index.isin(set(sub_valid["enrollment_id"]))
            X_tr = X_train.loc[mask_tr]
            y_tr = label_map.reindex(X_tr.index).to_numpy()
            X_va = X_train.loc[mask_va]

            fold_model = _make_model()
            fold_model.fit(X_tr, y_tr)
            oof_series.loc[X_va.index.astype(str)] = fold_model.predict_proba(X_va)

        y_train = label_map.reindex(X_train.index).to_numpy()
        oof_prob = oof_series.reindex(pd.Index(X_train.index.astype(str))).to_numpy()
        missing = int(np.isnan(oof_prob).sum())
        if missing:
            raise DataQualityError(
                f"OOF 预测与训练集对齐后有 {missing} 条缺失，无法选阈值",
                details={"experiment": spec.name, "n_train": int(len(X_train))},
            )
        best = cost_optimal_threshold(
            y_train, oof_prob, effective_matrix, n_points=threshold_thresholds
        )
        threshold = float(best["threshold"])

        # ---- 2) 全量训练集重训，测试集只做最终评估 ----
        model = _make_model()
        model.fit(X_train, y_train)
        prob_test = model.predict_proba(X_test)

        metrics_opt = _evaluate_at(data.y_test, prob_test, effective_matrix, threshold)
        metrics_05 = _evaluate_at(data.y_test, prob_test, effective_matrix, 0.5)
        metrics_elkan = _evaluate_at(
            data.y_test, prob_test, effective_matrix, effective_matrix.elkan_threshold
        )

        rows.append(
            {
                "experiment": spec.name,
                "model": spec.model_name,
                "weighting": spec.weighting,
                "cost_fn": effective_matrix.c_fn,
                "cost_fp": effective_matrix.c_fp,
                "train_cost_ratio": round(train_ratio, 2),
                "eval_cost_ratio": round(eval_ratio, 2),
                "threshold": round(threshold, 4),
                "roc_auc": metrics_opt["roc_auc"],
                "pr_auc": metrics_opt["pr_auc"],
                "recall": metrics_opt["recall"],
                "precision": metrics_opt["precision"],
                "f1": metrics_opt["f1"],
                "brier": metrics_opt["brier"],
                "cost_per_student": metrics_opt["cost_per_student"],
                "baseline_cost_trivial": metrics_opt["baseline_cost_trivial"],
                "cost_reduction_vs_trivial": metrics_opt["cost_reduction_vs_trivial"],
                "predicted_positive_rate": metrics_opt["predicted_positive_rate"],
                "f1_at_0.5": metrics_05["f1"],
                "recall_at_0.5": metrics_05["recall"],
                "cost_at_0.5": metrics_05["cost_per_student"],
                "cost_at_elkan": metrics_elkan["cost_per_student"],
                "n_high_risk": metrics_opt["n_high_risk"],
                "note": spec.note,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        logger.info(
            "实验完成",
            extra={"event": "experiment.done", "model_name": spec.model_name, "n_records": len(prob_test)},
        )

    return pd.DataFrame(rows)


def run_cost_ratio_sensitivity(
    logs: pd.DataFrame,
    enrollments: pd.DataFrame,
    *,
    settings: Settings | None = None,
    ratios: Iterable[float] = (1, 2, 5, 10, 20),
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """代价比敏感性分析：ρ 变化时 Recall / 预警人数的变化趋势。"""
    specs = [
        ExperimentSpec(
            f"CatBoost rho={int(ratio)}",
            "catboost",
            weighting="class_weight",
            cost_ratio=float(ratio),
            eval_ratio=float(ratio),
            note=f"训练权重与评估代价比均为 C_FN/C_FP={ratio}",
        )
        for ratio in ratios
    ]
    return run_comparison(
        logs, enrollments, settings=settings, specs=specs, model_params=model_params
    )


def run_feature_ablation(
    logs: pd.DataFrame,
    enrollments: pd.DataFrame,
    *,
    settings: Settings | None = None,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """特征消融：去掉「趋势衰减」特征组，验证时序特征贡献。"""
    specs = [
        ExperimentSpec("CatBoost full features", "catboost", weighting="class_weight", note="全特征"),
        ExperimentSpec(
            "CatBoost w/o temporal decay",
            "catboost",
            weighting="class_weight",
            drop_features=_DECAY_GROUP,
            note="去掉 8 个趋势/衰减特征",
        ),
    ]
    frame = run_comparison(
        logs, enrollments, settings=settings, specs=specs, model_params=model_params
    )
    if len(frame) == 2:
        delta = frame.iloc[0]["roc_auc"] - frame.iloc[1]["roc_auc"]
        frame.loc[1, "note"] = f"{frame.loc[1, 'note']}；ΔAUC={delta:+.4f}"
    return frame


def to_markdown(frame: pd.DataFrame, title: str = "实验结果") -> str:
    columns = [
        "experiment", "roc_auc", "pr_auc", "recall", "precision", "f1",
        "cost_per_student", "cost_at_0.5", "threshold",
    ]
    view = frame[[c for c in columns if c in frame.columns]].copy()
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False)
    ]
    return "\n".join([f"## {title}", "", header, separator, *body, ""])


def save_experiment(frame: pd.DataFrame, out_dir: str | Path, name: str, title: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(to_markdown(frame, title), encoding="utf-8")
    return csv_path
