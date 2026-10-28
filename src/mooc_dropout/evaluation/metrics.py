"""评估指标：以「业务代价」为主、排序指标为辅。

只报 AUC 是这个课题最常见的答辩攻击点：AUC 高不代表漏报少、更不代表省钱。
因此这里同时输出 PR-AUC、Recall、期望代价与风险分级分布。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .cost import CostMatrix

RISK_LEVELS = ("high", "medium", "low")


def risk_level(probability: float, threshold: float) -> str:
    """风险分级：>= 决策阈值 => 高风险；[阈值/2, 阈值) => 中风险；其余低风险。

    分级直接锚定代价最优阈值，而不是拍脑袋的 0.6/0.3，
    保证「高风险」名单与「需要干预」的决策边界一致。
    """
    if threshold <= 0:
        raise ValueError("threshold 必须为正数")
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2.0:
        return "medium"
    return "low"


def summarize_risk_bands(levels: list[str]) -> dict[str, int]:
    counts = {level: 0 for level in RISK_LEVELS}
    for level in levels:
        counts[level] = counts.get(level, 0) + 1
    return counts


def classification_metrics(
    y_true,
    y_prob,
    *,
    threshold: float,
    cost_matrix: CostMatrix,
) -> dict[str, float]:
    """统一评估入口。所有指标均为 float，便于直接写 JSON。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true 与 y_prob 长度不一致")
    if y_true.size == 0:
        raise ValueError("评估集为空")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    single_class = len(np.unique(y_true)) < 2
    roc_auc = float("nan") if single_class else float(roc_auc_score(y_true, y_prob))
    pr_auc = float("nan") if single_class else float(average_precision_score(y_true, y_prob))

    total_cost = float(cost_matrix.c_fn * fn + cost_matrix.c_fp * fp + cost_matrix.c_tp * tp)
    levels = [risk_level(p, threshold) for p in y_prob]

    # ---- 平凡策略基线 ----
    # 只报绝对代价会掩盖一个致命问题：当正类比例超过代价最优阈值时，
    # 「给所有人都报警」本身就是代价最优的，模型再准也只能追平基线。
    # 因此必须同时给出两条平凡策略的代价，并用相对下降衡量模型的真实增益。
    n_pos = float(y_true.sum())
    n_neg = float(y_true.size - y_true.sum())
    cost_all_positive = (
        cost_matrix.c_fp * n_neg + cost_matrix.c_tp * n_pos
    ) / y_true.size
    cost_all_negative = cost_matrix.c_fn * n_pos / y_true.size
    cost_trivial = min(cost_all_positive, cost_all_negative)
    cost_per_student = total_cost / y_true.size
    reduction = (1.0 - cost_per_student / cost_trivial) if cost_trivial > 0 else float("nan")

    return {
        "threshold": round(float(threshold), 4),
        "n_samples": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else float("nan"),
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else float("nan"),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "total_cost": round(total_cost, 2),
        "cost_per_student": round(cost_per_student, 2),
        "baseline_cost_all_positive": round(cost_all_positive, 2),
        "baseline_cost_all_negative": round(cost_all_negative, 2),
        "baseline_cost_trivial": round(cost_trivial, 2),
        "cost_reduction_vs_trivial": round(float(reduction), 4),
        "predicted_positive_rate": round(float(y_pred.mean()), 4),
        "n_high_risk": levels.count("high"),
        "n_medium_risk": levels.count("medium"),
        "n_low_risk": levels.count("low"),
    }
