"""评估指标与风险分级。"""

from __future__ import annotations

import numpy as np
import pytest

from mooc_dropout.evaluation.cost import CostMatrix
from mooc_dropout.evaluation.metrics import (
    classification_metrics,
    risk_level,
    summarize_risk_bands,
)


def test_perfect_prediction_has_zero_cost():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    metrics = classification_metrics(
        y_true, y_prob, threshold=0.5, cost_matrix=CostMatrix(1000, 100)
    )
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["total_cost"] == 0.0
    assert metrics["fn"] == 0 and metrics["fp"] == 0


def test_conservative_threshold_reduces_false_negatives():
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.55, 0.45, 0.35, 0.4, 0.3])
    matrix = CostMatrix(1000, 100)

    at_half = classification_metrics(y_true, y_prob, threshold=0.5, cost_matrix=matrix)
    at_low = classification_metrics(y_true, y_prob, threshold=0.3, cost_matrix=matrix)

    assert at_low["fn"] <= at_half["fn"]
    assert at_low["recall"] >= at_half["recall"]


def test_high_cost_matrix_penalises_false_negatives_more():
    y_true = np.array([1, 0])
    y_prob = np.array([0.1, 0.9])
    cheap = classification_metrics(y_true, y_prob, threshold=0.5, cost_matrix=CostMatrix(100, 100))
    expensive = classification_metrics(
        y_true, y_prob, threshold=0.5, cost_matrix=CostMatrix(1000, 10)
    )
    assert expensive["total_cost"] > cheap["total_cost"]


def test_single_class_auc_is_nan():
    metrics = classification_metrics(
        np.zeros(5), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), threshold=0.5, cost_matrix=CostMatrix()
    )
    assert np.isnan(metrics["roc_auc"])


def test_risk_level_bands_follow_threshold():
    threshold = 0.3
    assert risk_level(0.35, threshold) == "high"
    assert risk_level(0.3, threshold) == "high"
    assert risk_level(0.2, threshold) == "medium"
    assert risk_level(0.1, threshold) == "low"


def test_risk_level_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        risk_level(0.5, 0)


def test_summarize_risk_bands_counts_all_levels():
    counts = summarize_risk_bands(["high", "high", "low", "medium"])
    assert counts == {"high": 2, "medium": 1, "low": 1}


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        classification_metrics(np.array([1, 0]), np.array([0.5]), threshold=0.5, cost_matrix=CostMatrix())


# ---- 平凡策略基线（TM = trivial baseline）------------------------------------
# 这几条测试锁住一个容易被忽略的事实：绝对代价必须与「全报警/全放行」比较，
# 否则在正类占多数时，模型退化成全报警却看起来「代价很低」。

def test_trivial_baseline_all_positive_when_positive_is_majority():
    """正类占 80%：全报警的期望代价 = 0.2 * C_FP。"""
    y_true = np.array([1] * 8 + [0] * 2)
    y_prob = np.full(10, 0.9)
    metrics = classification_metrics(
        y_true, y_prob, threshold=0.5, cost_matrix=CostMatrix(1000, 100)
    )
    assert metrics["predicted_positive_rate"] == 1.0
    assert metrics["baseline_cost_all_positive"] == pytest.approx(20.0)
    assert metrics["baseline_cost_trivial"] == pytest.approx(20.0)
    # 模型退化成全报警 => 相对基线没有任何增益
    assert metrics["cost_reduction_vs_trivial"] == pytest.approx(0.0)


def test_trivial_baseline_all_negative_when_positive_is_minority():
    """正类占 20%：全放行的期望代价 = 0.2 * C_FN，小于全报警的 0.8 * C_FP。"""
    y_true = np.array([1] * 2 + [0] * 8)
    y_prob = np.full(10, 0.01)
    metrics = classification_metrics(
        y_true, y_prob, threshold=0.5, cost_matrix=CostMatrix(1000, 100)
    )
    assert metrics["baseline_cost_all_positive"] == pytest.approx(80.0)
    assert metrics["baseline_cost_all_negative"] == pytest.approx(200.0)
    assert metrics["baseline_cost_trivial"] == pytest.approx(80.0)


def test_good_model_has_positive_reduction_over_trivial():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    metrics = classification_metrics(
        y_true, y_prob, threshold=0.5, cost_matrix=CostMatrix(1000, 100)
    )
    assert metrics["cost_per_student"] == 0.0
    assert metrics["cost_reduction_vs_trivial"] == pytest.approx(1.0)


def test_metrics_expose_baseline_keys_for_every_call():
    """回归保护：下游 JSON/CSV 依赖这几个键，不能因为重构消失。"""
    metrics = classification_metrics(
        np.array([1, 0, 1, 0]), np.array([0.7, 0.4, 0.3, 0.6]),
        threshold=0.5, cost_matrix=CostMatrix(),
    )
    for key in (
        "baseline_cost_all_positive",
        "baseline_cost_all_negative",
        "baseline_cost_trivial",
        "cost_reduction_vs_trivial",
    ):
        assert key in metrics
