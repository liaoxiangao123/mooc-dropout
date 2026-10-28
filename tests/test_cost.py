"""代价矩阵与代价最优阈值：本课题最核心的逻辑，必须重点覆盖。"""

from __future__ import annotations

import numpy as np
import pytest

from mooc_dropout.errors import CostMatrixError
from mooc_dropout.evaluation.cost import (
    CostMatrix,
    cost_curve,
    cost_optimal_threshold,
    expected_cost,
)


def test_elkan_threshold_matches_closed_form():
    matrix = CostMatrix(c_fn=1000, c_fp=100)
    assert matrix.ratio == pytest.approx(10.0)
    # p* = C_FP / (C_FP + C_FN) = 100/1100
    assert matrix.elkan_threshold == pytest.approx(100 / 1100, rel=1e-6)


def test_threshold_shifts_when_intervention_cost_added():
    """加入正确预警的干预成本后，阈值应上升（更谨慎地预警）。"""
    base = CostMatrix(c_fn=1000, c_fp=100)
    with_tp = CostMatrix(c_fn=1000, c_fp=100, c_tp=50)
    assert with_tp.elkan_threshold > base.elkan_threshold


def test_invalid_cost_matrix_raises():
    with pytest.raises(CostMatrixError):
        CostMatrix(c_fn=0, c_fp=100)
    with pytest.raises(CostMatrixError):
        CostMatrix(c_fn=1000, c_fp=-1)


def test_cost_of_is_asymmetric():
    matrix = CostMatrix(c_fn=1000, c_fp=100)
    y_true = np.array([1, 0])
    assert matrix.cost_of(y_true, np.array([0, 1])).tolist() == [1000.0, 100.0]
    assert matrix.cost_of(y_true, y_true).tolist() == [0.0, 0.0]


def test_total_and_per_student_cost():
    matrix = CostMatrix(c_fn=1000, c_fp=100)
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert matrix.total_cost(y_true, y_pred) == pytest.approx(1100.0)
    assert matrix.per_student_cost(y_true, y_pred) == pytest.approx(275.0)


def test_optimal_threshold_beats_default_on_expected_cost():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.2, size=4000)
    # 概率带有区分度但不完美
    y_prob = np.clip(0.15 + 0.55 * y_true + rng.normal(0, 0.18, size=4000), 0.001, 0.999)
    matrix = CostMatrix(c_fn=1000, c_fp=100)

    best = cost_optimal_threshold(y_true, y_prob, matrix)
    assert 0 < best["threshold"] < 1
    # 代价最优阈值的期望代价不应高于默认 0.5
    assert best["total_cost"] <= expected_cost(y_true, y_prob, 0.5, matrix)
    # 高代价比应当把阈值压到 0.5 以下
    assert best["threshold"] < 0.5


def test_cost_curve_columns_and_argmin():
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.2, 0.4, 0.6, 0.1])
    matrix = CostMatrix(c_fn=500, c_fp=50)
    curve = cost_curve(y_true, y_prob, matrix, n_points=21)

    assert {"threshold", "tp", "fp", "fn", "total_cost", "per_student_cost"} <= set(curve.columns)
    assert len(curve) == 21
    # 阈值升高，预警人数不应增加
    assert curve["tp"].is_monotonic_decreasing
    row = curve.loc[curve["total_cost"].idxmin()]
    assert row["total_cost"] == curve["total_cost"].min()


def test_cost_curve_rejects_too_few_points():
    with pytest.raises(ValueError):
        cost_curve([0, 1], [0.2, 0.8], CostMatrix(), n_points=2)


def test_from_settings():
    class _S:
        cost_fn = 800.0
        cost_fp = 200.0

    matrix = CostMatrix.from_settings(_S())
    assert matrix.c_fn == 800
    assert matrix.c_fp == 200
    assert matrix.to_dict()["ratio"] == pytest.approx(4.0)


def test_optimal_threshold_is_exact_below_grid_resolution():
    """回归：最优阈值可能落在 0.01 以下，旧的等距网格只能停在边界。

    真实 KDD Cup 数据上就出现过这种情况：所有概率都很低（模型区分度有限、
    基率高），旧网格 ``linspace(0.01, 0.99, 99)`` 的最优点撞在 0.01 上，
    看起来像「模型建议全报警」，实际只是网格分辨率不够。
    """
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.004, 0.005, 0.006, 0.007, 0.008,
                       0.001, 0.002, 0.0025, 0.003, 0.0035])
    matrix = CostMatrix(c_fn=1000, c_fp=100)

    best = cost_optimal_threshold(y_true, y_prob, matrix)

    assert best["threshold"] < 0.01          # 网格搜不到的区域
    assert best["total_cost"] == 0.0          # 完美切分
    assert best["n_alerted"] == 5

    grid = cost_optimal_threshold(y_true, y_prob, matrix, n_points=99)
    assert grid["total_cost"] > best["total_cost"]
