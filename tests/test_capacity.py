"""容量分析：限额干预（top-K）下的召回、lift 与代价。"""

from __future__ import annotations

import numpy as np
import pytest

from mooc_dropout.evaluation.capacity import (
    best_capacity,
    capacity_curve,
    capacity_for_recall,
)
from mooc_dropout.evaluation.cost import CostMatrix


def _perfect(n_pos: int = 50, n_neg: int = 50) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([1] * n_pos + [0] * n_neg)
    p = np.concatenate([
        np.linspace(0.99, 0.51, n_pos),   # 正类概率全部偏高
        np.linspace(0.49, 0.01, n_neg),   # 负类概率全部偏低
    ])
    return y, p


def test_perfect_ranking_reaches_full_recall_at_capacity_equal_to_base_rate():
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(), capacities=(0.1, 0.5, 1.0))
    row = curve.loc[curve["capacity"] == 0.5].iloc[0]
    assert row["recall"] == 1.0
    assert row["precision"] == 1.0
    assert row["cost_per_student"] == 0.0


def test_full_capacity_equals_trivial_all_positive():
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(c_fn=1000, c_fp=100), capacities=(1.0,))
    row = curve.iloc[0]
    # 容量 100% 即「全报警」：精确率退化为基率，lift 为 1
    assert row["precision"] == pytest.approx(0.5)
    assert row["lift"] == pytest.approx(1.0)
    assert row["cost_per_student"] == pytest.approx(
        curve.attrs["baseline_all_positive"]
    )


def test_lift_decreases_towards_one_as_capacity_grows():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, size=2000)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, size=2000), 0.001, 0.999)
    curve = capacity_curve(y, p, CostMatrix(), capacities=(0.01, 0.1, 0.3, 1.0))
    assert curve["lift"].iloc[0] > curve["lift"].iloc[-1]
    assert curve["lift"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
    # 召回单调不减
    assert curve["recall"].is_monotonic_increasing


def test_attrs_carry_both_trivial_baselines():
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(c_fn=1000, c_fp=100), capacities=(0.5,))
    assert curve.attrs["n"] == 100
    assert curve.attrs["n_pos"] == 50
    assert curve.attrs["base_rate"] == pytest.approx(0.5)
    assert curve.attrs["baseline_all_positive"] == pytest.approx(50.0)
    assert curve.attrs["baseline_all_negative"] == pytest.approx(500.0)


def test_best_capacity_picks_minimum_cost():
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(c_fn=1000, c_fp=100), capacities=(0.1, 0.5, 1.0))
    best = best_capacity(curve)
    assert best["cost_per_student"] == curve["cost_per_student"].min()
    assert best["capacity"] == 0.5


def test_capacity_for_recall_reports_smallest_alert_list():
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(), capacities=(0.1, 0.5, 0.8, 1.0))
    hit = capacity_for_recall(curve, 0.8)
    assert hit["capacity"] == 0.5      # 基率 50% 才能覆盖 80% 召回
    assert hit["recall"] == 1.0


def test_capacity_rejects_out_of_range_and_bad_input():
    y, p = _perfect()
    with pytest.raises(ValueError):
        capacity_curve(y, p, CostMatrix(), capacities=(0.0,))
    with pytest.raises(ValueError):
        capacity_curve(y, p, CostMatrix(), capacities=(1.5,))
    with pytest.raises(ValueError):
        capacity_curve(y, p[:10], CostMatrix())
    bad = p.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        capacity_curve(y, bad, CostMatrix())


def test_capacity_for_recall_rejects_invalid_target():
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(), capacities=(0.5,))
    with pytest.raises(ValueError):
        capacity_for_recall(curve, 1.0)


def test_degenerate_all_positive_situation_has_lift_but_no_threshold_gain():
    """真实数据的困境：正类 80%，p* = 0.0909，几乎没人低于阈值。

    此时按阈值决策 = 全报警，代价等于平凡基线；但 top-K 视角下前 10% 仍然
    富集了高危学生（lift > 1），说明模型并非无用，只是阈值这个工具失效了。
    """
    rng = np.random.default_rng(3)
    n = 5000
    y = rng.binomial(1, 0.8, size=n)
    p = np.clip(0.8 + 0.15 * (y - 0.8) + rng.normal(0, 0.08, size=n), 0.001, 0.999)

    matrix = CostMatrix(c_fn=1000, c_fp=100)
    assert (p >= matrix.elkan_threshold).mean() > 0.99   # 阈值退化为全报警

    curve = capacity_curve(y, p, matrix, capacities=(0.05, 0.1, 1.0))
    assert curve["lift"].iloc[0] > 1.05
    assert curve["recall"].iloc[0] > 0.05


def test_random_baseline_equals_all_positive_at_full_capacity():
    """K=100% 时随机基线就是全报警，两者代价必须相等。"""
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(c_fn=1000, c_fp=100), capacities=(1.0,))
    row = curve.iloc[0]
    assert row["baseline_cost_random"] == pytest.approx(row["cost_per_student"])
    assert row["cost_reduction_vs_random"] == pytest.approx(0.0)


def test_cost_reduction_vs_random_is_positive_for_a_useful_ranking():
    """容量受限时，与「随机开同样长的名单」比才是有意义的基线。

    真实数据上（正类 79%）用「全报警」当基线会让 K<100% 的行全部显示为大负数，
    容易被误读成「模型让情况变糟」。随机基线把容量因素约掉。
    """
    y, p = _perfect()
    curve = capacity_curve(y, p, CostMatrix(c_fn=1000, c_fp=100), capacities=(0.1, 0.5, 1.0))
    assert (curve["cost_reduction_vs_random"] >= 0).all()
    # K=50% 恰好覆盖全部正类 => 代价为 0，相对随机下降 100%
    assert curve.loc[curve["capacity"] == 0.5, "cost_reduction_vs_random"].iloc[0] == pytest.approx(1.0)
    # K=10% 时虽然精确率 100%，但漏掉 40 名辍学生，代价仍高 => 相对随机只降 12%
    at_10 = curve.loc[curve["capacity"] == 0.1, "cost_reduction_vs_random"].iloc[0]
    assert 0.05 < at_10 < 0.3
