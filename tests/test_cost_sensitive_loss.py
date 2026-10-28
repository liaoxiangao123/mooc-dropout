"""代价敏感自定义损失的数学性质测试（不依赖 CatBoost）。"""

from __future__ import annotations

import numpy as np
import pytest

from mooc_dropout.models.cost_sensitive_loss import CostSensitiveLogloss


def test_sample_cost_assigns_by_label():
    loss = CostSensitiveLogloss(cost_fn=1000, cost_fp=100)
    costs = loss.sample_cost(np.array([1, 0, 1, 0]))
    assert costs.tolist() == [10.0, 1.0, 10.0, 1.0]
    assert loss.ratio == pytest.approx(10.0)


def test_gradient_sign_and_magnitude():
    loss = CostSensitiveLogloss(cost_fn=1000, cost_fp=100)
    # 概率 0.5 时，正类梯度为负（需要增大近似值），负类梯度为正
    grad, hess = loss.derivatives(np.array([0.0, 0.0]), np.array([1, 0]))
    assert grad[0] == pytest.approx(-0.5 * 10.0)
    assert grad[1] == pytest.approx(0.5 * 1.0)
    assert np.all(hess > 0)


def test_positive_class_gradient_scaled_by_cost_ratio():
    loss = CostSensitiveLogloss(cost_fn=500, cost_fp=100)
    grad, _ = loss.derivatives(np.array([0.0, 0.0]), np.array([1, 0]))
    assert abs(grad[0]) == pytest.approx(5.0 * abs(grad[1]))


def test_gradients_cancel_at_balanced_prior():
    """代价中立且正负各半时，0 近似值处的梯度应相互抵消。"""
    loss = CostSensitiveLogloss(cost_fn=100, cost_fp=100)
    grad, _ = loss.derivatives(np.zeros(100), np.array([1, 0] * 50))
    assert grad.sum() == pytest.approx(0.0, abs=1e-9)


def test_catboost_convention_is_negated():
    loss = CostSensitiveLogloss(cost_fn=1000, cost_fp=100)
    grad, hess = loss.derivatives(np.array([0.0]), np.array([1]))
    ders = loss.calc_ders_range([0.0], [1])
    assert ders[0][0] == pytest.approx(-grad[0])
    assert ders[0][1] == pytest.approx(-hess[0])


def test_derivatives_are_stable_for_extreme_logits():
    loss = CostSensitiveLogloss()
    grad, hess = loss.derivatives(np.array([-1e6, 1e6]), np.array([1, 0]))
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()
    assert np.all(hess >= 0)


def test_invalid_costs_rejected():
    with pytest.raises(ValueError):
        CostSensitiveLogloss(cost_fn=0, cost_fp=1)
