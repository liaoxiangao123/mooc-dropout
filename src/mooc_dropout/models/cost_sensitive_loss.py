"""代价敏感损失（探索性实现，路线 C）。

数学定义（逐样本代价系数 c_i）：
    c_i = C_FN  (y_i = 1)
    c_i = C_FP  (y_i = 0)
    L   = -(1/N) Σ c_i [ y_i log σ(a_i) + (1-y_i) log(1-σ(a_i)) ]

对近似值 a 的一阶 / 二阶导：
    dL/da = c_i (σ(a_i) - y_i)
    d²L/da² = c_i σ(a_i)(1-σ(a_i))

CatBoost 的 `calc_ders_range` 采用官方示例的符号约定（der1 = y - p, der2 = -p(1-p)），
其内部按 -Σder1 / Σder2 计算叶子值。两种约定只差一个负号，切勿混用。

使用注意：
    1. 使用本损失时**不能再设置 class_weights**，否则代价被重复计入；
    2. 首次在真实数据上使用，建议先用小样本验证 AUC > 0.5，以确认符号约定正确。
"""

from __future__ import annotations

import numpy as np

_SIGMOID_CLIP = 30.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_SIGMOID_CLIP, _SIGMOID_CLIP)))


class CostSensitiveLogloss:
    """可作为 `loss_function` 传给 CatBoostClassifier 的自定义目标。"""

    def __init__(self, cost_fn: float = 1000.0, cost_fp: float = 100.0) -> None:
        if cost_fn <= 0 or cost_fp <= 0:
            raise ValueError("C_FN 与 C_FP 必须为正数")
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)

    @property
    def ratio(self) -> float:
        return self.cost_fn / self.cost_fp

    def sample_cost(self, targets: np.ndarray) -> np.ndarray:
        """逐样本代价系数，按 C_FP 归一化以保持数值稳定。"""
        targets = np.asarray(targets).astype(float)
        return np.where(targets > 0.5, self.ratio, 1.0)

    # ---- 数学约定：返回 (dL/da, d²L/da²) ----
    def derivatives(self, approxes, targets) -> tuple[np.ndarray, np.ndarray]:
        approxes = np.asarray(approxes, dtype="float64").ravel()
        targets = np.asarray(targets, dtype="float64").ravel()
        prob = _sigmoid(approxes)
        cost = self.sample_cost(targets)
        grad = cost * (prob - targets)
        hess = cost * prob * (1.0 - prob)
        return grad, np.maximum(hess, 1e-12)

    # ---- CatBoost 约定：返回 (der1, der2) ----
    def calc_ders_range(self, approxes, targets, weights=None):
        grad, hess = self.derivatives(approxes, targets)
        der1 = -grad
        der2 = -hess
        if weights is not None:
            weights = np.asarray(weights, dtype="float64").ravel()
            der1 = der1 * weights
            der2 = der2 * weights
        return [(float(g), float(h)) for g, h in zip(der1, der2)]

    def __repr__(self) -> str:  # pragma: no cover
        return f"CostSensitiveLogloss(cost_fn={self.cost_fn}, cost_fp={self.cost_fp})"
