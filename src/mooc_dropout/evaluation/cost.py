"""代价矩阵与代价最优决策阈值 —— 本课题的理论核心。

符号约定（二分类，1 = 辍学）：
    C_FN  漏报代价：学生辍学但模型未预警 => 流失损失（学费/口碑/续报）
    C_FP  误报代价：学生未辍学但被预警 => 无效干预成本（教师工时/短信/辅导）
    C_TP  正确预警仍需付出的干预成本（默认 0，可扩展为三态模型）
    C_TN  正确不预警的收益（默认 0，仅用于统一公式）

决策规则（最小期望代价，Elkan 1997 的推广形式）：
    预测 1 的期望代价: p·C_TP + (1-p)·C_FP
    预测 0 的期望代价: p·C_FN + (1-p)·C_TN
    令二者相等，得到最优阈值
        p* = (C_TN - C_FP) / (C_TP - C_FP - C_FN + C_TN)
    C_TP = C_TN = 0 时退化为经典形式 p* = C_FP / (C_FP + C_FN)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import CostMatrixError


@dataclass(frozen=True)
class CostMatrix:
    c_fn: float = 1000.0
    c_fp: float = 100.0
    c_tp: float = 0.0
    c_tn: float = 0.0

    def __post_init__(self) -> None:
        if self.c_fn <= 0 or self.c_fp <= 0:
            raise CostMatrixError("C_FN 与 C_FP 必须为正数")
        if self.c_tp < 0 or self.c_tn < 0:
            raise CostMatrixError("C_TP 与 C_TN 不能为负")

    # ---- 派生量 ----
    @property
    def ratio(self) -> float:
        """代价比 C_FN / C_FP，即正类应获得的权重。"""
        return self.c_fn / self.c_fp

    @property
    def elkan_threshold(self) -> float:
        """解析最优阈值 p*。"""
        denominator = self.c_tp - self.c_fp - self.c_fn + self.c_tn
        if abs(denominator) < 1e-12:
            raise CostMatrixError("代价矩阵退化，无法求解最优阈值")
        threshold = (self.c_tn - self.c_fp) / denominator
        return float(np.clip(threshold, 0.0, 1.0))

    def cost_of(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """逐样本代价向量。"""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true 与 y_pred 长度不一致")

        table = np.array(
            [
                [self.c_tn, self.c_fp],  # 真实 0：预测 0 / 1
                [self.c_fn, self.c_tp],  # 真实 1：预测 0 / 1
            ],
            dtype="float64",
        )
        return table[y_true, y_pred]

    def total_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self.cost_of(y_true, y_pred).sum())

    def per_student_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        costs = self.cost_of(y_true, y_pred)
        return float(costs.mean()) if costs.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "ratio": round(self.ratio, 4),
                "elkan_threshold": round(self.elkan_threshold, 4),
            }
        )
        return payload

    @classmethod
    def from_settings(cls, settings) -> "CostMatrix":
        return cls(c_fn=settings.cost_fn, c_fp=settings.cost_fp)


def expected_cost(y_true, y_prob, threshold: float, matrix: CostMatrix) -> float:
    """给定阈值下的总期望代价（单位：元）。"""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return matrix.total_cost(np.asarray(y_true), y_pred)


def cost_curve(
    y_true,
    y_prob,
    matrix: CostMatrix,
    *,
    n_points: int = 99,
) -> pd.DataFrame:
    """扫描阈值，返回 阈值-总代价 / 漏报数 / 误报数 曲线，用于论文制图。"""
    if n_points < 3:
        raise ValueError("n_points 至少为 3")
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")

    thresholds = np.linspace(0.01, 0.99, n_points)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "total_cost": matrix.c_fn * fn + matrix.c_fp * fp + matrix.c_tp * tp,
                "per_student_cost": (
                    matrix.c_fn * fn + matrix.c_fp * fp + matrix.c_tp * tp
                )
                / max(len(y_true), 1),
            }
        )
    return pd.DataFrame(rows)


def cost_optimal_threshold(
    y_true,
    y_prob,
    matrix: CostMatrix,
    *,
    n_points: int | None = None,
) -> dict[str, Any]:
    """在验证集（建议用 OOF 预测）上搜索使期望代价最小的阈值。

    默认用**精确搜索**而不是等距网格。原因来自真实数据上踩到的坑：原先固定用
    ``np.linspace(0.01, 0.99, 99)``（步长 0.01），当 C_FN/C_FP = 10 时最优阈值约在
    0.09 附近，网格在这一段只有 8 个候选点；更糟的是真实数据上代价曲线继续往左下降，
    最优解直接撞到 0.01 这个**网格边界**，看起来像「模型只会全报警」，实际是网格
    分辨率不够造成的假象。

    精确做法：阈值只会在观测到的概率值处改变预测集合，因此把样本按概率降序排列，
    对每个「前 k 个报警」计算代价，取最小者 —— O(n log n)，一次扫描即得全局最优，
    且在任意小的概率区间都能分辨。

    n_points 仅用于兼容旧调用（传了就回到网格搜索），正常路径不要传。

    返回 dict：threshold / total_cost / per_student_cost / elkan_threshold / n_alerted / curve
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true 与 y_prob 长度不一致")
    if y_true.size == 0:
        raise ValueError("评估集为空")

    if n_points is not None:  # 旧行为，保留给作图/回归
        curve = cost_curve(y_true, y_prob, matrix, n_points=n_points)
        best_idx = int(curve["total_cost"].idxmin())
        best = curve.loc[best_idx]
        return {
            "threshold": float(best["threshold"]),
            "total_cost": float(best["total_cost"]),
            "per_student_cost": float(best["per_student_cost"]),
            "elkan_threshold": matrix.elkan_threshold,
            "n_alerted": int(best["tp"] + best["fp"]),
            "curve": curve,
        }

    order = np.argsort(-y_prob, kind="mergesort")
    prob_sorted = y_prob[order]
    y_sorted = y_true[order]

    n = int(y_sorted.size)
    n_pos = int(y_sorted.sum())
    tp = np.concatenate(([0], np.cumsum(y_sorted)))
    k = np.arange(n + 1)
    fp = k - tp
    fn = n_pos - tp
    cost = matrix.c_fn * fn + matrix.c_fp * fp + matrix.c_tp * tp

    best_k = int(np.argmin(cost))
    if best_k == 0:
        threshold = float(np.nextafter(prob_sorted[0], np.inf))  # 高于最大概率 => 全放行
    elif best_k == n:
        threshold = float(np.nextafter(prob_sorted[-1], -np.inf))  # 低于最小概率 => 全报警
    else:
        threshold = float((prob_sorted[best_k - 1] + prob_sorted[best_k]) / 2.0)
    threshold = float(np.clip(threshold, 0.0, 1.0))

    return {
        "threshold": threshold,
        "total_cost": float(cost[best_k]),
        "per_student_cost": float(cost[best_k] / n),
        "elkan_threshold": matrix.elkan_threshold,
        "n_alerted": int(k[best_k]),
        "curve": None,
    }
