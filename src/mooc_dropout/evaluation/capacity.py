"""容量分析：把「阈值决策」换成「限额干预」的评估口径。

为什么需要另一套口径：
    代价最优阈值只在「预警多少人就干预多少人」成立时可行。当正类比例高于 Elkan 阈值
    p\\* = C_FP/(C_FP+C_FN) 时，「给所有人都报警」既是代价最优又不可执行 —— 阈值视角
    在这类数据上直接失效（KDD Cup 2015 上实测如此，见 docs/11 第七节）。

    换一个提法就通了：教务处给出的不是阈值，而是人力。一学期最多能跟进 K 个人，
    决策就变成「按风险排序取前 K 个」。此时要回答的是三个可执行的问题：

        1. 名单开多大，才抓得住 80% 的辍学生？（召回 - 容量曲线）
        2. 人均期望代价随容量怎么变？谷底在哪？（代价 - 容量曲线）
        3. 前 K% 名单里的辍学生浓度是整体的几倍？（lift，衡量排序本身的价值）

    lift 与阈值无关，只取决于排序质量，因此它是「模型有没有用」最干净的度量：
    lift > 1 就说明模型把高危学生排到了前面，即使代价最优规则退化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cost import CostMatrix

DEFAULT_CAPACITIES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)


def capacity_curve(
    y_true,
    y_prob,
    matrix: CostMatrix,
    *,
    capacities: tuple[float, ...] = DEFAULT_CAPACITIES,
) -> pd.DataFrame:
    """按容量取 top-K，返回每个 K 下的召回/精确率/lift/人均代价。

    参数
    ----
    y_true : array-like of {0,1}
    y_prob : array-like of float，与 y_true 等长；不可为 NaN（校准器已保证）
    capacities : 容量比例，取值 (0, 1]，1.0 即「全报警」平凡策略

    返回的 DataFrame 带 ``attrs``：n / n_pos / base_rate / baseline_all_positive /
    baseline_all_negative，便于上层直接与平凡策略对照。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true 与 y_prob 长度不一致")
    if y_true.size == 0:
        raise ValueError("评估集为空")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob 含 NaN，无法排序")

    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    n = int(y_sorted.size)
    n_pos = int(y_sorted.sum())
    base_rate = n_pos / n
    cumulative_pos = np.cumsum(y_sorted)

    rows = []
    for capacity in capacities:
        if not 0 < capacity <= 1:
            raise ValueError(f"capacity 必须在 (0, 1] 内，收到 {capacity}")
        k = max(1, min(n, int(round(capacity * n))))
        tp = int(cumulative_pos[k - 1])
        fp = k - tp
        fn = n_pos - tp
        cost = matrix.c_fn * fn + matrix.c_fp * fp + matrix.c_tp * tp

        # 容量受限时唯一合法的基线是「同样开 K 个名额，但随机挑人」。
        # 与「全报警」比较没有意义：容量小于 100% 必然漏报更多，代价必然更高，
        # 那不是模型变差，而是约束变紧。随机基线把容量因素约掉，只留下排序的价值。
        tp_random = k * base_rate
        fp_random = k * (1.0 - base_rate)
        fn_random = n_pos - tp_random
        cost_random = (
            matrix.c_fn * fn_random + matrix.c_fp * fp_random + matrix.c_tp * tp_random
        )

        rows.append(
            {
                "capacity": float(capacity),
                "n_alerted": k,
                "threshold": float(p_sorted[k - 1]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": tp / n_pos if n_pos else float("nan"),
                "precision": tp / k,
                "lift": (tp / k) / base_rate if base_rate else float("nan"),
                "alert_rate": k / n,
                "cost_per_student": cost / n,
                "baseline_cost_random": cost_random / n,
                "cost_reduction_vs_random": (
                    1.0 - cost / cost_random if cost_random > 0 else float("nan")
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["n"] = n
    frame.attrs["n_pos"] = n_pos
    frame.attrs["base_rate"] = base_rate
    frame.attrs["baseline_all_positive"] = (
        matrix.c_fp * (n - n_pos) + matrix.c_tp * n_pos
    ) / n
    frame.attrs["baseline_all_negative"] = matrix.c_fn * n_pos / n
    return frame


def best_capacity(curve: pd.DataFrame) -> pd.Series:
    """返回人均代价最低的那一行（单调性不保证，直接取 argmin）。"""
    if curve.empty:
        raise ValueError("容量曲线为空")
    return curve.loc[curve["cost_per_student"].idxmin()]


def capacity_for_recall(curve: pd.DataFrame, target_recall: float) -> dict[str, float]:
    """给定召回目标，返回所需的最小容量及其精确率。

    这是教务处最常问的问题的正面回答：「想抓住 80% 的辍学生，名单要开多大」。
    """
    if not 0 < target_recall < 1:
        raise ValueError("target_recall 必须在 (0, 1) 内")
    hit = curve.loc[curve["recall"] >= target_recall]
    if hit.empty:
        # 容量列表没覆盖到，退化为用最后一行（全报警一定满足）
        row = curve.iloc[-1]
    else:
        row = hit.iloc[0]
    return {
        "capacity": float(row["capacity"]),
        "n_alerted": int(row["n_alerted"]),
        "recall": float(row["recall"]),
        "precision": float(row["precision"]),
        "lift": float(row["lift"]),
    }
