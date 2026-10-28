"""数据划分：以 uid（username）为分组，避免同一学生同时出现在训练集与测试集。

这是本课题最容易被忽视的泄漏点：同一用户的多条选课记录高度相关，
若按 enrollment 随机划分，指标会虚高 3~8 个点。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from ..errors import DataQualityError


def group_train_test_split(
    enrollments: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    group_col: str = "username",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按分组划分训练集/测试集，返回 (train, test)。

    若分组数过少（< 5），退化为按行随机分层划分，并在日志中标注。
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须位于 (0, 1) 区间")
    if group_col not in enrollments.columns:
        raise DataQualityError(f"缺少分组列 {group_col}")

    n_groups = enrollments[group_col].nunique()
    if n_groups < 5:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            enrollments,
            test_size=test_size,
            random_state=seed,
            stratify=enrollments["dropout"] if enrollments["dropout"].nunique() > 1 else None,
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(enrollments, groups=enrollments[group_col])
    )
    return (
        enrollments.iloc[train_idx].reset_index(drop=True),
        enrollments.iloc[test_idx].reset_index(drop=True),
    )


def oof_group_splits(
    enrollments: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "username",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """分组分层 K 折，用于生成 OOF 预测以选择决策阈值（避免阈值在测试集上调优）。"""
    min_class = int(enrollments["dropout"].value_counts().min())
    n_splits = max(2, min(n_splits, min_class))
    n_groups = enrollments[group_col].nunique()
    if n_groups < n_splits:
        n_splits = max(2, n_groups)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(
        cv.split(
            enrollments,
            y=enrollments["dropout"],
            groups=enrollments[group_col],
        )
    )
