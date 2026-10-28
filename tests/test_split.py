"""数据切分：必须保证同一学习者不跨训练/测试集。"""

from __future__ import annotations

import pytest

from mooc_dropout.data.split import group_train_test_split, oof_group_splits


def test_group_split_has_no_overlap(enrollments):
    train, test = group_train_test_split(enrollments, test_size=0.25, seed=1)
    assert set(train["username"]).isdisjoint(set(test["username"]))
    assert len(train) + len(test) == len(enrollments)


def test_group_split_ratio_reasonable(enrollments):
    train, test = group_train_test_split(enrollments, test_size=0.2, seed=3)
    ratio = len(test) / len(enrollments)
    assert 0.1 < ratio < 0.35


def test_both_splits_contain_both_classes(enrollments):
    train, test = group_train_test_split(enrollments, test_size=0.25, seed=5)
    assert train["dropout"].nunique() == 2
    assert test["dropout"].nunique() == 2


def test_invalid_test_size_raises(enrollments):
    with pytest.raises(ValueError):
        group_train_test_split(enrollments, test_size=0.0)


def test_oof_splits_cover_every_row_once(enrollments):
    splits = oof_group_splits(enrollments, n_splits=4, seed=11)
    seen: list[int] = []
    for train_idx, valid_idx in splits:
        seen.extend(valid_idx.tolist())
        assert set(enrollments.iloc[train_idx]["username"]).isdisjoint(
            set(enrollments.iloc[valid_idx]["username"])
        )
    assert sorted(seen) == list(range(len(enrollments)))
