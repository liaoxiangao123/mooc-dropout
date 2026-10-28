"""特征工程：契约一致性、防穿越、数值合法性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mooc_dropout.data.schema import validate_logs
from mooc_dropout.errors import DataQualityError, ValidationFailure
from mooc_dropout.features.behavior import summarize_events
from mooc_dropout.features.builder import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureBuilder,
)
from mooc_dropout.features.registry import feature_dictionary, spec_names


def test_registry_covers_all_columns():
    """代码产出的每一列都必须在特征字典中登记，避免文档与实现漂移。"""
    assert tuple(spec_names()) == ALL_FEATURES
    computed = summarize_events(
        pd.DataFrame(
            {
                "enrollment_id": ["E1", "E1", "E1"],
                "time": pd.to_datetime(["2014-03-01 10:00", "2014-03-01 10:05", "2014-03-02 21:00"]),
                "source": ["video", "problem", "discussion"],
                "event": ["play", "check", "post"],
                "object": ["v1", "p1", "f1"],
            }
        )
    )
    assert set(NUMERIC_FEATURES) == set(computed.columns)


def test_feature_dictionary_is_serialisable():
    items = feature_dictionary()
    assert len(items) == len(ALL_FEATURES)
    assert all({"name", "group", "description"} == set(item) for item in items)


def test_summarize_counts_events_within_window(logs):
    builder = FeatureBuilder()
    builder.fit(logs)
    sample_id = logs["enrollment_id"].iloc[0]
    subset = logs.loc[logs["enrollment_id"] == sample_id]
    features = builder.transform(subset)

    assert len(features) == 1
    assert features.iloc[0]["total_events"] == pytest.approx(len(subset))


def test_no_leakage_beyond_observation_window(logs):
    """观察窗之外的事件不得进入特征。"""
    sample_id = logs["enrollment_id"].iloc[0]
    subset = logs.loc[logs["enrollment_id"] == sample_id].copy().reset_index(drop=True)
    first_time = subset["time"].min()

    builder = FeatureBuilder()
    builder.fit(subset)
    baseline = builder.transform(subset).iloc[0]["total_events"]

    # 追加一条 60 天后的事件（远超 14 天窗口）
    late = subset.iloc[[0]].copy()
    late["time"] = first_time + pd.Timedelta(days=60)
    extended = pd.concat([subset, late], ignore_index=True)

    after = builder.transform(extended).iloc[0]["total_events"]
    assert after == baseline, "窗口外事件不应改变特征"


def test_numeric_features_are_finite_and_non_negative(logs):
    builder = FeatureBuilder().fit(logs)
    X = builder.transform(logs)

    assert X[list(NUMERIC_FEATURES)].isna().sum().sum() == 0
    assert np.isfinite(X[list(NUMERIC_FEATURES)].to_numpy()).all()
    count_like = [c for c in NUMERIC_FEATURES if c.startswith("n_") or c.endswith("_events")]
    assert (X[count_like] >= 0).all().all()


def test_ratio_features_bounded(logs):
    builder = FeatureBuilder().fit(logs)
    X = builder.transform(logs)
    for column in ("video_ratio", "problem_ratio", "discussion_ratio", "late_event_ratio", "active_day_ratio"):
        assert X[column].between(0, 1).all(), column


def test_transform_before_fit_raises(logs):
    with pytest.raises(ValidationFailure):
        FeatureBuilder().transform(logs)


def test_missing_required_column_raises(logs):
    broken = logs.drop(columns=["object"])
    with pytest.raises(DataQualityError):
        FeatureBuilder().fit(broken)


def test_unknown_event_type_raises(logs):
    broken = logs.copy()
    broken["source"] = broken["source"].astype(object)
    broken.loc[broken.index[0], "source"] = "hack"
    with pytest.raises(DataQualityError):
        validate_logs(broken)


def test_unseen_course_falls_back_to_unknown(logs):
    builder = FeatureBuilder().fit(logs)
    subset = logs.loc[logs["enrollment_id"] == logs["enrollment_id"].iloc[0]]
    meta = pd.DataFrame(
        [{"enrollment_id": subset["enrollment_id"].iloc[0], "course_id": "NEW_COURSE", "username": "new_user"}]
    )
    X = builder.transform(subset, meta)
    assert X.iloc[0]["course_id"] == "NEW_COURSE"
    assert X.iloc[0]["username"] == "new_user"


def test_single_row_matches_batch(logs):
    """单条推理与批量推理的特征必须一致，否则线上结果会漂移。"""
    builder = FeatureBuilder().fit(logs)
    batch = builder.transform(logs)
    sample_id = batch.index[0]
    single = builder.transform(logs.loc[logs["enrollment_id"] == sample_id])

    for column in NUMERIC_FEATURES:
        assert single.iloc[0][column] == pytest.approx(batch.loc[sample_id, column]), column


def test_empty_logs_raises():
    with pytest.raises(ValueError):
        summarize_events(pd.DataFrame(columns=["enrollment_id", "time", "source", "event", "object"]))
