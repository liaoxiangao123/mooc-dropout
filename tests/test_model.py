"""模型层：概率合法性、代价敏感是否真的提高正类召回、持久化一致性。"""

from __future__ import annotations

import numpy as np
import pytest

from mooc_dropout.features.builder import FeatureBuilder
from mooc_dropout.models import MODEL_NAMES, build_model, is_available
from mooc_dropout.models.catboost_model import CATBOOST_AVAILABLE


@pytest.fixture(scope="module")
def xy(logs, enrollments):
    builder = FeatureBuilder().fit(logs)
    X = builder.transform(logs, enrollments)
    y = enrollments.set_index("enrollment_id")["dropout"].reindex(X.index).to_numpy()
    return X, y


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "catboost"])
def test_sklearn_models_produce_valid_probabilities(name, xy):
    X, y = xy
    model = build_model(name, categorical_features=["course_id", "username"])
    model.fit(X, y)
    prob = model.predict_proba(X)

    assert prob.shape == (len(X),)
    assert np.all((prob >= 0) & (prob <= 1))
    importance = model.feature_importance()
    # HistGradientBoosting 无原生特征重要性，允许为空；其余模型必须给出降序结果
    if name != "hist_gbdt":
        assert not importance.empty
        assert importance.is_monotonic_decreasing


def test_cost_sensitive_weighting_increases_positive_rate(xy):
    """同一模型下，给正类更高权重应提高预警率（Recall 上升、Precision 下降）。"""
    X, y = xy
    neutral = build_model("hist_gbdt", cost_ratio=1.0)
    neutral.fit(X, y)
    sensitive = build_model("hist_gbdt", cost_ratio=10.0)
    sensitive.fit(X, y)

    rate_neutral = float((neutral.predict_proba(X) >= 0.5).mean())
    rate_sensitive = float((sensitive.predict_proba(X) >= 0.5).mean())
    assert rate_sensitive >= rate_neutral


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("magic_forest")


def test_model_save_load_roundtrip(tmp_path, xy):
    X, y = xy
    model = build_model("logreg", categorical_features=["course_id", "username"])
    model.fit(X, y)
    path = model.save(tmp_path / "m.joblib")

    from mooc_dropout.models.base import BaseRiskModel

    loaded = BaseRiskModel.load(path)
    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X), rtol=1e-9)


def test_single_class_training_raises(xy):
    X, _ = xy
    with pytest.raises(ValueError):
        build_model("hist_gbdt").fit(X, np.zeros(len(X), dtype=int))


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="未安装 catboost")
class TestCatBoost:
    def test_fit_and_predict(self, xy):
        X, y = xy
        model = build_model(
            "catboost",
            iterations=40,
            depth=4,
            cost_ratio=10.0,
            categorical_features=["course_id", "username"],
        )
        model.fit(X, y)
        prob = model.predict_proba(X)

        assert prob.shape == (len(X),)
        assert np.all((prob >= 0) & (prob <= 1))
        importance = model.feature_importance()
        assert len(importance) == len(X.columns)

    def test_weighting_none_is_supported(self, xy):
        X, y = xy
        model = build_model(
            "catboost",
            iterations=30,
            depth=3,
            weighting="none",
            categorical_features=["course_id", "username"],
        )
        model.fit(X, y)
        assert model.predict_proba(X).shape == (len(X),)

    def test_invalid_weighting_raises(self):
        with pytest.raises(ValueError):
            build_model("catboost", weighting="magic")

    def test_availability_flag_matches(self):
        assert is_available("catboost") is CATBOOST_AVAILABLE
