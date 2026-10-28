"""概率校准：单调性、尺度变换、阈值重选、产物加载。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mooc_dropout.models.calibration import ProbabilityCalibrator
from mooc_dropout.repository.artifact_repository import CALIBRATOR_FILE, ArtifactRepository
from mooc_dropout.training.trainer import TrainConfig, Trainer


def _toy_scores(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    # 加权模型的典型失真：正类概率被系统性放大
    prob = np.clip(0.25 + 0.45 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    return prob, y


def test_none_is_identity():
    cal = ProbabilityCalibrator("none")
    prob, _ = _toy_scores()
    assert np.allclose(cal.transform(prob), np.clip(prob, 0, 1))


def test_fit_requires_two_classes():
    cal = ProbabilityCalibrator("isotonic")
    with pytest.raises(ValueError):
        cal.fit(np.linspace(0.1, 0.9, 20), np.zeros(20, dtype=int))


def test_transform_before_fit_raises():
    from mooc_dropout.errors import ModelNotReadyError

    cal = ProbabilityCalibrator("platt")
    with pytest.raises(ModelNotReadyError):
        cal.transform(np.array([0.5]))


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_is_monotone(method):
    """单调变换是「不改排序、只改尺度」的前提；若排序被破坏，AUC 会变。"""
    prob, y = _toy_scores()
    cal = ProbabilityCalibrator(method).fit(prob, y)
    grid = np.linspace(0.01, 0.99, 200)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_platt_preserves_ranking():
    from sklearn.metrics import roc_auc_score

    prob, y = _toy_scores()
    cal = ProbabilityCalibrator("platt").fit(prob, y)
    assert roc_auc_score(y, cal.transform(prob)) == pytest.approx(
        roc_auc_score(y, prob), abs=1e-6
    )


def test_save_and_load(tmp_path):
    prob, y = _toy_scores()
    cal = ProbabilityCalibrator("isotonic").fit(prob, y)
    path = cal.save(tmp_path / "calibrator.joblib")
    loaded = ProbabilityCalibrator.load(path)
    assert loaded.method == "isotonic"
    assert np.allclose(loaded.transform(prob[:20]), cal.transform(prob[:20]))


def test_metadata_records_fit_scale():
    prob, y = _toy_scores()
    meta = ProbabilityCalibrator("isotonic").fit(prob, y).metadata()
    assert meta["method"] == "isotonic"
    assert meta["fitted"] is True
    assert meta["n_fit_samples"] == len(y)
    assert 0.0 <= meta["fitted_positive_rate"] <= 1.0


def test_trainer_with_calibration_reselects_threshold(test_settings, logs, enrollments, tmp_path):
    """启用校准后：阈值必须在校准尺度上重选，且产物里出现 calibrator。"""
    from mooc_dropout.config import Settings

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        cost_fn=1000.0,
        cost_fp=100.0,
        random_seed=7,
    )
    config = TrainConfig.from_settings(
        settings,
        model_name="hist_gbdt",
        n_splits=3,
        compute_oof=True,
        threshold_strategy="oof_cost",
        calibration="isotonic",
        model_params={"max_iter": 40},
    )
    result = Trainer(config, settings).run(logs, enrollments)

    assert (settings.model_dir / CALIBRATOR_FILE).exists()
    assert result.metrics["calibration"]["method"] == "isotonic"
    assert result.metrics["calibration"]["fitted"] is True
    # 校准前的指标单独留档，便于论文对比「校准前后」
    assert result.metrics["metrics_test_uncalibrated"]["brier"] > 0

    repository = ArtifactRepository(settings.model_dir)
    artifacts = repository.load()
    assert artifacts.is_calibrated
    assert artifacts.calibrator.method == "isotonic"

    metadata = json.loads((settings.model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["calibration"]["method"] == "isotonic"


def test_trainer_without_calibration_removes_stale_calibrator(test_settings, logs, enrollments, tmp_path):
    """重训时若关闭校准，必须删掉旧校准器，否则服务会拿旧尺度解释新概率。"""
    from mooc_dropout.config import Settings

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        cost_fn=1000.0,
        cost_fp=100.0,
        random_seed=7,
    )
    config = TrainConfig.from_settings(
        settings,
        model_name="hist_gbdt",
        compute_oof=False,
        threshold_strategy="elkan",
        model_params={"max_iter": 40},
        calibration="none",
    )
    stale = settings.model_dir / CALIBRATOR_FILE
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    Trainer(config, settings).run(logs, enrollments)

    assert not stale.exists()
    assert ArtifactRepository(settings.model_dir).load().is_calibrated is False
