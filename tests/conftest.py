"""测试夹具：统一的合成数据与已训练产物，避免每个用例重复训练。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import Settings  # noqa: E402
from mooc_dropout.data.loader import generate_synthetic_dataset, load_kdd_cup  # noqa: E402
from mooc_dropout.training.trainer import TrainConfig, Trainer  # noqa: E402


@pytest.fixture(scope="session")
def raw_dataset(tmp_path_factory) -> tuple:
    data_dir = tmp_path_factory.mktemp("synthetic")
    generate_synthetic_dataset(
        data_dir,
        n_enrollments=600,
        n_courses=6,
        n_users=200,
        dropout_rate=0.25,
        window_days=14,
        seed=7,
    )
    return load_kdd_cup(data_dir)


@pytest.fixture(scope="session")
def enrollments(raw_dataset):
    return raw_dataset[0]


@pytest.fixture(scope="session")
def logs(raw_dataset):
    return raw_dataset[1]


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("artifacts")
    return Settings(
        env="test",
        log_level="WARNING",
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        reports_dir=root / "reports",
        cost_fn=1000.0,
        cost_fp=100.0,
        feature_window_days=14,
        decay_lambda=0.15,
        session_gap_minutes=30,
        random_seed=7,
    )


@pytest.fixture(scope="session")
def trained(test_settings, logs, enrollments):
    """训练一个轻量模型（HGBDT，跳过 OOF）供服务与接口测试复用。"""
    config = TrainConfig.from_settings(
        test_settings,
        model_name="hist_gbdt",
        compute_oof=False,
        threshold_strategy="elkan",
        model_params={"max_iter": 60},
    )
    result = Trainer(config, test_settings).run(logs, enrollments)
    return test_settings, result
