"""训练流水线：切分、产物落盘、元数据完整性、阈值合法性。"""

from __future__ import annotations

import json

import pytest

from mooc_dropout.evaluation.cost import CostMatrix
from mooc_dropout.training.trainer import TrainConfig, Trainer


def test_prepare_keeps_groups_disjoint(test_settings, logs, enrollments):
    trainer = Trainer(TrainConfig.from_settings(test_settings, model_name="hist_gbdt"), test_settings)
    data = trainer.prepare(logs, enrollments)

    assert set(data.train_enrollments["username"]).isdisjoint(
        set(data.test_enrollments["username"])
    )
    assert len(data.X_train) == len(data.y_train)
    assert len(data.X_test) == len(data.y_test)
    assert data.X_train.columns.tolist() == data.X_test.columns.tolist()


def test_training_writes_artifacts(trained):
    settings, result = trained
    model_dir = settings.model_dir

    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "feature_builder.joblib").exists()
    assert (model_dir / "metadata.json").exists()
    assert (settings.resolved_reports_dir / "metrics.json").exists()

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "model_version",
        "threshold",
        "cost_matrix",
        "feature_names",
        "metrics_test",
        "metrics_test_at_0_5",
        "top_features",
    ):
        assert key in metadata, key

    assert 0 < metadata["threshold"] < 1
    assert metadata["cost_matrix"]["ratio"] == pytest.approx(10.0)
    assert metadata["n_train"] > metadata["n_test"]


def test_result_threshold_matches_elkan_strategy(trained):
    settings, result = trained
    assert result.threshold == pytest.approx(CostMatrix(1000, 100).elkan_threshold, abs=1e-6)


def test_metrics_include_cost_and_auc(trained):
    _, result = trained
    metrics = result.metrics["metrics_test"]
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert metrics["cost_per_student"] >= 0
    assert metrics["n_high_risk"] + metrics["n_medium_risk"] + metrics["n_low_risk"] == metrics["n_samples"]


def test_oof_threshold_pipeline(test_settings, logs, enrollments):
    """启用 OOF 阈值搜索时也应正常收敛，且阈值落在合理区间。"""
    config = TrainConfig.from_settings(
        test_settings,
        model_name="hist_gbdt",
        n_splits=3,
        compute_oof=True,
        model_params={"max_iter": 40},
        threshold_strategy="oof_cost",
    )
    result = Trainer(config, test_settings).run(logs, enrollments)
    assert 0 < result.threshold < 1
    assert result.metrics["metrics_train_oof"]


def test_oof_predictions_align_when_enrollment_ids_sort_non_numerically(
    test_settings, logs, enrollments
):
    """回归：enrollment_id 不是等宽补零字符串时，OOF 必须按 ID 对齐到 X_train.index。

    真实数据（KDD Cup 2015）的 ID 是 "1"、"2"、…、"120542"，字典序为
    "1"、"10"、"100"、…，与 `train_enrollments` 的行序完全不同。旧实现按行号填 OOF，
    导致阈值搜索在一组错位的标签上运行：OOF AUC 只有 0.51，而测试集 AUC 有 0.82。
    这个测试把 ID 改成不补零的形式，锁住对齐逻辑。
    """
    from sklearn.metrics import roc_auc_score

    # 把补零 ID 换成不补零 ID，制造「字典序 != 数值序」
    mapping = {
        old: str(i + 1)
        for i, old in enumerate(sorted(enrollments["enrollment_id"].unique()))
    }
    raw_enr = enrollments.copy()
    raw_logs = logs.copy()
    raw_enr["enrollment_id"] = raw_enr["enrollment_id"].map(mapping)
    raw_logs["enrollment_id"] = raw_logs["enrollment_id"].map(mapping)

    trainer = Trainer(
        TrainConfig.from_settings(
            test_settings,
            model_name="hist_gbdt",
            compute_oof=True,
            n_splits=3,
            model_params={"max_iter": 60},
        ),
        test_settings,
    )
    data = trainer.prepare(raw_logs, raw_enr)
    oof = trainer._oof_probabilities(data)

    assert len(oof) == len(data.y_train)
    assert not any(v != v for v in oof)  # 无 NaN

    # 排序能力必须明显优于随机；错位时这个值会掉到 0.5 附近
    auc = roc_auc_score(data.y_train, oof)
    assert auc > 0.6, f"OOF 与标签疑似错位，AUC={auc:.3f}"
