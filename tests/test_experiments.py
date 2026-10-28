"""实验编排：校准集切分、结果表结构、Markdown 渲染。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mooc_dropout.training.experiments import ExperimentSpec, run_comparison, to_markdown


def test_comparison_returns_comparable_rows(test_settings, logs, enrollments):
    specs = [
        ExperimentSpec("neutral", "hist_gbdt", cost_ratio=1.0),
        ExperimentSpec("costly", "hist_gbdt", cost_ratio=10.0, weighting="none"),
    ]
    frame = run_comparison(
        logs,
        enrollments,
        settings=test_settings,
        specs=specs,
        model_params={"iterations": 40},
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(specs)
    for column in ("experiment", "roc_auc", "pr_auc", "recall", "f1", "cost_per_student", "threshold"):
        assert column in frame.columns
    assert (frame["cost_per_student"] >= 0).all()
    assert frame["threshold"].between(0, 1).all()
    assert frame["roc_auc"].between(0, 1).all()


def test_comparison_oof_alignment_with_numerically_sortable_ids(
    test_settings, logs, enrollments
):
    """回归：enrollment_id 字典序与行序不一致时，OOF 与标签必须仍对齐。

    真实 KDD Cup 数据 ID 是 "1","2",…,"10","11"，字典序把 "10" 排在 "2" 前面，
    按行号回填 OOF 会让阈值搜索在打乱标签上运行，症状是阈值塌到边界、
    预警比例≈1.0（等价于「给所有人报警」），而测试集 AUC 依旧很高。
    这里把 ID 重映射为不等宽整数串复现该条件。
    """
    rng = np.random.default_rng(0)
    ids = enrollments["enrollment_id"].astype(str).to_numpy()
    perm = rng.permutation(len(ids)) + 1
    remap = {old: str(new) for old, new in zip(ids, perm)}

    shuffled_enrollments = enrollments.copy()
    shuffled_enrollments["enrollment_id"] = shuffled_enrollments["enrollment_id"].astype(str).map(remap)
    shuffled_logs = logs.copy()
    shuffled_logs["enrollment_id"] = shuffled_logs["enrollment_id"].astype(str).map(remap)

    lex_sorted = sorted(shuffled_enrollments["enrollment_id"])
    numeric_sorted = sorted(shuffled_enrollments["enrollment_id"], key=int)
    assert lex_sorted != numeric_sorted, "构造失败：ID 字典序应与数值序不同"

    frame = run_comparison(
        shuffled_logs,
        shuffled_enrollments,
        settings=test_settings,
        specs=[ExperimentSpec("aligned", "hist_gbdt", cost_ratio=1.0)],
        model_params={"iterations": 40},
    )

    row = frame.iloc[0]
    assert row["roc_auc"] > 0.6
    # 错位时阈值会在乱标签上选到最低概率点，退化成「全员报警」
    assert row["predicted_positive_rate"] < 0.95
    assert 0.02 < row["threshold"] < 0.9


def test_markdown_contains_table():
    frame = pd.DataFrame(
        [{"experiment": "A", "roc_auc": 0.7, "pr_auc": 0.3, "recall": 0.6,
          "precision": 0.4, "f1": 0.48, "cost_per_student": 10.0,
          "cost_at_0.5": 12.0, "threshold": 0.3}]
    )
    text = to_markdown(frame, "测试表")
    assert "## 测试表" in text
    assert "| experiment |" in text
    assert "| A |" in text
