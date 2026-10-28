"""真实数据接入：truth 文件合并（官方文件无表头，最容易踩的坑）。"""

from __future__ import annotations

import pandas as pd
import pytest

from mooc_dropout.data.loader import load_kdd_cup, read_enrollments
from mooc_dropout.errors import DataQualityError


def _write_dataset(root, truth_text: str | None, with_label: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"enrollment_id": f"E{i:03d}", "username": f"u{i:03d}", "course_id": "C01"}
        for i in range(1, 6)
    ]
    if with_label:
        rows = [dict(r, dropout=i % 2) for i, r in enumerate(rows)]
    pd.DataFrame(rows).to_csv(root / "enrollment_train.csv", index=False)

    if truth_text is not None:
        (root / "truth_train.csv").write_text(truth_text, encoding="utf-8")

    logs = []
    for i in range(1, 6):
        for d in range(3):
            logs.append(
                {
                    "enrollment_id": f"E{i:03d}",
                    "time": f"2014-03-0{d + 1}T10:00:00",
                    "source": "video",
                    "event": "play_video",
                    "object": f"v{i}_{d}",
                }
            )
    pd.DataFrame(logs).to_csv(root / "log_train.csv", index=False)


def test_truth_without_header_is_merged(tmp_path):
    """官方 truth_train.csv 无表头：首条记录不能被当成列名丢掉。"""
    _write_dataset(
        tmp_path,
        truth_text="E001,1\nE002,0\nE003,1\nE004,0\nE005,1\n",
    )
    enrollments, _ = load_kdd_cup(tmp_path)

    assert len(enrollments) == 5
    assert enrollments.set_index("enrollment_id").loc["E001", "dropout"] == 1
    assert enrollments["dropout"].sum() == 3


def test_truth_with_header_is_merged(tmp_path):
    _write_dataset(
        tmp_path,
        truth_text="enrollment_id,dropout\nE001,0\nE002,1\nE003,0\nE004,1\nE005,0\n",
    )
    enrollments, _ = load_kdd_cup(tmp_path)
    assert len(enrollments) == 5
    assert enrollments.set_index("enrollment_id").loc["E002", "dropout"] == 1


def test_merged_enrollment_without_truth_file(tmp_path):
    """已合并过标签的选课表可以独立使用，无需 truth 文件。"""
    _write_dataset(tmp_path, truth_text=None, with_label=True)
    enrollments, logs = load_kdd_cup(tmp_path)
    assert len(enrollments) == 5
    assert not logs.empty


def test_missing_label_and_missing_truth_raises(tmp_path):
    _write_dataset(tmp_path, truth_text=None)
    with pytest.raises(DataQualityError, match="缺少 dropout"):
        load_kdd_cup(tmp_path)


def test_partial_truth_coverage_raises(tmp_path):
    """truth 大面积缺失（>1%）时必须报错，而不是静默产生 NaN 标签。"""
    _write_dataset(tmp_path, truth_text="E001,1\nE002,0\n")
    with pytest.raises(DataQualityError, match="找不到标签"):
        read_enrollments(tmp_path / "enrollment_train.csv", tmp_path / "truth_train.csv")


def test_small_fraction_unlabeled_is_dropped(tmp_path):
    """官方数据里存在极少数无标签选课（如 139669），应剔除而非中断训练。"""
    n = 300
    pd.DataFrame(
        {
            "enrollment_id": [f"E{i:04d}" for i in range(n)],
            "username": [f"u{i:04d}" for i in range(n)],
            "course_id": "C01",
        }
    ).to_csv(tmp_path / "enrollment_train.csv", index=False)
    # 299 条有标签，仅 E0007 缺失（0.33% < 1%）
    truth = "\n".join(
        f"E{i:04d},{i % 2}" for i in range(n) if i != 7
    ) + "\n"
    (tmp_path / "truth_train.csv").write_text(truth, encoding="utf-8")

    enrollments = read_enrollments(
        tmp_path / "enrollment_train.csv", tmp_path / "truth_train.csv"
    )
    assert len(enrollments) == n - 1
    assert "E0007" not in set(enrollments["enrollment_id"])
    assert enrollments["dropout"].notna().all()


def test_official_log_layout_is_mapped(tmp_path):
    """官方日志 source=browser/server、event=事件类型，必须自动对调语义。"""
    pd.DataFrame(
        {
            "enrollment_id": ["E001", "E001", "E002"],
            "username": ["u1", "u1", "u2"],
            "course_id": ["C01", "C01", "C01"],
            "time": ["2014-06-14T09:38:29", "2014-06-14T09:38:39", "2014-06-14T10:00:00"],
            "source": ["server", "browser", "server"],
            "event": ["nagivate", "access", "video"],
            "object": ["o1", "o2", "o3"],
        }
    ).to_csv(tmp_path / "log_train.csv", index=False)
    pd.DataFrame(
        {
            "enrollment_id": ["E001", "E002"],
            "username": ["u1", "u2"],
            "course_id": ["C01", "C01"],
            "dropout": [1, 0],
        }
    ).to_csv(tmp_path / "enrollment_train.csv", index=False)

    _, logs = load_kdd_cup(tmp_path)

    # 拼写错误 nagivate 应归一化为 navigate
    assert set(logs["source"]) == {"navigate", "access", "video"}
    assert not logs["source"].isin(["browser", "server"]).any()


def test_project_layout_is_left_untouched(tmp_path):
    """已经是本项目口径（source 就是事件类型）的数据不应被二次映射。"""
    pd.DataFrame(
        {
            "enrollment_id": ["E001"],
            "time": ["2014-06-14T09:38:29"],
            "source": ["video"],
            "event": ["play_video"],
            "object": ["o1"],
        }
    ).to_csv(tmp_path / "log_train.csv", index=False)
    pd.DataFrame(
        {
            "enrollment_id": ["E001"],
            "username": ["u1"],
            "course_id": ["C01"],
            "dropout": [1],
        }
    ).to_csv(tmp_path / "enrollment_train.csv", index=False)

    _, logs = load_kdd_cup(tmp_path)
    assert logs["source"].tolist() == ["video"]
    assert logs["event"].tolist() == ["play_video"]
