"""数据契约：KDD Cup 2015 字段定义与边界校验。

任何进入系统的数据都必须先过这一层，绝不信任外部输入。
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from ..errors import DataQualityError

ENROLLMENT_COLUMNS: Final[tuple[str, ...]] = (
    "enrollment_id",
    "username",
    "course_id",
    "dropout",
)

LOG_COLUMNS: Final[tuple[str, ...]] = (
    "enrollment_id",
    "time",
    "source",
    "event",
    "object",
)

# KDD Cup 2015 中的 7 类行为事件
KDD_EVENT_TYPES: Final[tuple[str, ...]] = (
    "access",
    "discussion",
    "navigate",
    "page_close",
    "problem",
    "video",
    "wiki",
)

# 官方数据里存在拼写错误与变体（实测 log_train.csv 中 "nagivate" 大量出现）。
# 不做归一化会直接触发「未知事件类型」而中断加载 —— 这是真实数据接入的必踩项。
EVENT_TYPE_ALIASES: Final[dict[str, str]] = {
    "nagivate": "navigate",
    "navigat": "navigate",
    "navigates": "navigate",
    "navigate_": "navigate",
    "pageclose": "page_close",
    "page-close": "page_close",
    "page close": "page_close",
    "problems": "problem",
    "videos": "video",
    "discussions": "discussion",
    "wiki_page": "wiki",
    "accesses": "access",
}

# 官方日志的 source 列只有这两种取值，真正的事件类型在 event 列
LOG_SOURCE_CHANNELS: Final[tuple[str, ...]] = ("browser", "server")


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(
            f"{name} 缺少必需列: {missing}",
            details={"required": list(required), "actual": list(df.columns)},
        )


def validate_enrollments(df: pd.DataFrame) -> pd.DataFrame:
    """校验选课表：字段齐全、ID 唯一、标签为 0/1。"""
    _require_columns(df, ENROLLMENT_COLUMNS, "选课表")
    out = df.loc[:, list(ENROLLMENT_COLUMNS)].copy()

    if out["enrollment_id"].isna().any():
        raise DataQualityError("选课表存在空的 enrollment_id")
    out["enrollment_id"] = out["enrollment_id"].astype(str)

    duplicated = out["enrollment_id"].duplicated()
    if duplicated.any():
        sample = out.loc[duplicated, "enrollment_id"].head(5).tolist()
        raise DataQualityError(
            f"选课表 enrollment_id 重复，示例: {sample}",
            details={"duplicated_count": int(duplicated.sum())},
        )

    label = pd.to_numeric(out["dropout"], errors="coerce")
    if label.isna().any():
        raise DataQualityError("dropout 列存在非数值，无法作为二分类标签")
    if not label.isin([0, 1]).all():
        raise DataQualityError("dropout 必须是 0/1 二值标签")
    out["dropout"] = label.astype("int8")

    out["username"] = out["username"].astype(str)
    out["course_id"] = out["course_id"].astype(str)
    return out.reset_index(drop=True)


def validate_logs(df: pd.DataFrame) -> pd.DataFrame:
    """校验行为日志表：字段齐全、时间可解析、事件类型合法。"""
    _require_columns(df, LOG_COLUMNS, "行为日志")
    out = df.loc[:, list(LOG_COLUMNS)].copy()

    out["enrollment_id"] = out["enrollment_id"].astype(str)
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    out["source"] = out["source"].replace(EVENT_TYPE_ALIASES)
    out["event"] = out["event"].astype(str)
    out["object"] = out["object"].astype(str)

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    if out["time"].isna().any():
        sample = out.index[out["time"].isna()][:5].tolist()
        raise DataQualityError(
            "行为日志存在无法解析的时间戳",
            details={"rows": [int(i) for i in sample]},
        )

    unknown = sorted(set(out["source"].unique()) - set(KDD_EVENT_TYPES))
    if unknown:
        raise DataQualityError(
            f"出现未知事件类型: {unknown}",
            details={"allowed": list(KDD_EVENT_TYPES)},
        )

    out = out.sort_values(["enrollment_id", "time"], kind="mergesort").reset_index(drop=True)
    return out


def positive_rate(enrollments: pd.DataFrame) -> float:
    return float(enrollments["dropout"].mean())
