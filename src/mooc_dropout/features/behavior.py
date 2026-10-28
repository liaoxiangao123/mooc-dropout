"""行为特征聚合：从事件级日志汇总到「选课级」特征向量。

关键设计（防数据穿越）：
    每条选课记录以其首个事件的时刻作为 t0，只使用 [t0, t0 + window_days] 内的事件。
    标签来自课程结束时的最终状态，因此特征窗口严格早于标签时点。

实现上尽量使用向量化 groupby，避免对 12 万条选课做 apply（否则耗时以十分钟计）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import KDD_EVENT_TYPES

_SOURCE_SHORT = {
    "video": "video",
    "problem": "problem",
    "discussion": "discussion",
    "navigate": "navigate",
    "page_close": "page_close",
    "access": "access",
    "wiki": "wiki",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.astype("float64")
    den = denominator.astype("float64").replace(0, np.nan)
    return (num / den).fillna(0.0)


def summarize_events(
    logs: pd.DataFrame,
    *,
    window_days: int = 14,
    decay_lambda: float = 0.15,
    session_gap_minutes: int = 30,
) -> pd.DataFrame:
    """把事件日志聚合为选课级特征表（index 为 enrollment_id）。"""
    if logs.empty:
        raise ValueError("日志为空，无法构造特征")

    df = logs.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["enrollment_id"] = df["enrollment_id"].astype(str)
    df = df.sort_values(["enrollment_id", "time"], kind="mergesort").reset_index(drop=True)

    # ---- 观察窗裁剪 ----
    t0 = df.groupby("enrollment_id")["time"].transform("min")
    df["t0"] = t0
    window_end = t0 + pd.to_timedelta(window_days, unit="D")
    df = df.loc[df["time"] <= window_end].copy()

    df["elapsed_days"] = (df["time"] - df["t0"]).dt.total_seconds() / 86400.0
    df["day_idx"] = df["elapsed_days"].astype(int).clip(lower=0, upper=max(0, window_days - 1))
    df["date"] = df["time"].dt.normalize()
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.day_of_week

    by = df.groupby("enrollment_id", sort=True)
    feat = pd.DataFrame(index=by.size().index)
    feat.index.name = "enrollment_id"

    # ---- 基础活跃 ----
    feat["total_events"] = by.size().astype("float64")
    feat["active_days"] = by["date"].nunique().astype("float64")
    feat["unique_objects"] = by["object"].nunique().astype("float64")
    feat["span_days"] = (
        (by["time"].max() - by["time"].min()).dt.total_seconds() / 86400.0
    )
    group_window_end = by["t0"].first() + pd.to_timedelta(window_days, unit="D")
    feat["days_since_last_event"] = (
        (group_window_end - by["time"].max()).dt.total_seconds() / 86400.0
    )

    # ---- 分类型事件计数 ----
    src_counts = (
        df.groupby(["enrollment_id", "source"]).size().unstack(fill_value=0)
    )
    for source in KDD_EVENT_TYPES:
        column = f"n_{_SOURCE_SHORT[source]}"
        feat[column] = (
            src_counts[source].reindex(feat.index).fillna(0)
            if source in src_counts.columns
            else 0.0
        )

    # ---- 行为深度 ----
    depth = df.groupby(["enrollment_id", "source"])["object"].nunique().unstack(fill_value=0)
    for source, column in (
        ("video", "video_unique_objects"),
        ("problem", "problem_unique_objects"),
        ("discussion", "discussion_unique_objects"),
    ):
        feat[column] = (
            depth[source].reindex(feat.index).fillna(0)
            if source in depth.columns
            else 0.0
        )

    # ---- 时间节律 ----
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype("float64")
    df["is_weekend"] = (df["dow"] >= 5).astype("float64")
    feat["night_event_ratio"] = by["is_night"].mean()
    feat["weekend_event_ratio"] = by["is_weekend"].mean()

    hour_counts = (
        df.groupby(["enrollment_id", "hour"]).size().unstack(fill_value=0).astype("float64")
    )
    hour_prob = hour_counts.div(hour_counts.sum(axis=1).replace(0, np.nan), axis=0)
    feat["hour_entropy"] = (
        -(hour_prob * np.log(hour_prob.where(hour_prob > 0))).sum(axis=1).fillna(0.0)
    )

    # ---- 会话结构 ----
    gap_seconds = df.groupby("enrollment_id")["time"].diff().dt.total_seconds()
    df["gap_seconds"] = gap_seconds
    new_session = gap_seconds.isna() | (gap_seconds > session_gap_minutes * 60)
    df["session_id"] = new_session.groupby(df["enrollment_id"]).cumsum()

    session_size = df.groupby(["enrollment_id", "session_id"]).size()
    feat["n_sessions"] = session_size.groupby(level=0).size().reindex(feat.index).fillna(0.0)
    feat["mean_session_events"] = (
        session_size.groupby(level=0).mean().reindex(feat.index).fillna(0.0)
    )
    feat["max_session_events"] = (
        session_size.groupby(level=0).max().reindex(feat.index).fillna(0.0)
    )

    valid_gap = df.loc[df["gap_seconds"].notna()]
    gap_by = valid_gap.groupby("enrollment_id")["gap_seconds"]
    feat["mean_gap_hours"] = (gap_by.mean() / 3600.0).reindex(feat.index).fillna(0.0)
    feat["gap_std_hours"] = (gap_by.std(ddof=0) / 3600.0).reindex(feat.index).fillna(0.0)
    feat["max_gap_days"] = (
        (df.groupby("enrollment_id")["gap_seconds"].max() / 86400.0)
        .reindex(feat.index)
        .fillna(0.0)
    )

    # ---- 时间衰减 ----
    remaining = (window_days - df["elapsed_days"]).clip(lower=0.0)
    df["decay_weight"] = np.exp(-decay_lambda * remaining)
    feat["decayed_events"] = by["decay_weight"].sum()
    day_level = df.drop_duplicates(subset=["enrollment_id", "day_idx"])
    feat["decayed_active_days"] = (
        day_level.groupby("enrollment_id")["decay_weight"].sum().reindex(feat.index).fillna(0.0)
    )

    # ---- 前后半段趋势 ----
    half = window_days / 2.0
    early = df.loc[df["elapsed_days"] < half].groupby("enrollment_id").size()
    late = df.loc[df["elapsed_days"] >= half].groupby("enrollment_id").size()
    feat["early_events"] = early.reindex(feat.index).fillna(0.0)
    feat["late_events"] = late.reindex(feat.index).fillna(0.0)

    tail_start = max(0.0, window_days - 3.0)
    feat["first_3day_events"] = (
        df.loc[df["elapsed_days"] < 3.0].groupby("enrollment_id").size()
        .reindex(feat.index).fillna(0.0)
    )
    feat["last_3day_events"] = (
        df.loc[df["elapsed_days"] >= tail_start].groupby("enrollment_id").size()
        .reindex(feat.index).fillna(0.0)
    )

    # ---- 派生比率 ----
    window = float(window_days)
    feat["events_per_active_day"] = _safe_ratio(feat["total_events"], feat["active_days"])
    feat["events_per_day"] = feat["total_events"] / window
    feat["active_day_ratio"] = feat["active_days"] / window
    feat["span_ratio"] = feat["span_days"] / window
    feat["video_ratio"] = _safe_ratio(feat["n_video"], feat["total_events"])
    feat["problem_ratio"] = _safe_ratio(feat["n_problem"], feat["total_events"])
    feat["discussion_ratio"] = _safe_ratio(feat["n_discussion"], feat["total_events"])
    feat["navigate_ratio"] = _safe_ratio(feat["n_navigate"], feat["total_events"])
    feat["video_replay_ratio"] = _safe_ratio(
        feat["n_video"], feat["video_unique_objects"]
    )
    feat["problem_repeat_ratio"] = _safe_ratio(
        feat["n_problem"], feat["problem_unique_objects"]
    )

    total_window_events = feat["early_events"] + feat["late_events"]
    feat["late_event_ratio"] = _safe_ratio(feat["late_events"], total_window_events)
    feat["activity_trend"] = (
        (feat["late_events"] - feat["early_events"])
        / (total_window_events + 1.0)
    )

    feat = feat.fillna(0.0).astype("float64")
    return feat
