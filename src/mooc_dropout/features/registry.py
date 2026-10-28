"""特征字典：每个特征的分组与业务含义，供文档、/model/info 与论文表格共用。

约定：behavior.py 产出的每一列都必须在此登记，tests/test_features.py 会做一致性校验，
避免「代码加了特征、论文却没写」的漂移。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # ---- 基础活跃 ----
    FeatureSpec("total_events", "基础活跃", "观察窗内行为事件总数"),
    FeatureSpec("active_days", "基础活跃", "有行为记录的 distinct 天数"),
    FeatureSpec("unique_objects", "基础活跃", "触及的 distinct 学习对象数"),
    FeatureSpec("span_days", "基础活跃", "首个行为到最后一个行为的天数跨度"),
    FeatureSpec("days_since_last_event", "基础活跃", "距观察窗结束的天数（越大说明越早沉默）"),
    FeatureSpec("events_per_active_day", "基础活跃", "活跃日均事件数"),
    FeatureSpec("events_per_day", "基础活跃", "按窗口长度折算的日均事件数"),
    FeatureSpec("active_day_ratio", "基础活跃", "活跃天数占观察窗比例"),
    FeatureSpec("span_ratio", "基础活跃", "行为跨度占观察窗比例"),
    # ---- 分类型行为 ----
    FeatureSpec("n_video", "视频行为", "视频类事件数"),
    FeatureSpec("n_problem", "习题行为", "习题类事件数"),
    FeatureSpec("n_discussion", "论坛行为", "论坛类事件数"),
    FeatureSpec("n_navigate", "导航行为", "页面导航事件数"),
    FeatureSpec("n_page_close", "导航行为", "页面关闭事件数"),
    FeatureSpec("n_access", "导航行为", "课程访问事件数"),
    FeatureSpec("n_wiki", "导航行为", "wiki 浏览事件数"),
    FeatureSpec("video_ratio", "视频行为", "视频事件占全部事件比例"),
    FeatureSpec("problem_ratio", "习题行为", "习题事件占比"),
    FeatureSpec("discussion_ratio", "论坛行为", "论坛事件占比"),
    FeatureSpec("navigate_ratio", "导航行为", "导航事件占比"),
    # ---- 行为深度 ----
    FeatureSpec("video_unique_objects", "视频行为", "观看过的 distinct 视频数"),
    FeatureSpec("video_replay_ratio", "视频行为", "视频播放次数 / 去重视频数（重复观看强度）"),
    FeatureSpec("problem_unique_objects", "习题行为", "作答过的 distinct 题目数"),
    FeatureSpec("problem_repeat_ratio", "习题行为", "习题提交次数 / 去重题目数（反复尝试强度）"),
    FeatureSpec("discussion_unique_objects", "论坛行为", "参与的 distinct 讨论区/帖子数"),
    # ---- 时间节律 ----
    FeatureSpec("night_event_ratio", "时间节律", "22:00-06:00 事件占比"),
    FeatureSpec("weekend_event_ratio", "时间节律", "周末事件占比"),
    FeatureSpec("hour_entropy", "时间节律", "行为小时分布的香农熵，衡量学习节律规律性"),
    # ---- 会话结构 ----
    FeatureSpec("n_sessions", "会话结构", "按 30 分钟间隔切分的学习会话数"),
    FeatureSpec("mean_session_events", "会话结构", "平均每个会话的事件数"),
    FeatureSpec("max_session_events", "会话结构", "最长一次会话的事件数"),
    FeatureSpec("mean_gap_hours", "会话结构", "相邻事件平均间隔（小时）"),
    FeatureSpec("gap_std_hours", "会话结构", "相邻事件间隔标准差（小时）"),
    FeatureSpec("max_gap_days", "会话结构", "最长一次中断的天数"),
    # ---- 趋势与衰减 ----
    FeatureSpec("decayed_events", "趋势衰减", "按 exp(-λ·剩余天数) 加权后的事件量"),
    FeatureSpec("decayed_active_days", "趋势衰减", "按时间衰减加权后的活跃天数"),
    FeatureSpec("early_events", "趋势衰减", "窗口前半段事件数"),
    FeatureSpec("late_events", "趋势衰减", "窗口后半段事件数"),
    FeatureSpec("late_event_ratio", "趋势衰减", "后半段事件占比，衡量学习持续性"),
    FeatureSpec("activity_trend", "趋势衰减", "(后半 - 前半) / (后半 + 前半)，取值 [-1, 1]"),
    FeatureSpec("first_3day_events", "趋势衰减", "入学前 3 天事件数（冷启动信号）"),
    FeatureSpec("last_3day_events", "趋势衰减", "窗口最后 3 天事件数"),
    # ---- 类别标识 ----
    FeatureSpec("course_id", "类别标识", "课程 ID（CatBoost 原生类别特征）"),
    FeatureSpec("username", "类别标识", "学习者 ID（CatBoost 原生类别特征，高基数）"),
)


def feature_dictionary() -> list[dict[str, str]]:
    return [asdict(spec) for spec in FEATURE_SPECS]


def spec_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in FEATURE_SPECS)
