"""数据加载与合成数据生成。

真实数据：KDD Cup 2015 (XuetangX) —— 需要自行下载后放到 data/raw/kdd2015/。
合成数据：字段与真实数据完全一致，用于无网络/无数据时的端到端联调与单测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DataQualityError
from ..logging_config import get_logger
from .schema import (
    EVENT_TYPE_ALIASES,
    KDD_EVENT_TYPES,
    LOG_SOURCE_CHANNELS,
    validate_enrollments,
    validate_logs,
)

logger = get_logger(__name__)

_ENROLLMENT_CANDIDATES = ("enrollment_train.csv", "enrollments.csv", "enrollment.csv")
_LOG_CANDIDATES = ("log_train.csv", "logs.csv", "log.csv")
_TRUTH_CANDIDATES = ("truth_train.csv", "truth.csv", "truth_train.tsv")

# 供 scripts/check_data.py 等外部工具做体检时的必需文件清单
REQUIRED_FILES = ("enrollment_train.csv", "log_train.csv")


def _find(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def _pick(root: Path, candidates: tuple[str, ...], kind: str) -> Path:
    path = _find(root, candidates)
    if path is not None:
        return path
    found = sorted(p.name for p in root.glob("*.csv"))
    raise DataQualityError(
        f"在 {root} 未找到{kind}文件，期望其中之一: {list(candidates)}",
        details={"found": found},
    )


def _read_truth(path: Path) -> pd.DataFrame:
    """读取 truth 文件并统一成 (enrollment_id, dropout) 两列。

    KDD Cup 2015 官方 truth_train.csv **没有表头**，直接 pd.read_csv 会把第一条
    记录当成列名（enrollment_id 变成 "E0000001"、1 变成 "1"）。这里先探测首行是否
    是表头，再决定怎么读 —— 这是真实数据接入时最容易踩的坑之一。
    """
    head = pd.read_csv(path, nrows=1, header=None, dtype=str)
    first_cell = str(head.iloc[0, 0]).strip().lower()
    has_header = "enrollment" in first_cell or "dropout" in first_cell

    if has_header:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, header=None, names=["enrollment_id", "dropout"])

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    label_col = next((c for c in ("dropout", "label", "truth") if c in frame.columns), None)
    if label_col is None:
        raise DataQualityError(
            f"truth 文件缺少标签列，实际列: {list(frame.columns)}",
            details={"path": str(path)},
        )
    return frame.rename(columns={label_col: "dropout"})[["enrollment_id", "dropout"]]


def read_enrollments(path: str | Path, truth_path: str | Path | None = None) -> pd.DataFrame:
    """读取选课表；若表中没有 dropout 列，自动从 truth 文件合并。"""
    path = Path(path)
    frame = pd.read_csv(path)

    if "dropout" not in frame.columns:
        if truth_path is None:
            raise DataQualityError(
                "选课表缺少 dropout 列，且未找到 truth 文件；"
                "请把 truth_train.csv 与选课表放在同一目录",
                details={"file": str(path), "columns": list(frame.columns)},
            )
        truth = _read_truth(Path(truth_path))
        before = len(frame)
        frame = frame.merge(truth, on="enrollment_id", how="left")
        missing_mask = frame["dropout"].isna()
        missing = int(missing_mask.sum())
        if missing:
            # 官方训练集里存在极少数「无标签」选课（如 enrollment_id 139669）。
            # 少量缺失属于数据本身的瑕疵，剔除并告警即可；大面积缺失说明文件配错，必须报错。
            ratio = missing / max(before, 1)
            if ratio > 0.01:
                raise DataQualityError(
                    f"有 {missing}/{before} 条选课在 truth 文件中找不到标签（{ratio:.1%}），"
                    "请确认下载的是同一批 train 文件",
                    details={"truth": str(truth_path)},
                )
            dropped = frame.loc[missing_mask, "enrollment_id"].astype(str).head(5).tolist()
            logger.warning(
                "剔除无标签选课",
                extra={
                    "event": "data.drop_unlabeled",
                    "n_records": missing,
                    "total": before,
                    "sample": dropped,
                },
            )
            print(f"  提示：{missing} 条选课在 truth 中无标签，已剔除（示例 {dropped}）")
            frame = frame.loc[~missing_mask]
        logger.info(
            "已合并 truth 标签",
            extra={"n_records": int(len(frame)), "event": "data.truth_merged"},
        )

    return validate_enrollments(frame)


def normalize_official_layout(frame: pd.DataFrame) -> pd.DataFrame:
    """把官方日志布局映射成本项目的字段语义。

    官方 `log_train.csv` 的列是：
        enrollment_id, username, course_id, time, source, event, object
    其中 `source` 只有 browser / server（事件通道），真正的事件类型在 `event` 列
    （access / discussion / navigate / page_close / problem / video / wiki）。

    本项目的 `source` 表示「事件类型」，因此这里把两列对调语义：
    `source` <- event 列（并归一化拼写错误），原通道值保留在 `channel` 列，
    校验时会被丢弃（特征不使用通道信息）。若数据已经是项目自己的口径
    （source 列本身就是事件类型），则原样返回，不做改动。
    """
    columns = {c.lower(): c for c in frame.columns}
    if "source" not in columns or "event" not in columns:
        return frame

    source_values = frame[columns["source"]].astype(str).str.strip().str.lower()
    event_values = (
        frame[columns["event"]].astype(str).str.strip().str.lower().replace(EVENT_TYPE_ALIASES)
    )
    is_channel = source_values.isin(LOG_SOURCE_CHANNELS).mean() > 0.9
    is_event_type = event_values.isin(KDD_EVENT_TYPES).mean() > 0.5
    if not (is_channel and is_event_type):
        return frame

    out = frame.copy()
    out["channel"] = source_values
    out["source"] = event_values
    out["event"] = event_values
    logger.info(
        "检测到官方日志布局，已映射 source<-event",
        extra={"event": "data.layout_normalized", "n_records": int(len(out))},
    )
    return out


def read_logs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_logs(normalize_official_layout(frame))


def load_kdd_cup(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 KDD Cup 2015 训练集，返回 (选课表, 日志表)。

    兼容两种目录组织：
        1. 已经合并好的 enrollment_train.csv（含 dropout 列）；
        2. 官方原始结构 —— enrollment_train.csv + truth_train.csv 分离。
    """
    root = Path(root)
    if not root.exists():
        raise DataQualityError(f"数据目录不存在: {root}")

    enrollment_path = _pick(root, _ENROLLMENT_CANDIDATES, "选课表")
    truth_path = _find(root, _TRUTH_CANDIDATES)
    enrollments = read_enrollments(enrollment_path, truth_path)
    logs = read_logs(_pick(root, _LOG_CANDIDATES, "行为日志"))

    unknown = set(logs["enrollment_id"].unique()) - set(enrollments["enrollment_id"])
    if unknown:
        logger.warning("日志中存在未登记选课记录", extra={"n_records": len(unknown)})

    logger.info(
        "KDD Cup 2015 数据加载完成",
        extra={"n_records": int(len(enrollments)), "event": "data.loaded"},
    )
    return enrollments, logs


# --------------------------------------------------------------------------
# 合成数据：仅用于开发联调与自动化测试，不作为论文实验结论来源
# --------------------------------------------------------------------------

def _simulate_student(
    rng: np.random.Generator,
    enrollment_id: str,
    course_id: str,
    username: str,
    is_dropout: bool,
    window_days: int,
    start_date: pd.Timestamp,
) -> list[dict]:
    """按「潜在风险 -> 行为强度」的生成过程模拟一个学生的学习行为。"""
    if is_dropout:
        # 辍学者：活跃天数少、强度衰减、常在中途停止
        active_days = int(rng.integers(1, max(2, window_days // 2 + 3)))
        stop_day = int(rng.integers(1, max(2, active_days + 1)))
        base_events = rng.gamma(2.0, 1.2)
        decay = rng.uniform(0.45, 0.9)
        p_video, p_problem, p_discuss = 0.30, 0.10, 0.02
    else:
        active_days = int(rng.integers(max(2, window_days // 2), window_days + 1))
        stop_day = window_days
        base_events = rng.gamma(4.5, 1.5)
        decay = rng.uniform(0.95, 1.05)
        p_video, p_problem, p_discuss = 0.36, 0.22, 0.06

    day_choices = np.sort(rng.choice(np.arange(min(active_days, stop_day + 1)),
                                     size=min(active_days, stop_day + 1), replace=False))
    rows: list[dict] = []
    for day in day_choices:
        weight = decay ** day
        n_events = max(1, int(rng.poisson(base_events * weight)))
        for _ in range(n_events):
            hour = int(np.clip(rng.normal(20 if rng.random() < 0.6 else 10, 3), 0, 23))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            ts = start_date + pd.Timedelta(days=int(day), hours=hour, minutes=minute, seconds=second)

            u = rng.random()
            if u < p_video:
                source, event = "video", "play_video"
                obj = f"video_{course_id}_{rng.integers(1, 41)}"
            elif u < p_video + p_problem:
                source, event = "problem", "problem_check"
                obj = f"problem_{course_id}_{rng.integers(1, 21)}"
            elif u < p_video + p_problem + p_discuss:
                source, event = "discussion", "add_entry"
                obj = f"forum_{course_id}_{rng.integers(1, 6)}"
            elif u < p_video + p_problem + p_discuss + 0.10:
                source, event = "access", "seq_next"
                obj = f"chapter_{course_id}_{rng.integers(1, 15)}"
            elif u < p_video + p_problem + p_discuss + 0.18:
                source, event = "wiki", "wiki_view"
                obj = f"wiki_{course_id}_{rng.integers(1, 5)}"
            elif u < p_video + p_problem + p_discuss + 0.45:
                source, event = "navigate", "nav_next"
                obj = f"chapter_{course_id}_{rng.integers(1, 15)}"
            else:
                source, event = "page_close", "page_close"
                obj = "page"

            rows.append(
                {
                    "enrollment_id": enrollment_id,
                    "time": ts,
                    "source": source,
                    "event": event,
                    "object": obj,
                }
            )
    return rows


def generate_synthetic_dataset(
    out_dir: str | Path,
    n_enrollments: int = 4000,
    n_courses: int = 20,
    n_users: int | None = None,
    dropout_rate: float = 0.18,
    window_days: int = 14,
    seed: int = 42,
) -> tuple[Path, Path]:
    """生成与 KDD Cup 2015 同构的合成数据集，写入 CSV 并返回 (选课表路径, 日志表路径)。

    说明：正类（辍学）比例由 dropout_rate 控制，默认 18%，
    用于模拟真实场景中的类别不平衡，便于验证代价敏感机制。
    """
    if not 0 < dropout_rate < 1:
        raise ValueError("dropout_rate 必须位于 (0, 1) 区间")

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_users = n_users or max(50, n_enrollments // 3)
    users = [f"u{idx:06d}" for idx in range(n_users)]
    courses = [f"C{idx:02d}" for idx in range(n_courses)]

    # 课程难度差异：不同课程的辍学率不同（更贴近真实 MOOC）
    course_bias = rng.normal(0.0, 0.35, size=n_courses)

    enrollment_rows: list[dict] = []
    log_rows: list[dict] = []
    base_start = pd.Timestamp("2014-01-06")

    for i in range(n_enrollments):
        enrollment_id = f"E{i:07d}"
        course_idx = int(rng.integers(0, n_courses))
        course_id = courses[course_idx]
        username = users[int(rng.integers(0, n_users))]

        latent = 1.0 / (1.0 + np.exp(-(course_bias[course_idx] + rng.normal(0, 0.8))))
        p_drop = float(np.clip(0.04 + 0.30 * latent, 0.02, 0.60))
        is_dropout = rng.random() < p_drop

        # 行为噪声：约 18% 的学生行为与标签不一致，模拟"沉默但坚持""活跃但放弃"，
        # 使合成数据具备真实数据的可分性难度（否则 AUC 轻易到 1.0，失去演示意义）
        behavior_dropout = is_dropout if rng.random() > 0.18 else (not is_dropout)
        # 标签噪声：约 4% 的标签被翻转
        label = is_dropout if rng.random() > 0.04 else (not is_dropout)

        start = base_start + pd.Timedelta(days=int(rng.integers(0, 120)))
        enrollment_rows.append(
            {
                "enrollment_id": enrollment_id,
                "username": username,
                "course_id": course_id,
                "dropout": int(label),
            }
        )
        log_rows.extend(
            _simulate_student(
                rng, enrollment_id, course_id, username, behavior_dropout, window_days, start
            )
        )

    enrollments = validate_enrollments(pd.DataFrame(enrollment_rows))
    logs = validate_logs(pd.DataFrame(log_rows))

    enrollment_path = out_dir / "enrollment_train.csv"
    log_path = out_dir / "log_train.csv"
    enrollments.to_csv(enrollment_path, index=False)
    logs.to_csv(log_path, index=False)

    logger.info(
        "合成数据集生成完成",
        extra={
            "n_records": int(len(enrollments)),
            "event": "data.synthetic_generated",
            "positive_rate": round(float(enrollments["dropout"].mean()), 4),
            "n_events": int(len(logs)),
        },
    )
    return enrollment_path, log_path
