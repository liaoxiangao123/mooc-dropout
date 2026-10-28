"""派生「早期辍学」标签：把 79% 的官方标签改成有干预意义的少数类口径。

为什么需要重新定义标签：
    KDD Cup 2015 官方 dropout=1 表示「课程最后 10 天内无任何行为」，在训练集上占比
    79.3% —— 辍学是**多数类**，与「识别少数类辍学生」的表述根本不符。更严重的是，
    在 C_FN=1000 / C_FP=100 的代价矩阵下，「给所有人都报警」的人均代价只有约 20.7 元，
    与模型实际达到的代价几乎相等（实测 20.65 元），模型因此看起来毫无价值。

    真正有干预意义的问法是：**开课后两周内就停止学习的人是谁**。依据只有一个——
    老师在两周后是否还看得到这个学生。这个口径天然是少数类，且与 14 天观察窗的
    预测时点严格对齐：特征只用窗口内的事件，标签只看窗口之后还来不来。

标签定义（无穿越）：
    t0      = 该选课第一条行为事件的时间
    cut     = t0 + W 天（W 为观察窗）
    label   = 1  若 t_last <= cut（第 W 天后不再出现，早期辍学）
              0  若第 W 天后仍有行为（留存）
    特征只用 [t0, cut] 内的事件，标签只看 cut 之后的事件，两者时间上不重叠。

    对课程在 cut 之后很快结束的选课（观察期不足 min_remaining_days 天）予以剔除，
    否则会把「课已经上完了」误判成「学生辍学了」。

用法：
    python scripts/derive_early_dropout.py --data-dir data/raw/kdd2015
    python scripts/derive_early_dropout.py --window-days 21 --min-remaining-days 7

    默认会先扫描多个候选窗口并打印辍学率，便于按数据选择口径；加 --window-days 指定
    要落盘的那一个。

输出：
    data/processed/enrollment_early.csv   与 enrollment_train.csv 同构（dropout 已替换）
    reports/早期辍学标签-统计.md          口径、比例、多窗口对比、与官方标签的交叉表

内存策略：
    log_train.csv 有 1.1 GB / 816 万行，一次性读入会吃掉 2 GB 以上内存。这里按块
    流式读取，只保留 (enrollment_id -> t0/t_last) 与 (course_id -> end) 两张小表，
    峰值内存 ≈ 单块大小。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.data.loader import _find, _pick, read_enrollments  # noqa: E402

_TIME_FMT = "%Y-%m-%dT%H:%M:%S"


def scan_spans(log_path: Path, chunksize: int) -> tuple[pd.DataFrame, pd.Series]:
    """流式扫描日志，返回 (enrollment 的 t0/t_last, course 的最后事件时间)。"""
    per_enroll: list[pd.DataFrame] = []
    per_course: list[pd.Series] = []
    n_rows = 0
    n_bad_time = 0

    reader = pd.read_csv(
        log_path,
        usecols=["enrollment_id", "course_id", "time"],
        dtype={"enrollment_id": "int64", "course_id": "string", "time": "string"},
        chunksize=chunksize,
    )
    for i, chunk in enumerate(reader):
        n_rows += len(chunk)
        ts = pd.to_datetime(chunk["time"], format=_TIME_FMT, errors="coerce")
        n_bad_time += int(ts.isna().sum())
        chunk = chunk.assign(time=ts).dropna(subset=["time"])

        g = chunk.groupby("enrollment_id", sort=False)["time"]
        per_enroll.append(pd.DataFrame({"t0": g.min(), "t_last": g.max()}))
        per_course.append(chunk.groupby("course_id", sort=False)["time"].max())
        print(f"  块 {i + 1}: 累计 {n_rows:,} 行", end="\r", flush=True)

    print(" " * 40, end="\r")
    print(f"  日志 {n_rows:,} 行读取完成，时间解析失败 {n_bad_time:,} 行")

    enroll_span = pd.concat(per_enroll)
    enroll_span = enroll_span.groupby(level=0).agg({"t0": "min", "t_last": "max"})

    course_end = pd.concat(per_course).groupby(level=0).max()

    # 选课表里的 ID 在 validate_enrollments 里被统一成字符串，这里同步类型，
    # 否则 pandas 会拒绝 str 与 int64 的 merge（官方数据 ID 是纯数字，极易踩到）。
    enroll_span.index = enroll_span.index.astype(str)
    enroll_span.index.name = "enrollment_id"
    course_end.index = course_end.index.astype(str)
    course_end.index.name = "course_id"
    return enroll_span, course_end


def main() -> int:
    parser = argparse.ArgumentParser(description="派生早期辍学标签")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "kdd2015")
    parser.add_argument("--window-days", type=int, default=14, help="落盘用的观察窗长度（天）")
    parser.add_argument("--scan-windows", default="7,14,21,30",
                        help="候选窗口，逗号分隔；仅用于报告对比")
    parser.add_argument("--min-remaining-days", type=int, default=7,
                        help="观察窗结束后课程至少还要有这么多天，否则剔除该选课")
    parser.add_argument("--chunksize", type=int, default=1_500_000)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "processed" / "enrollment_early.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "早期辍学标签-统计.md")
    args = parser.parse_args()

    enroll_path = _pick(args.data_dir, ("enrollment_train.csv", "enrollments.csv", "enrollment.csv"), "选课")
    log_path = _pick(args.data_dir, ("log_train.csv", "logs.csv", "log.csv"), "日志")
    truth_path = _find(args.data_dir, ("truth_train.csv", "truth.csv", "truth_train.tsv"))

    print(f"读取选课表 {enroll_path.name} ...")
    enrollments = read_enrollments(enroll_path, truth_path)
    print(f"选课 {len(enrollments):,}，官方辍学率 {enrollments['dropout'].mean():.2%}")

    print(f"流式扫描日志 {log_path.name} ...")
    span, course_end = scan_spans(log_path, args.chunksize)
    print(f"有行为记录的选课 {len(span):,} / {len(enrollments):,}，课程 {len(course_end):,}")

    frame = enrollments.merge(span, left_on="enrollment_id", right_index=True, how="left")
    frame["course_end"] = frame["course_id"].map(course_end)

    no_event = int(frame["t0"].isna().sum())
    print(f"无任何行为记录的选课：{no_event:,}（无法定义 t0，剔除）")
    frame = frame.loc[frame["t0"].notna()].copy()

    windows = [int(w) for w in args.scan_windows.split(",") if w.strip()]
    if args.window_days not in windows:
        windows.append(args.window_days)
    windows = sorted(set(windows))

    rows = []
    for w in windows:
        cut = frame["t0"] + pd.Timedelta(days=w)
        remaining = (frame["course_end"] - cut).dt.total_seconds() / 86400.0
        usable = remaining >= args.min_remaining_days
        early = (frame["t_last"] <= cut).astype(int)
        rows.append({
            "window_days": w,
            "usable": int(usable.sum()),
            "dropped": int((~usable).sum()),
            "rate_usable": float(early[usable].mean()) if usable.any() else float("nan"),
            "rate_all": float(early.mean()),
        })
    scan = pd.DataFrame(rows)
    print("\n===== 多窗口扫描 =====")
    print(scan.to_string(index=False, float_format=lambda v: f"{v:.2%}"))

    # ---- 落盘选定的窗口 ----
    w = args.window_days
    cut = frame["t0"] + pd.Timedelta(days=w)
    remaining = (frame["course_end"] - cut).dt.total_seconds() / 86400.0
    usable = remaining >= args.min_remaining_days
    frame["early_dropout"] = (frame["t_last"] <= cut).astype(int)

    used = frame.loc[usable].copy()
    rate = float(used["early_dropout"].mean())
    crosstab = pd.crosstab(
        used["dropout"], used["early_dropout"],
        rownames=["官方 dropout"], colnames=["早期辍学"],
    )

    print(f"\n===== 落盘口径：观察窗 {w} 天，min_remaining {args.min_remaining_days} 天 =====")
    print(f"可用选课 {len(used):,} / {len(frame):,}（剔除观察期不足 {int((~usable).sum()):,}）")
    print(f"早期辍学率 {rate:.2%}（正类 {int(used['early_dropout'].sum()):,}）")
    print("\n与官方标签的交叉表：")
    print(crosstab.to_string())

    out_frame = used[["enrollment_id", "username", "course_id", "early_dropout"]].rename(
        columns={"early_dropout": "dropout"}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(args.out, index=False)
    print(f"\n已写出 {args.out}（{len(out_frame):,} 条，辍学率 {out_frame['dropout'].mean():.2%}）")

    # ---- 报告 ----
    n_all = len(frame)
    lines = [
        "# 早期辍学标签统计",
        "",
        f"- 数据：`{args.data_dir}`",
        f"- 选课总数：{len(enrollments):,}；有行为记录：{n_all:,}；无行为记录（剔除）：{no_event:,}",
        f"- 官方辍学率：{enrollments['dropout'].mean():.2%}（**多数类**，不适合作为「识别辍学生」的目标）",
        "",
        "## 标签定义",
        "",
        f"以每条选课的首条行为时间为 t0，第 {w} 天为 cut：",
        "",
        "- `dropout = 1`：cut 之后不再出现（早期辍学）",
        "- `dropout = 0`：cut 之后仍有行为（留存）",
        f"- 观察期保护：课程在 cut 之后至少还有 {args.min_remaining_days} 天，否则剔除",
        "",
        "特征只用 `[t0, cut]` 内的事件，标签只看 cut 之后的事件，两个时间区间不重叠，"
        "因此这不是循环论证，而是标准的「观察窗—预测窗」划分。",
        "",
        "## 多窗口扫描",
        "",
        "| 观察窗(天) | 可用选课 | 剔除 | 辍学率(可用) | 辍学率(全量) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        star = " ←采用" if r["window_days"] == w else ""
        lines.append(
            f"| {r['window_days']}{star} | {r['usable']:,} | {r['dropped']:,} | "
            f"{r['rate_usable']:.2%} | {r['rate_all']:.2%} |"
        )

    lines += [
        "",
        "窗口越短，被判定为「早期辍学」的人越少，但干预时点越早、可干预价值越高；"
        "窗口越长，标签越接近官方口径、正类比例越高。选择时兼顾两点：正类不能退化成"
        "多数类（否则代价矩阵失去意义），窗口也不能长到干预已经来不及。",
        "",
        "## 落盘口径与交叉表",
        "",
        f"- 采用窗口：**{w} 天**，min_remaining_days = {args.min_remaining_days}",
        f"- 可用选课：{len(used):,}；早期辍学率：**{rate:.2%}**",
        "",
        "```",
        crosstab.to_string(),
        "```",
        "",
        "交叉表读法：官方 dropout=0（拿到证书）却早期辍学的人，是「中途停了又回来」；"
        "官方 dropout=1 且早期辍学的人，是「一停就再没回来」——后者才是干预的真正目标。",
        "",
        "## 使用方式",
        "",
        "```bash",
        f"python scripts/derive_early_dropout.py --window-days {w}",
        "python scripts/train.py --data-dir data/raw/kdd2015 \\",
        "    --label-file data/processed/enrollment_early.csv",
        "```",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"统计报告 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
