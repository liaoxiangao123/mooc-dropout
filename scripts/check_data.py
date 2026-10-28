"""数据验收：拿到 KDD Cup 2015 原始文件后，先跑这一条再训练。

为什么单独写一个脚本：
    真实数据最常见的三个坑是「文件缺一个」「编码/分隔符不对」「标签与选课对不上」，
    它们都会在训练跑到一半时才暴露，浪费十几分钟。这里在训练前一次性体检，
    并把关键分布打出来 —— 与论文 3.1 节的数据描述逐项对得上，才算数据可信。

用法：
    python scripts/check_data.py --data-dir data/raw/kdd2015
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.data.loader import REQUIRED_FILES, load_kdd_cup  # noqa: E402


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 KDD Cup 2015 数据是否可用于训练")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "kdd2015")
    args = parser.parse_args()
    data_dir: Path = args.data_dir

    print(f"数据目录：{data_dir}")
    if not data_dir.exists():
        print("目录不存在。请先下载数据并解压到该目录。")
        return 2

    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        print("缺少文件：" + "、".join(missing))
        print("需要 enrollment_train.csv / log_train.csv / truth_train.csv（可选 object.csv、date.csv）")
        return 2

    problems: list[str] = []
    try:
        enrollments, logs = load_kdd_cup(data_dir)
    except Exception as exc:  # noqa: BLE001 - 这里就是要展示原始错误
        print(f"加载失败：{type(exc).__name__}: {exc}")
        return 1

    n_enr, n_log = len(enrollments), len(logs)
    print("\n===== 规模 =====")
    print(f"选课记录：{_fmt_int(n_enr)}")
    print(f"行为事件：{_fmt_int(n_log)}")
    print(f"课程数  ：{enrollments['course_id'].nunique()}")
    print(f"学习者数：{enrollments['username'].nunique()}")
    print(f"平均每人事件数：{n_log / max(n_enr, 1):.1f}")

    print("\n===== 标签 =====")
    rate = float(enrollments["dropout"].mean())
    print(f"辍学率：{rate:.2%}（正例 {_fmt_int(int(enrollments['dropout'].sum()))}）")
    if not 0.05 <= rate <= 0.95:
        problems.append(f"辍学率 {rate:.2%} 异常，检查 truth 文件是否正确合并")
    per_course = enrollments.groupby("course_id")["dropout"].agg(["mean", "size"])
    print(
        f"各课程辍学率范围：{per_course['mean'].min():.1%} ~ {per_course['mean'].max():.1%}"
        f"（课程样本量 {int(per_course['size'].min())} ~ {int(per_course['size'].max())}）"
    )

    print("\n===== 时间跨度 =====")
    span = logs["time"].max() - logs["time"].min()
    print(f"{logs['time'].min()}  ~  {logs['time'].max()}（{span.days} 天）")

    print("\n===== 事件类型分布 =====")
    dist = logs["source"].value_counts()
    for name, count in dist.items():
        print(f"  {name:<12} {_fmt_int(int(count)):>12}  {count / n_log:.2%}")

    print("\n===== 一致性检查 =====")
    unknown = set(logs["enrollment_id"].unique()) - set(enrollments["enrollment_id"])
    if unknown:
        problems.append(f"有 {len(unknown)} 个 enrollment_id 出现在日志但不在选课表")
    no_log = set(enrollments["enrollment_id"]) - set(logs["enrollment_id"])
    if no_log:
        print(f"  提示：{len(no_log)} 条选课没有任何事件（特征全零，属正常噪声）")

    dup = int(enrollments["enrollment_id"].duplicated().sum())
    if dup:
        problems.append(f"选课表有 {dup} 条重复 enrollment_id")

    neg_gap = int((logs["time"] < logs["time"].min()).sum())
    if neg_gap:
        problems.append("存在非法时间戳")

    # 观察窗覆盖：每条选课在首事件后 14 天内是否有事件
    first = logs.groupby("enrollment_id")["time"].min().rename("t0")
    merged = logs.merge(first, on="enrollment_id")
    in_window = (merged["time"] - merged["t0"]).dt.days <= 14
    print(f"  观察窗内事件占比：{in_window.mean():.2%}（其余会被裁剪，属预期）")

    if problems:
        print("\n发现以下问题：")
        for item in problems:
            print(f"  ✗ {item}")
        return 1

    print("\n✓ 数据通过验收，可以开始训练：")
    print(f"  python scripts/train.py --data-dir {data_dir.as_posix()} --iterations 600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
