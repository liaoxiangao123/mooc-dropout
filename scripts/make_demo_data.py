"""生成演示用合成数据集（字段与 KDD Cup 2015 完全一致）。

用法：
    python scripts/make_demo_data.py --n 4000 --out data/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.data.loader import generate_synthetic_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 MOOC 行为合成数据")
    parser.add_argument("--n", type=int, default=4000, help="选课记录数")
    parser.add_argument("--courses", type=int, default=20, help="课程数")
    parser.add_argument("--dropout-rate", type=float, default=0.18, help="辍学比例")
    parser.add_argument("--window-days", type=int, default=14, help="行为观察窗长度")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    args = parser.parse_args()

    enrollment_path, log_path = generate_synthetic_dataset(
        args.out,
        n_enrollments=args.n,
        n_courses=args.courses,
        dropout_rate=args.dropout_rate,
        window_days=args.window_days,
        seed=args.seed,
    )
    print(f"选课表: {enrollment_path}")
    print(f"日志表: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
