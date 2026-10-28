"""对比实验 / 消融实验 / 代价比敏感性分析，一键产出实验台账。

用法：
    python scripts/run_experiments.py --auto-synthetic --iterations 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import get_settings  # noqa: E402
from mooc_dropout.data.loader import (  # noqa: E402
    generate_synthetic_dataset,
    load_kdd_cup,
    read_enrollments,
)
from mooc_dropout.logging_config import configure_logging  # noqa: E402
from mooc_dropout.training.experiments import (  # noqa: E402
    run_comparison,
    run_cost_ratio_sensitivity,
    run_feature_ablation,
    save_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行对比/消融实验")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--label-file", type=Path, default=None, help="用指定选课表替换自动发现的标签（如早期辍学口径）")
    parser.add_argument("--auto-synthetic", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    reports = settings.resolved_reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir or (ROOT / "data" / "raw" / "kdd2015")
    if not (data_dir / "enrollment_train.csv").exists():
        if not args.auto_synthetic:
            print("未找到数据，请指定 --data-dir 或加 --auto-synthetic")
            return 2
        data_dir = ROOT / "data" / "synthetic"
        if not (data_dir / "enrollment_train.csv").exists():
            generate_synthetic_dataset(data_dir, n_enrollments=args.synthetic_n)

    enrollments, logs = load_kdd_cup(data_dir)
    if args.label_file:
        enrollments = read_enrollments(args.label_file)
        print(f"标签文件: {args.label_file}（{len(enrollments):,} 条，辍学率 {enrollments['dropout'].mean():.2%}）")
    params = {"iterations": args.iterations}

    table = run_comparison(logs, enrollments, settings=settings, model_params=params)
    path = save_experiment(table, reports, "exp_comparison", "基线对比与代价敏感消融")
    print(f"[1/3] 对比实验 -> {path}")
    print(
        table[
            [
                "experiment",
                "roc_auc",
                "pr_auc",
                "recall",
                "f1",
                "cost_per_student",
                "baseline_cost_trivial",
                "cost_reduction_vs_trivial",
            ]
        ].to_string(index=False)
    )

    if not args.skip_sensitivity:
        table2 = run_cost_ratio_sensitivity(
            logs, enrollments, settings=settings, model_params=params
        )
        path2 = save_experiment(table2, reports, "exp_cost_ratio", "代价比敏感性分析")
        print(f"\n[2/3] 敏感性分析 -> {path2}")
        print(table2[["experiment", "train_cost_ratio", "recall", "precision", "cost_per_student"]].to_string(index=False))

    if not args.skip_ablation:
        table3 = run_feature_ablation(logs, enrollments, settings=settings, model_params=params)
        path3 = save_experiment(table3, reports, "exp_feature_ablation", "趋势特征消融")
        print(f"\n[3/3] 特征消融 -> {path3}")
        print(table3[["experiment", "roc_auc", "pr_auc", "recall", "cost_per_student"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
