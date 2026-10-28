"""训练主脚本：数据 -> 特征 -> 代价敏感 CatBoost -> 阈值 -> 产物落盘。

用法：
    # 用真实 KDD Cup 2015 数据
    python scripts/train.py --data-dir data/raw/kdd2015

    # 无数据时自动生成合成数据并训练（答辩演示推荐）
    python scripts/train.py --auto-synthetic
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
from mooc_dropout.training.trainer import TrainConfig, Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练代价敏感 CatBoost 辍学预测模型")
    parser.add_argument("--data-dir", type=Path, help="KDD Cup 2015 数据目录")
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="用指定的选课表替换自动发现的标签（如早期辍学口径）",
    )
    parser.add_argument("--auto-synthetic", action="store_true", help="数据缺失时自动生成合成数据")
    parser.add_argument("--synthetic-n", type=int, default=4000, help="合成数据选课记录数")
    parser.add_argument("--model", default="catboost", choices=["catboost", "logreg", "random_forest", "hist_gbdt"])
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-splits", type=int, default=5, help="OOF 折数（大数据集可降到 3）")
    parser.add_argument(
        "--weighting",
        default="class_weight",
        choices=["class_weight", "none"],
        help="class_weight=按代价比给正类加权；none=不加权（正类已是多数类时应选 none）",
    )
    parser.add_argument("--cost-fn", type=float, default=None)
    parser.add_argument("--cost-fp", type=float, default=None)
    parser.add_argument("--no-oof", action="store_true", help="跳过 OOF 阈值搜索（用 Elkan 解析阈值）")
    parser.add_argument(
        "--calibration",
        default="none",
        choices=["none", "platt", "isotonic"],
        help="概率校准方式（在 OOF 上拟合，阈值随之重选）",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="从 JSON 读取模型超参（scripts/tune.py 产出的 best_params.json）",
    )
    parser.add_argument(
        "--threshold-strategy",
        default=None,
        choices=["oof_cost", "elkan", "fixed"],
        help="决策阈值策略，默认 oof_cost",
    )
    return parser.parse_args()


def resolve_data(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.data_dir:
        return args.data_dir, False
    default_dir = ROOT / "data" / "raw" / "kdd2015"
    if (default_dir / "enrollment_train.csv").exists():
        return default_dir, False
    if args.auto_synthetic:
        return ROOT / "data" / "synthetic", True
    print("未找到数据。请用 --data-dir 指定 KDD Cup 2015 目录，或加 --auto-synthetic 生成演示数据。")
    raise SystemExit(2)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    data_dir, is_synthetic = resolve_data(args)
    if is_synthetic and not (data_dir / "enrollment_train.csv").exists():
        generate_synthetic_dataset(
            data_dir, n_enrollments=args.synthetic_n, window_days=settings.feature_window_days
        )

    enrollments, logs = load_kdd_cup(data_dir)
    if args.label_file:
        enrollments = read_enrollments(args.label_file)
        print(f"标签文件: {args.label_file}（{len(enrollments):,} 条）")
    label = "合成数据" if ("synthetic" in str(data_dir).lower() or is_synthetic) else "真实数据"
    print(f"数据: {data_dir}（{label}）")
    print(f"选课记录 {len(enrollments):,} 条，行为事件 {len(logs):,} 条，辍学率 {enrollments['dropout'].mean():.2%}")

    overrides = {
        "model_name": args.model,
        "model_params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        },
        "compute_oof": not args.no_oof,
        "calibration": args.calibration,
        "n_splits": args.n_splits,
        "weighting": args.weighting,
    }
    if args.params_json:
        import json

        tuned = json.loads(args.params_json.read_text(encoding="utf-8"))
        tuned = tuned.get("best_params", tuned)
        overrides["model_params"].update(tuned)
        print(f"已载入调参结果: {args.params_json} -> {tuned}")
    if args.cost_fn is not None:
        overrides["cost_fn"] = args.cost_fn
    if args.cost_fp is not None:
        overrides["cost_fp"] = args.cost_fp
    if args.threshold_strategy:
        overrides["threshold_strategy"] = args.threshold_strategy

    config = TrainConfig.from_settings(settings, **overrides)
    result = Trainer(config, settings).run(logs, enrollments)

    metrics = result.metrics["metrics_test"]
    print("\n===== 测试集结果（代价最优阈值）=====")
    print(f"阈值          : {result.threshold:.4f}")
    print(f"ROC-AUC       : {metrics['roc_auc']}")
    print(f"PR-AUC        : {metrics['pr_auc']}")
    print(f"Recall        : {metrics['recall']}   Precision: {metrics['precision']}   F1: {metrics['f1']}")
    print(f"漏报/误报     : {metrics['fn']} / {metrics['fp']}")
    print(f"人均代价      : ¥{metrics['cost_per_student']}")
    print(
        f"平凡基线      : 全报警 ¥{metrics['baseline_cost_all_positive']} / "
        f"全放行 ¥{metrics['baseline_cost_all_negative']} -> "
        f"相对下降 {metrics['cost_reduction_vs_trivial']:.2%}"
    )
    print(f"预测正类比例  : {metrics['predicted_positive_rate']:.2%}")
    print(f"高风险人数    : {metrics['n_high_risk']}")
    cal = result.metrics.get("calibration", {})
    if cal.get("method", "none") != "none":
        raw = result.metrics.get("metrics_test_uncalibrated", {})
        print(
            f"概率校准      : {cal.get('label')}（Brier {raw.get('brier')} -> {metrics.get('brier')}，"
            f"阈值已重选）"
        )
    print(f"\n模型产物: {result.artifact_dir}")
    print(f"指标报告: {settings.resolved_reports_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
