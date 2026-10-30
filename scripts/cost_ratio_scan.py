"""代价比敏感性扫描：找出「模型还能说话」的代价区间。

背景（真实数据上踩到的坑）：
    在 KDD Cup 2015 上，辍学率 79.3%，C_FN=1000 / C_FP=100（ρ=10）时，Elkan 阈值为
    0.0909。模型校准后的概率几乎全部高于 0.0909，于是代价最优决策退化为「给所有人
    报警」，人均代价恰好等于平凡基线（全报警）的 20.65 元，模型看起来毫无价值。
    换成早期辍学标签（正类 67.6%）后，同一现象仍然存在。

    这不是模型失败，而是**代价参数与业务基率不匹配**：当漏报代价是误报的 10 倍、
    而 8 成学生都会辍学时，「预警」在经济上等于免费，最优策略当然是人人都报。

    要判断模型在什么条件下真正有价值，必须扫描代价比 ρ，观察：
        - 阈值 p* = 1/(1+ρ) 如何移动；
        - 预测正类比例如何从 1.0 掉下来（退化解消失）；
        - 相对平凡基线的代价下降何时变为正数。

用法：
    python scripts/cost_ratio_scan.py --run artifacts/model_realdata_weighted
    python scripts/cost_ratio_scan.py --auto

输入：模型目录下的 predictions_oof.csv（选阈值）与 predictions_test.csv（评估）。
      两者都由 Trainer 落盘，因此本脚本**不需要重新训练**。
输出：reports/代价比扫描.md、reports/cost_ratio_scan.csv、docs/figs/fig5-6-cost-ratio-real.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.evaluation.cost import CostMatrix, cost_optimal_threshold  # noqa: E402
from mooc_dropout.evaluation.metrics import classification_metrics  # noqa: E402

DEFAULT_RATIOS = (1, 2, 3, 5, 8, 10, 20, 50)


def scan(run_dir: Path, ratios: tuple[int, ...], c_fp: float) -> pd.DataFrame:
    oof = pd.read_csv(run_dir / "predictions_oof.csv")
    test = pd.read_csv(run_dir / "predictions_test.csv")

    rows = []
    for rho in ratios:
        matrix = CostMatrix(c_fn=c_fp * rho, c_fp=c_fp)
        threshold = float(
            cost_optimal_threshold(oof["y_true"].to_numpy(), oof["probability"].to_numpy(), matrix)["threshold"]
        )
        metrics = classification_metrics(
            test["y_true"].to_numpy(), test["probability"].to_numpy(),
            threshold=threshold, cost_matrix=matrix,
        )
        rows.append({
            "cost_ratio": rho,
            "c_fn": matrix.c_fn,
            "elkan_threshold": round(matrix.elkan_threshold, 4),
            "selected_threshold": round(threshold, 4),
            "predicted_positive_rate": metrics["predicted_positive_rate"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "cost_per_student": metrics["cost_per_student"],
            "baseline_cost_trivial": metrics["baseline_cost_trivial"],
            "cost_reduction_vs_trivial": metrics["cost_reduction_vs_trivial"],
            "degenerate": metrics["predicted_positive_rate"] >= 0.999,
        })
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, name: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x = table["cost_ratio"]

    axes[0].plot(x, table["predicted_positive_rate"], marker="o", color="#2f5f8f", label="预测正类比例")
    axes[0].plot(x, table["recall"], marker="s", ms=4, color="#c0392b", label="召回率")
    axes[0].plot(x, table["precision"], marker="^", ms=4, color="#27ae60", label="精确率")
    axes[0].axhline(1.0, color="#888888", linestyle=":", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("代价比 ρ = C_FN / C_FP")
    axes[0].set_ylabel("比例")
    axes[0].set_title("预测正类比例随代价比的变化")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, table["cost_per_student"], marker="o", color="#2f5f8f", label="模型人均代价")
    axes[1].plot(x, table["baseline_cost_trivial"], marker="s", ms=4, linestyle="--",
                 color="#888888", label="平凡基线")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("代价比 ρ = C_FN / C_FP")
    axes[1].set_ylabel("人均期望代价（元）")
    axes[1].set_title("模型代价 vs 平凡基线")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="代价比敏感性扫描")
    parser.add_argument("--run", action="append", default=[], metavar="模型目录")
    parser.add_argument("--name", default="真实数据", help="报告标题里的数据口径名")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--ratios", default=",".join(str(r) for r in DEFAULT_RATIOS))
    parser.add_argument("--cost-fp", type=float, default=100.0)
    parser.add_argument("--out-md", type=Path, default=ROOT / "reports" / "代价比扫描.md")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "reports" / "cost_ratio_scan.csv")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "docs" / "figs" / "fig5-6-cost-ratio-real.png")
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    ratios = tuple(int(r) for r in args.ratios.split(",") if r.strip())

    # --run 支持「名称=目录」形式；只给目录时用目录名当标签
    runs: list[tuple[str, Path]] = []
    for item in args.run:
        name, _, path = item.partition("=")
        if not path:
            name, path = "", name
        runs.append((name.strip(), (ROOT / path.strip()).resolve()))

    if args.auto or not runs:
        runs = [
            ("", p.parent) for p in sorted((ROOT / "artifacts").rglob("predictions_oof.csv"))
        ]
    runs = [
        (name, path) for name, path in runs
        if (path / "predictions_oof.csv").exists() and (path / "predictions_test.csv").exists()
    ]
    if not runs:
        print("没有找到同时含 predictions_oof.csv 与 predictions_test.csv 的模型目录。")
        return 1

    blocks = []
    for name, run_dir in runs:
        table = scan(run_dir, ratios, args.cost_fp)
        # 图标题用可读名（artifacts/realdata_official/model -> realdata_official）
        display = name or (run_dir.parent.name if run_dir.name == "model" else run_dir.name)
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        label = run_dir.relative_to(ROOT).as_posix()
        blocks.append((f"{display}（{label}）" if name else label, table, meta, display))
        print(f"\n=== {label} ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lines = [
        "# 代价比敏感性扫描（真实数据）",
        "",
        "正类比例一旦超过 Elkan 阈值 p\\* = 1/(1+ρ)，「给所有人都报警」就同时是代价最优"
        "与不可执行的答案。本表固定 C_FP = ¥100、扫描 ρ，用 **OOF 预测重选阈值**、在测试集上"
        "评估，从而在不重新训练的前提下找出模型真正有决策价值的区间。",
        "",
    ]
    for label, table, meta, _display in blocks:
        metrics = meta.get("metrics_test", {})
        lines += [
            f"## {label}",
            "",
            f"- 测试集正类比例：{metrics.get('positive_rate', 'n/a')}",
            f"- 部署阈值下的结果：阈值 {metrics.get('threshold')}、"
            f"预测正类比例 {metrics.get('predicted_positive_rate')}、人均代价 ¥{metrics.get('cost_per_student')}",
            "",
            "| ρ | p\\* | 选用阈值 | 预测正类比例 | 召回 | 精确率 | 人均代价 | 平凡基线 | 相对下降 | 是否退化 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {int(row['cost_ratio'])} | {row['elkan_threshold']:.4f} | {row['selected_threshold']:.4f} | "
                f"{row['predicted_positive_rate']:.1%} | {row['recall']:.1%} | {row['precision']:.1%} | "
                f"¥{row['cost_per_student']:.2f} | ¥{row['baseline_cost_trivial']:.2f} | "
                f"{row['cost_reduction_vs_trivial']:+.1%} | {'是' if row['degenerate'] else '否'} |"
            )
        non_degenerate = table.loc[~table["degenerate"]]
        if not non_degenerate.empty:
            best = non_degenerate.loc[non_degenerate["cost_reduction_vs_trivial"].idxmax()]
            lines += [
                "",
                f"**结论**：ρ ≤ {int(non_degenerate['cost_ratio'].max())} 时退化解消失；"
                f"在 ρ = {int(best['cost_ratio'])} 处模型相对平凡基线下降最多（{best['cost_reduction_vs_trivial']:+.1%}）。",
            ]
        else:
            lines.append("")
            lines.append("**结论**：在本数据上，所有被扫描的 ρ 都退化为全报警，说明区分度不足以支撑阈值决策，应改用容量分析。")
        lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    combined = pd.concat(
        [t.assign(run=label) for label, t, _, _ in blocks], ignore_index=True
    )
    combined.to_csv(args.out_csv, index=False, float_format="%.4f")
    print(f"\n已写出 {args.out_md} 与 {args.out_csv}")

    if not args.no_fig:
        try:
            _label, table, _meta, display = blocks[0]
            plot(table, display, args.out_fig)
            print(f"已写出 {args.out_fig}")
        except Exception as exc:
            print(f"作图跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
