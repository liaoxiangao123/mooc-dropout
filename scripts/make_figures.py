"""生成论文用图（图 5-1 ~ 5-4），统一风格：白底、无衬线中文字体、浅灰网格。

用法：
    python scripts/make_figures.py --data-dir data/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import get_settings  # noqa: E402
from mooc_dropout.data.loader import load_kdd_cup  # noqa: E402
from mooc_dropout.evaluation.cost import CostMatrix, cost_curve  # noqa: E402
from mooc_dropout.repository.artifact_repository import ArtifactRepository  # noqa: E402
from mooc_dropout.training.trainer import TrainConfig, Trainer  # noqa: E402

PRIMARY = "#2f6fed"
ACCENT = "#d93025"
MUTED = "#6b7280"
GRID = "#e3e6eb"

FIG_DIR = ROOT / "docs" / "figs"


def setup_font() -> None:
    """优先使用系统中文字体，避免图中出现方块。"""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d0d4da",
            "axes.labelcolor": "#1f2329",
            "text.color": "#1f2329",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "savefig.dpi": 200,
            "figure.dpi": 120,
        }
    )


def fig_model_comparison(reports: Path) -> Path:
    frame = pd.read_csv(reports / "exp_comparison.csv")
    labels = ["LR", "RF", "HGBDT", "CatBoost\n(无代价)", "CatBoost\n(代价敏感)"]
    frame = frame.head(5)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(frame))
    axes[0].bar(x, frame["roc_auc"], width=0.55, color=["#9fb8e8"] * 4 + [PRIMARY])
    axes[0].set_ylim(0.6, 0.8)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("(a) 排序能力", fontsize=10)
    for xi, value in zip(x, frame["roc_auc"]):
        axes[0].text(xi, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)

    axes[1].bar(x, frame["cost_per_student"], width=0.55, color=["#f2b8b5"] * 4 + [ACCENT])
    axes[1].set_ylabel("人均期望代价（元）")
    axes[1].set_title("(b) 代价最优阈值下的代价", fontsize=10)
    for xi, value in zip(x, frame["cost_per_student"]):
        axes[1].text(xi, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "fig5-1-model-comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cost_curve(data_dir: Path, settings) -> tuple[Path, float]:
    """重新计算测试集代价曲线（与训练脚本同一特征路径）。"""
    enrollments, logs = load_kdd_cup(data_dir)
    config = TrainConfig.from_settings(settings, model_name="catboost")
    data = Trainer(config, settings).prepare(logs, enrollments)

    artifacts = ArtifactRepository(settings.model_dir).load()
    prob = artifacts.model.predict_proba(data.X_test)
    matrix = CostMatrix(settings.cost_fn, settings.cost_fp)
    curve = cost_curve(data.y_test, prob, matrix, n_points=99)
    threshold = artifacts.threshold

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(curve["threshold"], curve["per_student_cost"], color=PRIMARY, linewidth=2, label="期望代价")

    best = curve.loc[curve["per_student_cost"].idxmin()]
    ax.scatter([threshold], [best["per_student_cost"]], color=ACCENT, zorder=5, s=36)
    ax.annotate(
        f"t* = {threshold:.2f}\n¥{best['per_student_cost']:.1f}/人",
        xy=(threshold, best["per_student_cost"]),
        xytext=(threshold + 0.10, best["per_student_cost"] + 8),
        fontsize=8,
        color=ACCENT,
        arrowprops={"arrowstyle": "-", "color": ACCENT, "lw": 0.8},
    )

    at_half = curve.iloc[(curve["threshold"] - 0.5).abs().argsort()[:1]]["per_student_cost"].iloc[0]
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.92, "默认阈值 0.5", fontsize=8, color=MUTED)

    ax.set_xlabel("决策阈值 t")
    ax.set_ylabel("人均期望代价（元）")
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    path = FIG_DIR / "fig5-2-cost-curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path, float(at_half)


def fig_sensitivity(reports: Path) -> Path:
    frame = pd.read_csv(reports / "exp_cost_ratio.csv")
    ratios = frame["train_cost_ratio"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
    ax1.plot(ratios, frame["recall"], marker="o", color=PRIMARY, linewidth=2, label="Recall")
    ax1.plot(ratios, frame["precision"], marker="s", color=MUTED, linewidth=1.6, linestyle="--", label="Precision")
    ax1.set_xlabel("代价比 ρ = C_FN / C_FP")
    ax1.set_ylabel("指标值")
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale("log")
    ax1.set_xticks(ratios)
    ax1.set_xticklabels([f"{int(r)}" for r in ratios])

    ax2 = ax1.twinx()
    ax2.bar(ratios, frame["n_high_risk"], width=0.35, alpha=0.25, color="#f2b8b5", label="预警人数")
    ax2.set_ylabel("预警人数", color=MUTED)
    ax2.grid(False)

    lines = ax1.get_legend_handles_labels()
    bars = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + bars[0], lines[1] + bars[1], frameon=False, fontsize=8, loc="center right")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "fig5-3-sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_importance(settings) -> Path:
    frame = pd.read_csv(settings.model_dir / "feature_importance.csv").head(15).iloc[::-1]
    colors = [PRIMARY if value >= frame["importance"].median() else "#9fb8e8" for value in frame["importance"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(frame.iloc[:, 0], frame["importance"], color=colors, height=0.65)
    ax.set_xlabel("CatBoost 特征重要性")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    path = FIG_DIR / "fig3-2-feature-importance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成论文图 5-1 ~ 5-4")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    args = parser.parse_args()

    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    reports = settings.resolved_reports_dir

    print(fig_model_comparison(reports))
    curve_path, at_half = fig_cost_curve(args.data_dir, settings)
    print(curve_path)
    print(f"0.5 阈值下人均代价: {at_half:.2f}")
    print(fig_sensitivity(reports))
    print(fig_importance(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
