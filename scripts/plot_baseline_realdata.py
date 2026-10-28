"""真实数据上的五基线横向对比图（图 5-8）。

为什么单独作图：
    表 5-4 的合成数据对比图（图 5-1）只说明「合成数据上模型排名不可分辨」。
    真实数据上多了一层现象：两种标签口径下的代价列都贴着「全报警」基线，
    但排序指标（AUC/PR-AUC）明显更高。若只画代价柱状图，读者会误以为
    「真实数据上模型没学到东西」；若只画 AUC，又会忽略决策规则已失效。
    因此本图固定为「上排排序能力、下排期望代价」，并在代价子图上画出
    该口径的平凡基线，直接暴露「代价差异被基线吃掉」这一事实。

用法：
    python scripts/plot_baseline_realdata.py

输入：reports/realdata_official/compare/exp_comparison.csv、
      reports/realdata_early/compare/exp_comparison.csv
输出：docs/figs/fig5-8-baseline-realdata.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS = (
    ("官方标签（辍学率 79.29%）", ROOT / "reports/realdata_official/compare/exp_comparison.csv"),
    ("早期辍学口径（正类 67.63%）", ROOT / "reports/realdata_early/compare/exp_comparison.csv"),
)

SHORT = {
    "LR (baseline)": "LR",
    "RF (baseline)": "RF",
    "HGBDT (baseline)": "HGBDT",
    "CatBoost (no cost)": "CatBoost\n(无代价)",
    "CatBoost (cost-sensitive)": "CatBoost\n(代价敏感)",
}


def plot() -> Path:
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

    frames = [(title, pd.read_csv(path)) for title, path in RUNS]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    bar_color = "#2f5f8f"
    accent = "#c0392b"

    for col, (title, frame) in enumerate(frames):
        labels = [SHORT.get(name, name) for name in frame["experiment"]]
        x = list(range(len(frame)))

        # ---- 上排：排序能力 ----
        # y 轴从 0 起：本图要说明的正是「五个模型的排序能力挤在一起」，
        # 截断 y 轴会把 0.01 的差距画成天壤之别，恰好与结论相反。
        ax = axes[0][col]
        width = 0.38
        ax.bar([i - width / 2 for i in x], frame["roc_auc"], width, label="ROC-AUC", color=bar_color)
        ax.bar([i + width / 2 for i in x], frame["pr_auc"], width, label="PR-AUC", color="#f0a35e")
        for i, (auc, pr) in enumerate(zip(frame["roc_auc"], frame["pr_auc"])):
            ax.text(i - width / 2, auc + 0.012, f"{auc:.3f}", ha="center", fontsize=8)
            ax.text(i + width / 2, pr + 0.012, f"{pr:.3f}", ha="center", fontsize=8)
        auc_spread = frame["roc_auc"].max() - frame["roc_auc"].min()
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("指标值")
        ax.set_title(
            f"{title}\n排序能力｜蓝=ROC-AUC，橙=PR-AUC，AUC 全距 {auc_spread:.3f}",
            fontsize=10.5,
        )
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # ---- 下排：期望代价 vs 平凡基线 ----
        ax = axes[1][col]
        ax.bar(x, frame["cost_per_student"], 0.55, color=bar_color, label="人均期望代价")
        trivial = float(frame["baseline_cost_trivial"].iloc[0])
        ax.axhline(trivial, color=accent, linestyle="--", linewidth=1.5,
                   label=f"全报警基线 {trivial:.2f} 元")
        for i, (cost, rate) in enumerate(zip(frame["cost_per_student"], frame["predicted_positive_rate"])):
            # 标注放在柱内：柱顶紧贴虚线，放外面会与基线重叠
            ax.text(i, cost * 0.94, f"{cost:.2f}", ha="center", va="top", fontsize=8.4,
                    color="white", fontweight="bold")
            ax.text(i, cost * 0.82, f"预警 {rate:.1%}", ha="center", va="top", fontsize=7.4,
                    color="white")
        spread = frame["cost_per_student"].max() - frame["cost_per_student"].min()
        ax.set_ylim(0, frame["cost_per_student"].max() * 1.32)
        ax.set_ylabel("人均期望代价（元/人）")
        ax.set_title(
            f"期望代价｜虚线=全报警基线 {trivial:.2f} 元，五模型全距仅 {spread:.2f} 元",
            fontsize=10.5,
        )
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)

    fig.suptitle("图 5-8 真实数据上的五基线横向对比：排序能力有差异，代价差异被平凡基线吃掉", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = ROOT / "docs/figs/fig5-8-baseline-realdata.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = plot()
    print(f"图已保存 -> {path}")
