"""为阶段展示幻灯片重画图 5-2（代价-阈值曲线）。

与 `scripts/make_figures.py::fig_cost_curve` 同源同数据，只改呈现参数：
字号放大、画布收窄，以便在 512px 宽的幻灯片区域内仍然可读。

数据源与论文完全一致（`artifacts/extra/pred_cb_cs_seed42.npz` 的冻结预测 +
OOF 选阈），因此图上任一数字都与表 5-6 对得上。

用法：
    python slides/阶段展示汇报/make_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import get_settings  # noqa: E402
from mooc_dropout.evaluation.cost import (  # noqa: E402
    CostMatrix,
    cost_curve,
    cost_optimal_threshold,
)

PRIMARY = "#1E4FA8"
ACCENT = "#D9534F"
MUTED = "#8B97A8"
GRID = "#E5E7EB"

NPZ = ROOT / "artifacts" / "extra" / "pred_cb_cs_seed42.npz"
OUT = Path(__file__).resolve().parent / "assets" / "cost_curve.png"


def setup_font() -> None:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D6DCE5",
            "axes.labelcolor": "#1A2230",
            "text.color": "#1A2230",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "savefig.dpi": 220,
        }
    )


def main() -> None:
    setup_font()
    settings = get_settings()

    frozen = np.load(NPZ)
    y_oof, prob_oof = frozen["y_train"], frozen["oof_train"]
    y_test, prob_test = frozen["y_test"], frozen["prob_test"]

    matrix = CostMatrix(settings.cost_fn, settings.cost_fp)
    threshold = cost_optimal_threshold(y_oof, prob_oof, matrix)["threshold"]
    # n_points 必须与 scripts/make_figures.py 一致（99），否则曲线最低点会因采样密度
    # 变化而落在别的阈值上，与论文正文的 75.91 元 / t=0.39 对不上
    curve = cost_curve(y_test, prob_test, matrix, n_points=99)

    chosen = matrix.per_student_cost(y_test, (prob_test >= threshold).astype(int))
    best = curve.loc[curve["per_student_cost"].idxmin()]
    all_alarm = matrix.per_student_cost(y_test, np.ones_like(y_test))

    fig, ax = plt.subplots(figsize=(6.5, 4.05))
    ax.plot(curve["threshold"], curve["per_student_cost"], color=PRIMARY, linewidth=2.4)

    # 左端点即「全部预警」：阈值为 0 时所有人都被判为正类
    ax.scatter([0.0], [all_alarm], color=MUTED, s=46, zorder=5)
    ax.annotate(
        f"全部预警  ¥{all_alarm:.2f}",
        xy=(0.0, all_alarm),
        xytext=(0.035, 104),
        fontsize=15,
        color="#4A5568",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1.0},
    )

    ax.scatter([threshold], [chosen], color=ACCENT, s=64, zorder=6)
    ax.annotate(
        f"OOF 选阈 {threshold:.2f}\n¥{chosen:.2f}",
        xy=(threshold, chosen),
        xytext=(0.16, 136),
        fontsize=15,
        color=ACCENT,
        arrowprops={"arrowstyle": "-", "color": ACCENT, "lw": 1.0},
    )

    ax.scatter(
        [best["threshold"]],
        [best["per_student_cost"]],
        facecolors="white",
        edgecolors="#4A5568",
        linewidths=1.4,
        zorder=5,
        s=46,
    )
    ax.annotate(
        f"曲线最低 {best['threshold']:.2f}  ¥{best['per_student_cost']:.2f}",
        xy=(best["threshold"], best["per_student_cost"]),
        xytext=(0.52, 116),
        fontsize=14,
        color="#4A5568",
        arrowprops={"arrowstyle": "-", "color": "#4A5568", "lw": 1.0},
    )

    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.515, 186, "默认阈值 0.5", fontsize=14, color=MUTED)

    ax.set_xlabel("决策阈值 t", fontsize=18)
    ax.set_ylabel("人均期望代价（元）", fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(68, 208)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT)
    plt.close(fig)

    print(f"已写出 {OUT}　（t*={threshold:.4f}　本文 ¥{chosen:.2f}　全报警 ¥{all_alarm:.2f}　最低 ¥{best['per_student_cost']:.2f}）")


if __name__ == "__main__":
    main()
