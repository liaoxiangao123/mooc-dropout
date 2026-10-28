"""补充实验配图：多种子稳定性（图 5-5）。

数据全部读自已落盘的结果 CSV，不重新训练——图与表必须来自同一份台账，
否则一旦重跑实验，图和表就会各自漂移。

用法：
    python scripts/make_extra_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "extra" / "multiseed_raw.csv"
FIG_DIR = ROOT / "docs" / "figs"

# 配色沿用正文其余插图：深蓝为主模型，灰阶为基线
COLORS = {
    "cb_cs": "#1F3864",
    "cb_none": "#8FAADC",
    "hgbdt": "#C55A11",
    "lr": "#7F7F7F",
}
MARKERS = {"cb_cs": "o", "cb_none": "s", "hgbdt": "^", "lr": "D"}

# 横轴放全称会互相挤压，改成两行短标签；全称仍可从正文表 5-9 查到
SHORT = {
    "cb_cs": "CatBoost\n代价敏感",
    "cb_none": "CatBoost\n无代价",
    "hgbdt": "HGBDT",
    "lr": "逻辑回归",
}


def setup_fonts() -> None:
    # 一次性给出候选列表；逐次赋值会把列表覆盖成单个元素，导致中文全部回退成方框
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#4A4A4A"
    plt.rcParams["axes.linewidth"] = 0.8


def panel_seeds(ax, df: pd.DataFrame, order: list[str]) -> None:
    """左图：每个种子一个人均代价点，横向轻微错开避免重叠。"""
    for i, key in enumerate(order):
        sub = df.loc[df["config"] == key]
        if sub.empty:
            continue
        x = i + np.linspace(-0.11, 0.11, len(sub))
        ax.scatter(
            x, sub["cost_per_student"], s=34, color=COLORS.get(key, "#333333"),
            marker=MARKERS.get(key, "o"), alpha=0.85, linewidths=0.6,
            edgecolors="white", zorder=3,
        )
        mean = sub["cost_per_student"].mean()
        sd = sub["cost_per_student"].std(ddof=1)
        ax.hlines(mean, i - 0.28, i + 0.32, color="#111111", lw=1.6, zorder=4)
        ax.hlines([mean - sd, mean + sd], i - 0.18, i + 0.18,
                  color="#111111", lw=1.0, zorder=4)
        ax.vlines(i, mean - sd, mean + sd, color="#111111", lw=1.0, zorder=4)
        # 均值标在线尾右侧，避免与折线上的散点、误差棒重叠
        ax.annotate(f"{mean:.2f}", (i + 0.32, mean), textcoords="offset points",
                    xytext=(4, 0), va="center", ha="left", fontsize=8.2, color="#111111")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT.get(k, k) for k in order], fontsize=8.4)
    ax.set_xlim(-0.55, len(order) - 0.35)
    ax.set_ylabel("测试集人均期望代价（元/人）", fontsize=10)
    ax.set_title("(a) 五个随机种子下的人均代价", fontsize=10.5, pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def panel_paired(ax, df: pd.DataFrame, base: str, order: list[str]) -> None:
    """右图：逐种子配对差。所有差值都落在零线附近的同一窄带内，
    说明「谁更便宜」主要由这一次的切分决定，而非模型本身。"""
    others = [k for k in order if k != base and not df.loc[df["config"] == k].empty]
    labels, means, los, his = [], [], [], []
    for key in others:
        a = df.loc[df["config"] == key].set_index("seed")["cost_per_student"]
        b = df.loc[df["config"] == base].set_index("seed")["cost_per_student"]
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        d = (a.loc[common] - b.loc[common]).to_numpy(dtype="float64")
        # 用 t 分位数而非 1.96：n=5 时 t=2.776，正态近似会把区间画窄，
        # 与正文表 5-10 的配对 t 检验结论对不上
        t_crit = float(stats.t.ppf(0.975, df=len(d) - 1))
        half = t_crit * d.std(ddof=1) / np.sqrt(len(d))
        labels.append(SHORT.get(key, key).replace("\n", " "))
        means.append(d.mean())
        los.append(d.mean() - half)
        his.append(d.mean() + half)

    y = np.arange(len(labels))
    ax.axvline(0, color="#111111", lw=1.0, ls="--", zorder=1)
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los),
                                np.array(his) - np.array(means)],
                fmt="o", ms=6, color="#1F3864", ecolor="#1F3864",
                elinewidth=1.4, capsize=4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlabel("与 CatBoost（代价敏感）的人均代价差（元/人）", fontsize=10)
    ax.set_title("(b) 逐种子配对差与 95% 区间（t 分布）", fontsize=10.5, pad=8)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="cb_cs")
    args = parser.parse_args()

    if not RAW.exists():
        print(f"缺少 {RAW}，请先运行 scripts/run_extra_experiments.py")
        return 2

    setup_fonts()
    df = pd.read_csv(RAW)
    order = [k for k in ("cb_cs", "cb_none", "hgbdt", "lr") if (df["config"] == k).any()]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=300,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    panel_seeds(axes[0], df, order)
    panel_paired(axes[1], df, args.base, order)

    fig.tight_layout()
    out = FIG_DIR / "fig5-5-multiseed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"已生成 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
