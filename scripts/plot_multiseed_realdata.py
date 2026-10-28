"""多种子重复实验作图（图 5-9）。

要传达的一件事：
    「模型之间的差」和「换种子带来的波动」哪个大？

为什么分三栏而不是两栏：
    两种标签口径的人均代价分别落在 20 元与 29 元附近，画在同一个纵轴上时，
    官方口径那 0.2 元的种子离散会被压成一条平线——而 0.2 元恰好就是要看的量。
    因此 (a)(b) 两栏各自独立定纵轴，只比较「同一栏内部」的散点铺开与均值线间距；
    (c) 栏画逐种子配对差（同一 seed 内同一批测试学生，只差模型），零线是「无差异」。

用法：
    python scripts/plot_multiseed_realdata.py
输入：reports/multiseed/{official,early}/{raw.csv,paired.csv}
输出：docs/figs/fig5-9-multiseed-realdata.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS = (
    ("official", "（a）官方标签", "#2f5f8f"),
    ("early", "（b）早期辍学口径", "#c0392b"),
)
LABEL_MAP = {
    "cb_cs": "CatBoost\n(代价敏感)",
    "cb_none": "CatBoost\n(无代价)",
    "hgbdt": "HGBDT",
    "lr": "LR",
}
ORDER = ("cb_cs", "cb_none", "hgbdt", "lr")
PAIR_MAP = {
    "cb_cs - cb_none": "代价敏感\nvs 无代价",
    "cb_cs − cb_none": "代价敏感\nvs 无代价",
    "cb_cs - hgbdt": "代价敏感\nvs HGBDT",
    "cb_cs − hgbdt": "代价敏感\nvs HGBDT",
    "cb_cs - lr": "代价敏感\nvs LR",
    "cb_cs − lr": "代价敏感\nvs LR",
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

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    # ---- (a)(b)：逐种子散点 + 均值 ----
    for ax, (tag, title, color) in zip(axes[:2], RUNS):
        raw_path = ROOT / "reports/multiseed" / tag / "raw.csv"
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        for i, key in enumerate(ORDER):
            sub = raw[raw["config"] == key]
            if sub.empty:
                continue
            jitter = (pd.Series(range(len(sub))) - (len(sub) - 1) / 2) * 0.07
            ax.scatter(i + jitter.to_numpy(), sub["cost_per_student"], s=42,
                       color=color, alpha=0.62, edgecolor="white", linewidth=0.7, zorder=3)
            mean = sub["cost_per_student"].mean()
            ax.hlines(mean, i - 0.30, i + 0.30, color="#222222", linewidth=2.0, zorder=4)
            ax.annotate(f"{mean:.2f}", (i, mean), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color="#222222", zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                              "edgecolor": "none", "alpha": 0.85})

        mean_range = raw.groupby("config")["cost_per_student"].mean().max() - \
            raw.groupby("config")["cost_per_student"].mean().min()
        seed_std = raw.groupby("config")["cost_per_student"].std(ddof=1).mean()
        ax.set_xticks(range(len(ORDER)), [LABEL_MAP[k] for k in ORDER], fontsize=8.5)
        ax.set_xlim(-0.62, len(ORDER) - 1 + 0.58)
        ax.set_ylabel("人均期望代价（元/人）")
        ax.set_title(f"{title}｜{raw['seed'].nunique()} 个种子", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.18)
        # 一句话把「波动 vs 差异」写在图里，避免读者自己算
        ax.text(0.98, 0.03,
                f"配置间均值极差 {mean_range:.2f} 元\n种子间标准差（均值） {seed_std:.2f} 元",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "#bbbbbb", "alpha": 0.9})

    # ---- (c)：配对差 ----
    ax = axes[2]
    rows = []
    for tag, _, color in RUNS:
        pair_path = ROOT / "reports/multiseed" / tag / "paired.csv"
        if not pair_path.exists():
            continue
        pairs = pd.read_csv(pair_path)
        for _, r in pairs.iterrows():
            rows.append((tag, color, r["pair"], r["diff_mean"], r["ci_low"], r["ci_high"],
                         int(r["n_negative"]), int(r["n_positive"])))
    for i, (tag, color, pair, mean, lo, hi, nneg, npos) in enumerate(rows):
        offset = -0.16 if tag == "official" else 0.16
        ax.errorbar(i + offset, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, capsize=4, markersize=6, linewidth=1.5, zorder=3)
        ax.annotate(f"{mean:+.2f}", (i + offset, min(mean, lo)),
                    textcoords="offset points", xytext=(0, -12), ha="center",
                    fontsize=7.5, color=color)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1, zorder=2)
    ax.set_xticks(range(len(rows)),
                  [PAIR_MAP.get(r[2], r[2]) for r in rows], fontsize=8)
    ax.set_ylabel("配对差（元/人，负 = 代价敏感更便宜）")
    ax.set_title("（c）逐种子配对差与 Bootstrap 95% 区间", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.16)
    for tag, _, color in RUNS:
        name = "官方标签" if tag == "official" else "早期辍学口径"
        ax.scatter([], [], color=color, label=name, s=30)
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("图 5-9 真实数据上的种子间波动与模型间差异（5 种子 × 4 配置 × 2 口径）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "docs/figs/fig5-9-multiseed-realdata.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("图已保存 ->", plot())
