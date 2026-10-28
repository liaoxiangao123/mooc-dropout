"""容量分析：固定干预人力下的 top-K 决策曲线。

为什么需要它：
    代价最优阈值只有在「预警多少人就干预多少人」成立时才有意义。真实场景里教师的
    干预人力是有限的（一学期能跟进的名单就那么多）。当正类比例高于 Elkan 阈值时，
    「全报警」会同时成为代价最优与不可执行的答案 —— 两者矛盾，说明阈值视角在这类
    基率下已经失效。容量分析把问题换成一个更贴近现实的形式：

        给定只能干预 K 个人，按风险从高到低取前 K 个，能得到多少召回、省多少钱？

它同时给出三条曲线：
    1. 召回率随 K 的变化（要抓住 80% 的辍学生，名单要多大）；
    2. 人均期望代价随 K 的变化（与「全报警」「全放行」两条水平线对比）；
    3. 提升曲线（lift）—— 前 K% 里辍学生的浓度是整体辍学率的多少倍。

用法：
    python scripts/capacity_analysis.py \
        --run 官方标签=artifacts/model_realdata_weighted \
        --run 早期辍学=artifacts/model_realdata_early

    # 自动扫描 artifacts/ 下所有含 predictions_test.csv 的模型目录
    python scripts/capacity_analysis.py --auto

输入：模型目录下的 predictions_test.csv 与 metadata.json（由 Trainer 落盘）。
输出：reports/容量分析-K曲线.md、reports/capacity_curve.csv、docs/figs/fig5-7-capacity.png
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

from mooc_dropout.evaluation.capacity import (  # noqa: E402
    DEFAULT_CAPACITIES,
    best_capacity,
    capacity_curve,
    capacity_for_recall,
)
from mooc_dropout.evaluation.cost import CostMatrix  # noqa: E402

DEFAULT_KS = DEFAULT_CAPACITIES


def load_run(artifact_dir: Path) -> dict:
    pred_path = artifact_dir / "predictions_test.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"{artifact_dir} 下没有 predictions_test.csv，请先重新训练")
    frame = pd.read_csv(pred_path)
    meta_path = artifact_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    # metadata 里的 cost_matrix 含派生字段 ratio / elkan_threshold，不能直接展开成构造参数
    raw_matrix = meta.get("cost_matrix") or {}
    fields = {k: raw_matrix[k] for k in ("c_fn", "c_fp", "c_tp", "c_tn") if k in raw_matrix}
    matrix = CostMatrix(**fields) if fields else CostMatrix()
    return {"dir": artifact_dir, "frame": frame, "meta": meta, "matrix": matrix}


def curve_for(item: dict, ks: tuple[float, ...]) -> pd.DataFrame:
    return capacity_curve(
        item["frame"]["y_true"].to_numpy(),
        item["frame"]["probability"].to_numpy(),
        item["matrix"],
        capacities=ks,
    )


def plot(curves: dict[str, pd.DataFrame], out_path: Path) -> None:
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#2f5f8f", "#c0392b", "#27ae60", "#8e44ad"]

    for i, (name, curve) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        x = curve["capacity"] * 100
        axes[0].plot(x, curve["recall"], marker="o", ms=3, color=color, label=name)
        axes[1].plot(x, curve["cost_per_student"], marker="o", ms=3, color=color, label=name)
        axes[2].plot(x, curve["lift"], marker="o", ms=3, color=color, label=name)

        first = next(iter(curves.values()))
        for ax, key, style in (
            (axes[1], "baseline_all_positive", "--"),
            (axes[1], "baseline_all_negative", ":"),
        ):
            ax.axhline(first.attrs[key], color="#888888", linestyle=style, linewidth=1)

    axes[0].set_xlabel("干预容量 K（风险最高的前 K%）")
    axes[0].set_ylabel("召回率")
    axes[0].set_title("要抓住多少辍学生，名单需要多大")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("干预容量 K（%）")
    axes[1].set_ylabel("人均期望代价（元）")
    axes[1].set_title("代价随容量的变化（虚线=全报警，点线=全放行）")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("干预容量 K（%）")
    axes[2].set_ylabel("提升倍数 lift")
    axes[2].set_title("前 K% 名单中辍学生的富集倍数")
    axes[2].axhline(1.0, color="#888888", linestyle=":", linewidth=1)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="容量分析（top-K 决策曲线）")
    parser.add_argument("--run", action="append", default=[], metavar="名称=模型目录")
    parser.add_argument("--auto", action="store_true", help="自动扫描 artifacts/ 下的模型目录")
    parser.add_argument("--out-md", type=Path, default=ROOT / "reports" / "容量分析-K曲线.md")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "reports" / "capacity_curve.csv")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "docs" / "figs" / "fig5-7-capacity.png")
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    runs: dict[str, Path] = {}
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run 需要写成 名称=目录 的形式，收到: {item}")
        name, path = item.split("=", 1)
        runs[name] = (ROOT / path).resolve()

    if args.auto or not runs:
        for meta_path in sorted((ROOT / "artifacts").rglob("metadata.json")):
            directory = meta_path.parent
            if (directory / "predictions_test.csv").exists():
                runs[directory.relative_to(ROOT).as_posix()] = directory
    if not runs:
        print("没有找到含 predictions_test.csv 的模型目录，先跑一次训练。")
        return 1

    loaded = {name: load_run(path) for name, path in runs.items()}
    curves = {name: curve_for(item, DEFAULT_KS) for name, item in loaded.items()}

    all_rows = []
    for name, curve in curves.items():
        part = curve.copy()
        part.insert(0, "run", name)
        all_rows.append(part)
    combined = pd.concat(all_rows, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False, float_format="%.4f")

    lines = [
        "# 容量分析：固定干预人力下的 top-K 决策曲线",
        "",
        "> 当正类比例高于 Elkan 阈值（本文 0.0909）时，「全报警」既是代价最优又不可执行，"
        "阈值视角失去意义。容量分析把决策规则换成「只能干预 K 个人，按风险排序取前 K」，"
        "回答的是教务处真正会问的问题：名单要开多大，才抓得住多少辍学生。",
        "",
    ]

    for name, curve in curves.items():
        attrs = curve.attrs
        meta = loaded[name]["meta"]
        lines += [
            f"## {name}",
            "",
            f"- 模型目录：`{loaded[name]['dir'].relative_to(ROOT).as_posix()}`",
            f"- 测试集样本：{attrs['n']:,}；正类：{attrs['n_pos']:,}（{attrs['base_rate']:.2%}）",
            f"- 代价矩阵：C_FN = ¥{loaded[name]['matrix'].c_fn:.0f}，C_FP = ¥{loaded[name]['matrix'].c_fp:.0f}",
            f"- 平凡基线：全报警 ¥{attrs['baseline_all_positive']:.2f}/人，"
            f"全放行 ¥{attrs['baseline_all_negative']:.2f}/人",
        ]
        metrics = meta.get("metrics_test", {})
        if metrics:
            lines.append(
                f"- 阈值策略结果：阈值 {metrics.get('threshold')}，"
                f"召回 {metrics.get('recall')}，精确率 {metrics.get('precision')}，"
                f"人均代价 ¥{metrics.get('cost_per_student')}"
            )
        lines += [
            "",
            "| 干预容量 K | 名单人数 | 等价阈值 | 命中(TP) | 召回率 | 精确率 | 提升 lift | 人均代价(元) | 随机基线(元) | 相对随机 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, row in curve.iterrows():
            lines.append(
                f"| {row['capacity']:.1%} | {int(row['n_alerted']):,} | {row['threshold']:.3f} | "
                f"{int(row['tp']):,} | {row['recall']:.1%} | {row['precision']:.1%} | "
                f"{row['lift']:.2f}× | {row['cost_per_student']:.2f} | "
                f"{row['baseline_cost_random']:.2f} | {row['cost_reduction_vs_random']:+.1%} |"
            )
        lines.append("")

    lines += [
        "## 怎么读这张表",
        "",
        "- **lift > 1** 表示名单确实富集了高危学生，模型有排序价值；lift 随 K 增大趋近 1，"
        "因为名单最终覆盖全体。",
        "- **「相对随机」是容量受限时唯一合法的增益口径**：同样开 K 个名额，随机挑人得到的"
        "期望代价作为基线，两者的比值约掉了容量因素，只剩排序质量。若拿「全报警」当基线，"
        "K < 100% 时该列必然是大负数——那不是模型变差，而是容量约束变紧、漏报必然增加。",
        "- **人均代价的谷底**就是该代价比下该用的容量。若谷底出现在 K=1.0（全报警），说明在"
        "当前代价比下，模型给不出比「人人预警」更省钱的方案，此时应调低 C_FN/C_FP，"
        "或直接用「相对随机」列评估排序价值。",
        "- 与 `reports/实验记录-合成数据.md` 对照：合成数据正类 20%，阈值方案能正常工作；"
        "真实数据正类 68%~79%，阈值方案几乎总落在「全报警」一端，只有容量视角还能给出"
        "可执行结论。",
        "",
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {args.out_md} 与 {args.out_csv}")

    if not args.no_fig:
        try:
            plot(curves, args.out_fig)
            print(f"已写出 {args.out_fig}")
        except Exception as exc:  # 画图失败不影响表
            print(f"作图跳过：{exc}")

    for name, curve in curves.items():
        best = best_capacity(curve)
        hit_80 = capacity_for_recall(curve, 0.8) if not curve["recall"].isna().all() else None
        print(
            f"{name}: 代价最优容量 K={best['capacity']:.1%}，召回 {best['recall']:.1%}，"
            f"人均代价 ¥{best['cost_per_student']:.2f}"
        )
        if hit_80:
            print(
                f"    想抓住 80% 辍学生：名单需覆盖 {hit_80['capacity']:.1%}"
                f"（{hit_80['n_alerted']:,} 人），精确率 {hit_80['precision']:.1%}，"
                f"lift {hit_80['lift']:.2f}×"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
