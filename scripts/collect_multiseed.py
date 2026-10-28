"""从已落盘的预测文件重建多种子汇总（不重训）。

为什么单独有这个脚本：
    ``multiseed_realdata.py`` 每跑完一个 tag 就把 raw/summary/paired 三个 CSV 整体重写。
    如果中途因内存不足崩在最后一个种子上、事后用 ``--reuse --seeds 3407`` 补跑，
    那份 CSV 就只剩一个种子。本脚本直接读 ``artifacts/multiseed/{tag}/pred_*.npz``，
    按同一套指标定义重建三个 CSV，既省掉重新 prepare 的 10 分钟与 2GB 内存，
    也保证补跑与原有种子落在同一张表里。

    阈值不做第二次优化：直接取 npz 里存的值（该值本来就是在训练集 OOF 上选的，
    测试集没有参与），因此重建结果与训练时的打印逐位一致。

用法：
    python scripts/collect_multiseed.py
    python scripts/collect_multiseed.py --tags official early --seeds 42 7 2024 1234 3407
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mooc_dropout.config import get_settings  # noqa: E402
from mooc_dropout.evaluation.cost import CostMatrix  # noqa: E402
from mooc_dropout.evaluation.metrics import classification_metrics  # noqa: E402
from multiseed_realdata import (  # noqa: E402
    DEFAULT_ORDER,
    DEFAULT_SEEDS,
    KEY_METRICS,
    SPECS,
    paired,
    summarize,
)


def collect(tag: str, seeds: list[int], configs: list[str], cost_matrix: CostMatrix) -> pd.DataFrame:
    out_dir = ROOT / "artifacts/multiseed" / tag
    rows: list[dict] = []
    missing: list[str] = []
    for seed in seeds:
        for key in configs:
            npz = out_dir / f"pred_{key}_seed{seed}.npz"
            if not npz.exists():
                missing.append(f"{key}/seed{seed}")
                continue
            blob = np.load(npz)
            threshold = float(blob["threshold"][0])
            metrics = classification_metrics(
                blob["y_test"], blob["prob_test"], threshold=threshold, cost_matrix=cost_matrix
            )
            metrics["threshold"] = round(threshold, 6)
            row = {
                "seed": seed,
                "config": key,
                "label": SPECS[key].label,
                "n_test": int(metrics["n_samples"]),
            }
            row.update({k: round(float(metrics[k]), 4) for k in KEY_METRICS})
            row.update(fn=int(metrics["fn"]), fp=int(metrics["fp"]), elapsed_s=0.0)
            rows.append(row)
    if missing:
        print(f"  [{tag}] 缺 {len(missing)} 个预测：{', '.join(missing)}")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 npz 重建多种子汇总")
    parser.add_argument("--tags", nargs="+", default=["official", "early"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_ORDER), choices=list(SPECS))
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    settings = get_settings()
    cost_matrix = CostMatrix(c_fn=settings.cost_fn, c_fp=settings.cost_fp)

    for tag in args.tags:
        raw = collect(tag, args.seeds, args.configs, cost_matrix)
        if raw.empty:
            print(f"[{tag}] 无可用预测，跳过")
            continue
        raw["config"] = pd.Categorical(raw["config"], categories=args.configs, ordered=True)
        raw = raw.sort_values(["seed", "config"]).reset_index(drop=True)
        raw["config"] = raw["config"].astype(str)

        report_dir = ROOT / "reports/multiseed" / tag
        report_dir.mkdir(parents=True, exist_ok=True)
        raw.to_csv(report_dir / "raw.csv", index=False, encoding="utf-8-sig")
        order = tuple(c for c in DEFAULT_ORDER if c in set(raw["config"]))
        summary = summarize(raw, order)
        summary.to_csv(report_dir / "summary.csv", index=False, encoding="utf-8-sig")
        pairs = paired(raw, n_boot=args.n_boot)
        pairs.to_csv(report_dir / "paired.csv", index=False, encoding="utf-8-sig")

        print(f"\n===== [{tag}] 汇总（{raw['seed'].nunique()} 个种子）=====")
        print(summary[["label", "n_seeds", "roc_auc_mean", "roc_auc_std",
                       "cost_per_student_mean", "cost_per_student_std",
                       "cost_min", "cost_max", "cost_reduction_vs_trivial_mean"]].to_string(index=False))
        print("\n----- 逐种子配对差（负=左项更便宜）-----")
        print(pairs.to_string(index=False))
        print(f"产物: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
