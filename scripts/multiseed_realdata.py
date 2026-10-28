"""真实数据多种子重复实验（E14）。

为什么做：
    5.9.6 节的五基线对比是**单次切分**的结果。真实数据上 CatBoost（代价敏感）与
    CatBoost（无代价）的人均代价只差 0.02 元（官方口径）/ 0.05 元（早期口径），
    这个差到底是真是假，单次切分回答不了——换一个随机种子，切分变了，测试集换了，
    数字就会动。本脚本把 5.8 节合成数据上的做法（多种子重复 + 逐种子配对差 + Bootstrap）
    照搬到真实数据上，用来回答一个问题：

        「模型之间的差异」与「种子之间的波动」哪个更大？

    如果前者远小于后者，那么任何「A 模型比 B 模型便宜 0.05 元」的说法都不成立，
    论文只能主张决策链路（阈值/容量）的价值，而不能主张模型排名。

与 5.9.6 的差异（有意为之）：
    折数 3（与 5.9.6 一致，真实数据 12 万条不宜用 5 折 × 5 种子 × 4 配置）、
    迭代 300、配置取 4 个（去掉 CatBoost 无代价以外的重复项）。

用法：
    # 官方标签，5 个种子
    python scripts/multiseed_realdata.py --data-dir data/raw/kdd2015 \
        --tag official --seeds 42 7 2024 1234 3407

    # 早期辍学口径
    python scripts/multiseed_realdata.py --data-dir data/raw/kdd2015 \
        --label-file data/processed/enrollment_early.csv --tag early --seeds 42 7 2024 1234 3407

    # 复用已有预测，只重算汇总
    python scripts/multiseed_realdata.py --tag official --reuse

输出：
    artifacts/multiseed/{tag}/pred_{config}_seed{seed}.npz
    reports/multiseed/{tag}/raw.csv       逐种子逐配置明细
    reports/multiseed/{tag}/summary.csv   均值 ± 标准差
    reports/multiseed/{tag}/paired.csv    逐种子配对差 + Bootstrap 区间
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import get_settings  # noqa: E402
from mooc_dropout.data.loader import load_kdd_cup, read_enrollments  # noqa: E402
from mooc_dropout.evaluation.cost import cost_optimal_threshold  # noqa: E402
from mooc_dropout.evaluation.metrics import classification_metrics  # noqa: E402
from mooc_dropout.training.trainer import PreparedData, TrainConfig, Trainer  # noqa: E402

DEFAULT_SEEDS = (42, 7, 2024, 1234, 3407)
KEY_METRICS = (
    "threshold", "roc_auc", "pr_auc", "recall", "precision", "f1",
    "brier", "cost_per_student", "predicted_positive_rate",
    "cost_reduction_vs_trivial",
)


@dataclass(frozen=True)
class Spec:
    """一个待评配置。cost_ratio 只作用于训练权重，评估一律用业务代价矩阵。"""

    key: str
    label: str
    model_name: str
    weighting: str
    cost_ratio: float


SPECS: dict[str, Spec] = {
    "cb_cs": Spec("cb_cs", "CatBoost（代价敏感）", "catboost", "class_weight", 10.0),
    "cb_none": Spec("cb_none", "CatBoost（无代价）", "catboost", "none", 1.0),
    "hgbdt": Spec("hgbdt", "HGBDT", "hist_gbdt", "none", 1.0),
    "lr": Spec("lr", "LR", "logreg", "none", 1.0),
}
DEFAULT_ORDER = ("cb_cs", "cb_none", "hgbdt", "lr")


def _params_for(spec: Spec, iterations: int) -> dict[str, Any]:
    """按模型族翻译超参：CatBoost 用 iterations，sklearn 的树用 max_iter，LR 不吃这两个。"""
    params: dict[str, Any] = {"cost_ratio": spec.cost_ratio}
    if spec.model_name == "catboost":
        params["iterations"] = iterations
    elif spec.model_name == "hist_gbdt":
        params["max_iter"] = iterations
    return params


def make_trainer(seed: int, spec: Spec | None, iterations: int, n_splits: int) -> Trainer:
    settings = get_settings()
    overrides: dict[str, Any] = {"seed": seed, "n_splits": n_splits}
    if spec is not None:
        overrides.update(
            model_name=spec.model_name,
            weighting=spec.weighting,
            model_params=_params_for(spec, iterations),
        )
    return Trainer(TrainConfig.from_settings(settings, **overrides), settings)


def fit_evaluate(
    data: PreparedData, seed: int, spec: Spec, iterations: int, n_splits: int
) -> dict[str, Any]:
    """一次完整的「训练 → OOF 选阈值 → 测试集评估」。"""
    trainer = make_trainer(seed, spec, iterations, n_splits)
    model = trainer.create_model()
    model.fit(data.X_train, data.y_train)

    # 阈值只在训练集 OOF 上选，测试集不参与
    oof = trainer._oof_probabilities(data)
    threshold = float(
        cost_optimal_threshold(data.y_train, oof, trainer.cost_matrix)["threshold"]
    )
    prob_test = model.predict_proba(data.X_test)
    metrics = classification_metrics(
        data.y_test, prob_test, threshold=threshold, cost_matrix=trainer.cost_matrix
    )
    metrics["threshold"] = round(threshold, 6)
    return {"metrics": metrics, "prob_test": np.asarray(prob_test, dtype="float64")}


def run(args: argparse.Namespace) -> pd.DataFrame:
    settings = get_settings()
    matrix_label = f"C_FN={settings.cost_fn:g} / C_FP={settings.cost_fp:g}"
    data_dir = args.data_dir or (ROOT / "data" / "raw" / "kdd2015")
    enrollments, logs = load_kdd_cup(data_dir)
    if args.label_file:
        enrollments = read_enrollments(args.label_file)
        print(f"标签口径: {args.label_file}（{len(enrollments):,} 条，正类 {enrollments['dropout'].mean():.2%}）")
    print(f"数据: {data_dir.name}｜{len(enrollments):,} 选课 / {len(logs):,} 事件｜{matrix_label}")

    out_dir = ROOT / "artifacts/multiseed" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        # 特征工程与切分只做一次：同一 seed 下四个配置共享同一份数据与同一个测试集，
        # 否则「配对」就无从谈起（配对要求同一 seed 内所有配置面对同一批学生）。
        started = time.perf_counter()
        prep = make_trainer(seed, None, args.iterations, args.n_splits)
        data = prep.prepare(logs, enrollments)
        prep_s = time.perf_counter() - started
        print(
            f"\n[seed={seed}] 切分与特征 {prep_s:.0f}s｜训练 {len(data.X_train):,} / 测试 {len(data.X_test):,}"
            f"｜测试集正类 {data.y_test.mean():.2%}",
            flush=True,
        )

        for key in args.configs:
            spec = SPECS[key]
            npz = out_dir / f"pred_{key}_seed{seed}.npz"
            if args.reuse and npz.exists():
                blob = np.load(npz)
                threshold = float(blob["threshold"][0])
                metrics = classification_metrics(
                    data.y_test, blob["prob_test"], threshold=threshold,
                    cost_matrix=prep.cost_matrix,
                )
                metrics["threshold"] = round(threshold, 6)
                elapsed = 0.0
                tag_note = "（复用）"
            else:
                t0 = time.perf_counter()
                result = fit_evaluate(data, seed, spec, args.iterations, args.n_splits)
                metrics = result["metrics"]
                np.savez_compressed(
                    npz,
                    y_test=np.asarray(data.y_test).astype(int),
                    prob_test=result["prob_test"],
                    y_train=np.asarray(data.y_train).astype(int),
                    threshold=np.array([metrics["threshold"]]),
                )
                elapsed = time.perf_counter() - t0
                tag_note = f"({elapsed:.0f}s)"

            row = {"seed": seed, "config": key, "label": spec.label, "n_test": int(metrics["n_samples"])}
            row.update({k: round(float(metrics[k]), 4) for k in KEY_METRICS})
            row.update(fn=int(metrics["fn"]), fp=int(metrics["fp"]), elapsed_s=round(elapsed, 1))
            rows.append(row)
            print(
                f"  {key:<8} thr={metrics['threshold']:.4f}  AUC={metrics['roc_auc']:.4f}  "
                f"recall={metrics['recall']:.4f}  cost={metrics['cost_per_student']:7.2f}  "
                f"基线={metrics['baseline_cost_trivial']:.2f}  "
                f"↓{metrics['cost_reduction_vs_trivial']:+.2%}  {tag_note}",
                flush=True,
            )

    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, order: tuple[str, ...]) -> pd.DataFrame:
    """按配置汇总：均值、标准差、极差、以及「种子间波动 vs 配置间差异」的比。"""
    rows = []
    for key in order:
        sub = raw[raw["config"] == key]
        if sub.empty:
            continue
        row = {"config": key, "label": sub["label"].iloc[0], "n_seeds": int(len(sub))}
        for col in ("roc_auc", "recall", "precision", "brier", "cost_per_student",
                    "cost_reduction_vs_trivial", "predicted_positive_rate", "threshold"):
            row[f"{col}_mean"] = round(float(sub[col].mean()), 4)
            row[f"{col}_std"] = round(float(sub[col].std(ddof=1)), 4) if len(sub) > 1 else 0.0
        row["cost_min"] = round(float(sub["cost_per_student"].min()), 2)
        row["cost_max"] = round(float(sub["cost_per_student"].max()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def paired(raw: pd.DataFrame, baseline: str = "cb_cs", n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """逐种子配对差 + Bootstrap 区间。

    配对的意义：同一 seed 下所有配置面对同一批测试学生，因此「A 减 B」的差
    只反映模型，不掺切分噪声。区间用对配对差重采样得到（n=5 时区间偏窄，
    仅作量级参考，结论以「差 vs 种子间标准差」的对比为准）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key in raw["config"].unique():
        if key == baseline:
            continue
        a = raw[raw["config"] == baseline].set_index("seed")["cost_per_student"]
        b = raw[raw["config"] == key].set_index("seed")["cost_per_student"]
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        diff = (a.loc[common] - b.loc[common]).to_numpy(dtype="float64")
        boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
        rows.append({
            "pair": f"{baseline} − {key}",
            "n_seeds": len(diff),
            "diff_mean": round(float(diff.mean()), 3),
            "diff_std": round(float(diff.std(ddof=1)), 3),
            "diff_min": round(float(diff.min()), 3),
            "diff_max": round(float(diff.max()), 3),
            "ci_low": round(float(np.percentile(boots, 2.5)), 3),
            "ci_high": round(float(np.percentile(boots, 97.5)), 3),
            "crosses_zero": bool(np.percentile(boots, 2.5) <= 0 <= np.percentile(boots, 97.5)),
            "n_negative": int((diff < 0).sum()),
            "n_positive": int((diff > 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="真实数据多种子重复（E14）")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--tag", required=True, help="输出子目录名，如 official / early")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_ORDER), choices=list(SPECS))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--reuse", action="store_true", help="复用已有预测，只重算汇总")
    args = parser.parse_args()

    raw = run(args)
    report_dir = ROOT / "reports/multiseed" / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(report_dir / "raw.csv", index=False, encoding="utf-8-sig")

    order = tuple(c for c in DEFAULT_ORDER if c in raw["config"].unique())
    summary = summarize(raw, order)
    summary.to_csv(report_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pairs = paired(raw, n_boot=args.n_boot)
    pairs.to_csv(report_dir / "paired.csv", index=False, encoding="utf-8-sig")

    print("\n===== 汇总（均值 ± 标准差）=====")
    print(summary[["label", "n_seeds", "roc_auc_mean", "roc_auc_std",
                   "cost_per_student_mean", "cost_per_student_std",
                   "cost_min", "cost_max"]].to_string(index=False))
    print("\n===== 逐种子配对差（负数=左项更便宜）=====")
    print(pairs.to_string(index=False))
    print(f"\n产物: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
