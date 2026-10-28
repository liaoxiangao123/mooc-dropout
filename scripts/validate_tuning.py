"""E9b：把调参结果拿到测试集上验证，避免「只看 OOF 涨了就宣布胜利」。

为什么必须有这一步：
    E9 的目标函数是**训练集 OOF** 上的期望代价。超参是在同一批 OOF 上挑出来的，
    因此 OOF 上的提升天然偏乐观——挑得越久，过拟合 OOF 的程度越高。
    学术上唯一可信的做法是：在**完全没参与调参**的测试集上，比较「默认参数」与
    「调参后参数」的配对差异，并给出 Bootstrap 区间。

结论怎么写：
    * 区间不跨零 -> 调参带来了可分辨的收益，可在正文报告；
    * 区间跨零   -> 收益小于抽样噪声，只能写「未观察到显著差异」，
                    这与 E6/E7 关于「模型差异不可分辨」的结论一致。

用法：
    python scripts/validate_tuning.py                       # 用 reports/optuna/best_params.json
    python scripts/validate_tuning.py --folds 5 --n-boot 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rel(path: object) -> str:
    """报告里只写相对路径：绝对路径会把本机用户名带进产物，公开仓库时是隐私泄漏。"""
    try:
        return Path(str(path)).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return str(path).replace("\\", "/")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mooc_dropout.config import get_settings                      # noqa: E402
from mooc_dropout.data.loader import load_kdd_cup                 # noqa: E402
from mooc_dropout.evaluation.cost import CostMatrix, cost_optimal_threshold  # noqa: E402
from mooc_dropout.evaluation.metrics import classification_metrics  # noqa: E402
from mooc_dropout.training.trainer import TrainConfig, Trainer     # noqa: E402
from run_extra_experiments import paired_bootstrap                 # noqa: E402

DEFAULT_PARAMS = {"iterations": 400, "depth": 6, "learning_rate": 0.05}
REPORT = ROOT / "reports" / "调参验证-E9b.md"


def resolve_report(params_path: Path, explicit: Path | None) -> Path:
    """报告路径跟着参数文件走：reports/optuna/<tag>/best_params.json
    -> reports/调参验证-E9b-<数据集名>.md，避免真实数据的验证结论覆盖合成数据那份。"""
    if explicit is not None:
        return explicit
    tag = Path(params_path).resolve().parent.name
    if tag == "optuna":
        return REPORT
    from tune import tag_label

    return ROOT / "reports" / f"调参验证-E9b-{tag_label(tag)}.md"


def evaluate(data, params: dict, matrix: CostMatrix, folds: int, seed: int):
    """训练 -> OOF 选阈值 -> 测试集评估，返回 (测试概率, 阈值, 指标)。"""
    trainer = Trainer(
        TrainConfig.from_settings(
            get_settings(), seed=seed, model_name="catboost", n_splits=folds,
            model_params=dict(params),
        ),
        get_settings(),
    )
    model = trainer.create_model()
    model.fit(data.X_train, data.y_train)

    oof = trainer._oof_probabilities(data)
    threshold = float(cost_optimal_threshold(data.y_train, oof, matrix)["threshold"])
    prob = model.predict_proba(data.X_test)
    metrics = classification_metrics(data.y_test, prob, threshold=threshold, cost_matrix=matrix)
    return prob, threshold, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="在测试集上验证调参收益")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--params", type=Path, default=ROOT / "reports" / "optuna" / "best_params.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--report", type=Path, default=None, help="报告路径（默认按参数文件所在目录推导）")
    args = parser.parse_args()
    report = resolve_report(args.params, args.report)

    settings = get_settings()
    matrix = CostMatrix(c_fn=settings.cost_fn, c_fp=settings.cost_fp)

    payload = json.loads(args.params.read_text(encoding="utf-8"))
    tuned = dict(payload.get("best_params", payload))
    print(f"默认参数：{DEFAULT_PARAMS}")
    print(f"调参参数：{tuned}")

    enrollments, logs = load_kdd_cup(args.data_dir)
    trainer = Trainer(
        TrainConfig.from_settings(settings, seed=args.seed, model_name="catboost"), settings
    )
    data = trainer.prepare(logs, enrollments)
    print(f"训练 {len(data.X_train):,} / 测试 {len(data.X_test):,}（测试集未参与调参）")

    results = {}
    for name, params in (("default", DEFAULT_PARAMS), ("tuned", tuned)):
        started = time.perf_counter()
        prob, thr, metrics = evaluate(data, params, matrix, args.folds, args.seed)
        results[name] = {"prob": prob, "threshold": thr, "metrics": metrics}
        print(
            f"  {name:<8} thr={thr:.3f}  AUC={metrics['roc_auc']:.4f}  "
            f"recall={metrics['recall']:.3f}  cost=¥{metrics['cost_per_student']:.2f}  "
            f"({time.perf_counter() - started:.0f}s)"
        )

    base, best = results["default"], results["tuned"]
    delta = paired_bootstrap(
        data.y_test, best["prob"], best["threshold"], base["prob"], base["threshold"],
        n_boot=args.n_boot,
    )

    rows = []
    for label, key in (("默认参数", "default"), ("调参后", "tuned")):
        m = results[key]["metrics"]
        rows.append(
            {
                "配置": label,
                "阈值": round(results[key]["threshold"], 4),
                "ROC-AUC": round(float(m["roc_auc"]), 4),
                "PR-AUC": round(float(m["pr_auc"]), 4),
                "召回率": round(float(m["recall"]), 4),
                "精确率": round(float(m["precision"]), 4),
                "Brier": round(float(m["brier"]), 4),
                "人均代价(元)": round(float(m["cost_per_student"]), 2),
                "平凡基线(元)": round(float(m["baseline_cost_trivial"]), 2),
                "预警率": f"{float(m['predicted_positive_rate']):.2%}",
                "漏报": int(m["fn"]),
                "误报": int(m["fp"]),
            }
        )
    table = pd.DataFrame(rows)

    gain = float(base["metrics"]["cost_per_student"]) - float(best["metrics"]["cost_per_student"])
    crosses_zero = delta["跨零"] == "是"

    lines = [
        "# E9b · 调参收益的测试集验证",
        "",
        f"- 数据：`{_rel(args.data_dir)}`（训练 {len(data.X_train):,} / 测试 {len(data.X_test):,}）",
        f"- 阈值：两配置各自在训练集 {args.folds} 折 OOF 上重选，测试集只做评估",
        f"- 参数来源：`{_rel(args.params)}`",
        "",
        "## 测试集结果",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "## 配对 Bootstrap（调参后 − 默认，负值表示更便宜）",
        "",
        "| 观测差(元) | 重采样均值 | 95% 下限 | 95% 上限 | 跨零 | P(差值>0) |",
        "|---|---|---|---|---|---|",
        f"| {delta['观测差(元)']} | {delta['重采样均值差']} | {delta['下限']} | "
        f"{delta['上限']} | {delta['跨零']} | {delta['P(差值>0)']} |",
        "",
        "## 结论",
        "",
    ]
    # 分辨率＝Bootstrap 区间半宽：比它更小的差异在这份测试集上无法与抽样波动区分。
    # 这个数依赖测试集规模（合成 822 人 → 约 1.2 元，真实 2.4 万人 → 约 0.2 元），
    # 所以必须算出来而不是写死。
    half_width = round((float(delta["上限"]) - float(delta["下限"])) / 2, 2)
    n_test = len(data.y_test)

    if crosses_zero:
        lines += [
            f"测试集上人均代价下降 **{gain:.2f} 元**，但 95% 区间 "
            f"[{delta['下限']}, {delta['上限']}] **跨零**，即收益与抽样噪声不可分辨。",
            "",
            "因此 E9 只能作为「调参流程可用」的工程证据，**不能写成「调参显著提升了效果」**。",
            f"这份测试集 n = {n_test:,}，人均代价的抽样分辨率（区间半宽）为 {half_width:.2f} 元/人，"
            f"本次观测到的差异 {abs(gain):.2f} 元落在分辨率之内。",
        ]
    else:
        lines += [
            f"测试集上人均代价下降 **{gain:.2f} 元**，95% 区间 "
            f"[{delta['下限']}, {delta['上限']}] 不跨零，调参收益具备统计可分辨性"
            f"（测试集 n = {n_test:,}，分辨率 {half_width:.2f} 元/人）。",
            "",
            "仍需注意：区间较窄时也要结合 E6 的种子间标准差一并讨论，避免把「同一份测试集上的"
            "偶然优势」当作稳定收益。",
        ]

    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + table.to_string(index=False))
    print("\n配对 Bootstrap：", delta)
    print(f"\n报告：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
