"""E9：超参搜索 —— 目标函数是「期望代价」，不是 AUC。

为什么不用 AUC 做目标：
    AUC 只衡量排序，不衡量「在哪个阈值上切」。本课题的结论是「决策链路优于模型」，
    若调参以 AUC 为目标，最优参数很可能在代价上并不是最优 —— 二者在类别不平衡时
    经常给出不同的偏好（AUC 偏爱整体排序，代价偏爱少数类的高分区）。
    因此这里直接最小化「OOF 上的期望代价」，与第 4 章的决策规则同源。

为什么在 OOF 上搜：
    与阈值选择同理，测试集全程不参与。用测试集调参是隐性数据穿越，会让论文里的
    提升幅度被系统性高估。代价是 OOF 代价会偏乐观，故最终仍需在同一测试集上
    与默认参数做配对比较（见 E9b）。

搜索成本控制：
    每个 trial 做 K 折 OOF（默认 3 折），每完成一折向 pruner 报告一次中间代价，
    明显劣于中位数的 trial 提前剪枝。相比「训练完再评估」，实测能省 40% 左右时间。

用法：
    python scripts/tune.py --trials 30
    python scripts/tune.py --trials 60 --n-splits 5 --data-dir data/raw/kdd2015
    python scripts/tune.py --trials 30 --calibration isotonic   # 校准后再评估代价

产出：
    reports/optuna/trials.csv          每个 trial 的参数与指标
    reports/optuna/best_params.json    最优参数（可直接喂给 scripts/train.py --params-json）
    reports/optuna/study.db            Optuna 数据库，可 resume
    reports/调参实验-E9.md             论文可直接引用的汇总表
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rel(path: object) -> str:
    """报告里只写相对路径：绝对路径会把本机用户名带进产物，公开仓库时是隐私泄漏。"""
    try:
        return str(Path(str(path)).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import get_settings                      # noqa: E402
from mooc_dropout.data.loader import load_kdd_cup                 # noqa: E402
from mooc_dropout.evaluation.cost import CostMatrix, cost_optimal_threshold  # noqa: E402
from mooc_dropout.evaluation.metrics import classification_metrics  # noqa: E402
from mooc_dropout.models.calibration import ProbabilityCalibrator  # noqa: E402
from mooc_dropout.models.registry import build_model              # noqa: E402
from mooc_dropout.training.trainer import TrainConfig, Trainer     # noqa: E402

OUT_DIR = ROOT / "reports" / "optuna"
REPORT_MD = ROOT / "reports" / "调参实验-E9.md"
DEFAULT_PARAMS = {"iterations": 400, "depth": 6, "learning_rate": 0.05}
CATEGORICAL = ["course_id", "username"]


def suggest_params(trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 900, step=50),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
    }


def oof_evaluate(
    data,
    params: dict[str, Any],
    matrix: CostMatrix,
    *,
    seed: int,
    n_splits: int,
    calibration: str,
    trial=None,
) -> dict[str, float]:
    """在训练集上做分组 K 折，返回 OOF 代价与辅助指标。

    每折结束后向 Optuna 报告中间代价；未填满的位置不参与统计，
    否则零值会被当成「低风险」而人为压低代价。
    """
    from mooc_dropout.data.split import oof_group_splits

    n = len(data.train_enrollments)
    oof = np.full(n, np.nan, dtype="float64")
    positions = pd.Series(
        np.arange(n), index=data.train_enrollments["enrollment_id"].astype(str).to_numpy()
    )

    for fold, (tr_idx, va_idx) in enumerate(
        oof_group_splits(data.train_enrollments, n_splits=n_splits, seed=seed), start=1
    ):
        sub_train = data.train_enrollments.iloc[tr_idx]
        sub_valid = data.train_enrollments.iloc[va_idx]
        train_mask = data.train_logs["enrollment_id"].isin(set(sub_train["enrollment_id"]))
        valid_mask = data.train_logs["enrollment_id"].isin(set(sub_valid["enrollment_id"]))

        X_tr = data.builder.transform(data.train_logs.loc[train_mask], sub_train)
        X_va = data.builder.transform(data.train_logs.loc[valid_mask], sub_valid)
        y_tr = sub_train.set_index("enrollment_id")["dropout"].reindex(X_tr.index).to_numpy()

        model = build_model(
            "catboost",
            cost_ratio=matrix.ratio,
            weighting="class_weight",
            random_seed=seed,
            categorical_features=CATEGORICAL,
            **params,
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_va)
        for enrollment_id, p in zip(X_va.index, prob):
            oof[int(positions[enrollment_id])] = p

        filled = ~np.isnan(oof)
        if trial is not None and filled.sum() > 0:
            interim = _cost_at_optimal_threshold(data.y_train[filled], oof[filled], matrix)
            trial.report(interim, fold)
            if trial.should_prune():
                raise _Pruned(interim)

    y = np.asarray(data.y_train).astype(int)
    prob = np.nan_to_num(oof, nan=0.0)

    calibrator = ProbabilityCalibrator(calibration)
    if calibration != "none":
        calibrator.fit(prob, y)
        prob = calibrator.transform(prob)

    threshold = float(cost_optimal_threshold(y, prob, matrix)["threshold"])
    metrics = classification_metrics(y, prob, threshold=threshold, cost_matrix=matrix)
    return {
        "per_student_cost": float(metrics["cost_per_student"]),
        "threshold": threshold,
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "recall": float(metrics["recall"]),
        "precision": float(metrics["precision"]),
        "brier": float(metrics["brier"]),
    }


class _Pruned(Exception):
    """内部信号：由 Optuna 的 TrialPruned 转换而来，避免在派生异常上做字符串判断。"""


def _cost_at_optimal_threshold(y, prob, matrix: CostMatrix) -> float:
    thr = float(cost_optimal_threshold(y, prob, matrix)["threshold"])
    return matrix.per_student_cost(np.asarray(y).astype(int), (np.asarray(prob) >= thr).astype(int))


def main() -> int:
    parser = argparse.ArgumentParser(description="以期望代价为目标的 CatBoost 超参搜索")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", default="none", choices=["none", "platt", "isotonic"])
    parser.add_argument("--timeout", type=int, default=None, help="秒；到时即停")
    parser.add_argument("--resume", action="store_true", help="复用 reports/optuna/study.db")
    args = parser.parse_args()

    import optuna
    from optuna.exceptions import TrialPruned

    settings = get_settings()
    matrix = CostMatrix(c_fn=settings.cost_fn, c_fp=settings.cost_fp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    enrollments, logs = load_kdd_cup(args.data_dir)
    trainer = Trainer(
        TrainConfig.from_settings(
            settings, seed=args.seed, model_name="catboost", n_splits=args.n_splits
        ),
        settings,
    )
    print(f"数据: {args.data_dir}  选课 {len(enrollments):,}  事件 {len(logs):,}")
    print("准备特征（只做一次，所有 trial 复用）...")
    data = trainer.prepare(logs, enrollments)
    print(f"训练集 {len(data.X_train):,} 条 / 测试集 {len(data.X_test):,} 条（测试集不参与调参）")

    # ---- 基线 ----
    print("\n[基线] 默认参数 ", DEFAULT_PARAMS)
    t0 = time.perf_counter()
    base = oof_evaluate(
        data, dict(DEFAULT_PARAMS), matrix, seed=args.seed,
        n_splits=args.n_splits, calibration=args.calibration,
    )
    base_elapsed = time.perf_counter() - t0
    print(
        f"  OOF 人均代价 ¥{base['per_student_cost']:.2f}  AUC={base['roc_auc']:.4f}  "
        f"thr={base['threshold']:.3f}  ({base_elapsed:.0f}s)"
    )

    # ---- 搜索 ----
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    storage = f"sqlite:///{(OUT_DIR / 'study.db').as_posix()}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"mooc_cost_{args.calibration}",
        storage=storage,
        load_if_exists=args.resume,
    )

    def objective(trial):
        params = suggest_params(trial)
        try:
            result = oof_evaluate(
                data, params, matrix, seed=args.seed, n_splits=args.n_splits,
                calibration=args.calibration, trial=trial,
            )
        except _Pruned:
            raise TrialPruned()
        for key, value in result.items():
            trial.set_user_attr(key, value)
        return result["per_student_cost"]

    print(f"\n[E9] 开始搜索：{args.trials} 个 trial，{args.n_splits} 折 OOF")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    print(f"完成 {len(completed)} / 剪枝 {len(pruned)} / 总计 {len(study.trials)}")

    best = study.best_trial
    print("\n最优参数:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")
    print(f"OOF 人均代价 ¥{best.value:.2f}（基线 ¥{base['per_student_cost']:.2f}）")

    # ---- 落盘 ----
    rows = []
    for trial in study.trials:
        row = {"trial": trial.number, "state": trial.state.name, "cost": trial.value}
        row.update(trial.params)
        row.update({f"metric_{k}": v for k, v in trial.user_attrs.items()})
        rows.append(row)
    trials_df = pd.DataFrame(rows).sort_values("trial")
    trials_df.to_csv(OUT_DIR / "trials.csv", index=False, encoding="utf-8-sig")

    baseline_gap = base["per_student_cost"] - float(best.value)
    best_payload = {
        "best_params": dict(best.params),
        "objective": "oof_cost_per_student",
        "calibration": args.calibration,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_trials": len(study.trials),
        "best_oof_cost": round(float(best.value), 4),
        "baseline_oof_cost": round(float(base["per_student_cost"]), 4),
        "baseline_params": DEFAULT_PARAMS,
        "best_metrics": {k: round(float(v), 4) for k, v in best.user_attrs.items()},
        "data_dir": _rel(args.data_dir),
    }
    (OUT_DIR / "best_params.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 可直接引用的汇总 ----
    top = trials_df.loc[trials_df["state"] == "COMPLETE"].nsmallest(10, "cost")
    param_cols = list(DEFAULT_PARAMS) + ["l2_leaf_reg", "random_strength",
                                          "bagging_temperature", "border_count"]
    top = top[["trial", "cost"] + [c for c in param_cols if c in top.columns]]
    top.insert(1, "相对基线", (base["per_student_cost"] - top["cost"]).round(2))

    lines = [
        "# E9 · 以期望代价为目标的超参搜索",
        "",
        f"- 数据：`{_rel(args.data_dir)}`（选课 {len(enrollments):,} 条，事件 {len(logs):,} 条）",
        f"- 目标函数：训练集 {args.n_splits} 折 OOF 上的人均期望代价（测试集不参与）",
        f"- 搜索：TPE + MedianPruner，{len(study.trials)} 个 trial，"
        f"完成 {len(completed)}、剪枝 {len(pruned)}",
        f"- 校准：{args.calibration}（选择阈值与校准器均在同一 OOF 尺度上）",
        "",
        "## 基线 vs 最优",
        "",
        "| 配置 | 参数 | OOF 人均代价(元) | ROC-AUC | 阈值 |",
        "|---|---|---|---|---|",
        f"| 默认 | {json.dumps(DEFAULT_PARAMS, ensure_ascii=False)} | "
        f"{base['per_student_cost']:.2f} | {base['roc_auc']:.4f} | {base['threshold']:.3f} |",
        f"| 调参后 | {json.dumps(best.params, ensure_ascii=False)} | "
        f"{float(best.value):.2f} | {best.user_attrs.get('roc_auc', float('nan')):.4f} | "
        f"{best.user_attrs.get('threshold', float('nan')):.3f} |",
        "",
        f"OOF 代价下降 **{baseline_gap:.2f} 元/人**（{baseline_gap / base['per_student_cost']:.1%}）。",
        "",
        "## 前 10 名 trial",
        "",
        "| " + " | ".join(str(c) for c in top.columns) + " |",
        "|" + "---|" * len(top.columns),
    ]
    for _, row in top.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "## 必须写进论文的限定",
        "",
        "1. 上表是 **OOF** 代价，天然偏乐观；调参带来的真实收益需在测试集上与默认参数做配对比较，",
        "   不能直接把 OOF 的降幅写进结论。",
        "2. 搜索预算（trial 数）有限，最优解只是「当前预算下的最好」，不构成全局最优。",
        "3. 若调参收益小于多种子重复的种子间标准差，应如实说明「差异不可分辨」，",
        "   这与 E6/E7 的结论一致。",
        "",
        "复现：",
        "",
        "```bash",
        f"python scripts/tune.py --trials {args.trials} --n-splits {args.n_splits} "
        f"--calibration {args.calibration} --resume",
        "python scripts/train.py --data-dir data/synthetic --params-json reports/optuna/best_params.json",
        "```",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n结果目录: {OUT_DIR}")
    print(f"汇总报告: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
