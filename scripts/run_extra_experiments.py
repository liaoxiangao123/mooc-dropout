"""论文补充实验：多种子重复（E6）、Bootstrap 置信区间（E7）、概率校准（E8）。

对应论文附录 D 的第 3、4 项待办：
    3. 补充多种子重复与 Bootstrap 置信区间 —— 使模型比较具备统计意义
    4. 补做概率校准实验（Platt / 保序回归） —— 校准后重选阈值

方法学立场（写进论文时必须保持）：
    * 每次重复同时改变「数据切分」与「模型随机种子」。只换模型种子而切分不变，
      测到的只是优化过程的随机性，不是结论的稳定性；反过来只换切分，则混淆了
      「这份测试集恰好容易」与「模型确实更稳」。
    * 同一 seed 下各配置共享切分，因此可以逐 seed 配对求差，比「分别求置信区间
      看是否重叠」灵敏得多——后者在相关性高时会严重低估差异的显著性。
    * 校准器只在训练集的 OOF 预测上拟合，测试集全程不参与。用测试集调校准参数
      是最常见的隐性数据穿越。
    * 校准改变了概率尺度，旧阈值随即失效，必须在校准后的同一尺度上重选阈值。
      漏掉这一步，「校准后效果反而变差」几乎是必然的。

用法：
    python scripts/run_extra_experiments.py                       # 全跑
    python scripts/run_extra_experiments.py --seeds 42 7 2024     # 指定种子
    python scripts/run_extra_experiments.py --configs cb_cs cb_none lr
    python scripts/run_extra_experiments.py --reuse               # 复用已有预测，不重训
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mooc_dropout.config import get_settings                      # noqa: E402
from mooc_dropout.data.loader import load_kdd_cup                 # noqa: E402
from mooc_dropout.evaluation.cost import (                        # noqa: E402
    CostMatrix,
    cost_optimal_threshold,
)
from mooc_dropout.evaluation.metrics import classification_metrics  # noqa: E402
from mooc_dropout.training.trainer import TrainConfig, Trainer      # noqa: E402

DATA_DIR = ROOT / "data" / "synthetic"
REPORT_DIR = ROOT / "reports"
PRED_DIR = ROOT / "artifacts" / "extra"
FIG_DIR = ROOT / "docs" / "figs"

# 全局代价矩阵：与主实验同一口径，所有函数共用同一份，避免各函数自己取值导致口径漂移
_SETTINGS = get_settings()
MATRIX = CostMatrix(c_fn=_SETTINGS.cost_fn, c_fp=_SETTINGS.cost_fp)

# 与正文 5.1 节主实验完全一致，否则补充实验不能用来支撑正文结论
MODEL_PARAMS = {"iterations": 400, "depth": 6, "learning_rate": 0.05}
MAIN_SEED = 42
KEY_METRICS = ("threshold", "roc_auc", "pr_auc", "recall", "precision", "f1",
               "brier", "cost_per_student", "predicted_positive_rate")


# ---------------------------------------------------------------- 实验配置


@dataclass(frozen=True)
class ModelSpec:
    """一个待评配置。

    cost_ratio 作用在训练权重上：CatBoost 取 10（代价敏感），基线一律取 1
    （代价中立）。若让基线也按 10 加权，配置之间的差异就不再是"模型差异"，
    而是"是否做代价敏感"的差异，比较失去意义。
    """

    key: str
    label: str
    model_name: str
    weighting: str
    cost_ratio: float
    params: dict[str, Any] = field(default_factory=dict)


SPECS: dict[str, ModelSpec] = {
    "cb_cs": ModelSpec(
        "cb_cs", "CatBoost（代价敏感）", "catboost", "class_weight", 10.0,
        dict(MODEL_PARAMS),
    ),
    "cb_none": ModelSpec(
        "cb_none", "CatBoost（无代价）", "catboost", "none", 1.0,
        dict(MODEL_PARAMS),
    ),
    "hgbdt": ModelSpec(
        "hgbdt", "HGBDT（代价中立）", "hist_gbdt", "none", 1.0, {},
    ),
    "lr": ModelSpec(
        "lr", "逻辑回归（代价中立）", "logreg", "none", 1.0, {},
    ),
}
DEFAULT_ORDER = ["cb_cs", "cb_none", "hgbdt", "lr"]


# ---------------------------------------------------------------- 单次运行


def build_trainer(seed: int, spec: ModelSpec) -> Trainer:
    settings = get_settings()
    params: dict[str, Any] = dict(spec.params)
    params["cost_ratio"] = spec.cost_ratio
    params["weighting"] = spec.weighting
    cfg = TrainConfig.from_settings(
        settings,
        seed=seed,
        model_name=spec.model_name,
        weighting=spec.weighting,
        model_params=params,
        n_splits=5,
    )
    return Trainer(cfg, settings)


@dataclass
class Outcome:
    """一次「训练—选阈值—评估」的完整产物，供后续统计与重采样复用。"""

    key: str
    label: str
    seed: int
    threshold: float
    y_test: np.ndarray
    prob_test: np.ndarray
    y_train: np.ndarray
    oof_train: np.ndarray
    metrics: dict[str, float]

    @property
    def metrics_at_05(self) -> dict[str, float]:
        """固定 0.5 阈值下的指标——用来量化「不做阈值决策」会损失多少。"""
        return classification_metrics(
            self.y_test, self.prob_test, threshold=0.5, cost_matrix=MATRIX
        )


def run_once(seed: int, spec: ModelSpec) -> Outcome:
    trainer = build_trainer(seed, spec)
    enrollments, logs = load_kdd_cup(DATA_DIR)
    data = trainer.prepare(logs, enrollments)

    model = trainer.create_model()
    model.fit(data.X_train, data.y_train)

    # 阈值只能在训练集的 OOF 预测上选，测试集不参与
    oof = trainer._oof_probabilities(data)
    threshold = float(cost_optimal_threshold(data.y_train, oof, MATRIX)["threshold"])

    prob_test = model.predict_proba(data.X_test)
    metrics = classification_metrics(
        data.y_test, prob_test, threshold=threshold, cost_matrix=MATRIX
    )
    return Outcome(
        key=spec.key,
        label=spec.label,
        seed=seed,
        threshold=threshold,
        y_test=np.asarray(data.y_test).astype(int),
        prob_test=np.asarray(prob_test, dtype="float64"),
        y_train=np.asarray(data.y_train).astype(int),
        oof_train=np.asarray(oof, dtype="float64"),
        metrics=metrics,
    )


def save_outcome(out: Outcome) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRED_DIR / f"pred_{out.key}_seed{out.seed}.npz",
        y_test=out.y_test,
        prob_test=out.prob_test,
        y_train=out.y_train,
        oof_train=out.oof_train,
        threshold=np.array([out.threshold]),
    )


def load_outcome(key: str, seed: int) -> Outcome | None:
    path = PRED_DIR / f"pred_{key}_seed{seed}.npz"
    if not path.exists():
        return None
    blob = np.load(path)
    spec = SPECS[key]
    threshold = float(blob["threshold"][0])
    metrics = classification_metrics(
        blob["y_test"], blob["prob_test"], threshold=threshold, cost_matrix=MATRIX
    )
    return Outcome(
        key=key,
        label=spec.label,
        seed=seed,
        threshold=threshold,
        y_test=blob["y_test"],
        prob_test=blob["prob_test"],
        y_train=blob["y_train"],
        oof_train=blob["oof_train"],
        metrics=metrics,
    )


# ---------------------------------------------------------------- E6 多种子


def experiment_multiseed(
    seeds: list[int], keys: list[str], *, reuse: bool
) -> tuple[pd.DataFrame, dict[tuple[str, int], Outcome]]:
    rows: list[dict[str, Any]] = []
    outcomes: dict[tuple[str, int], Outcome] = {}

    for seed in seeds:
        enr, logs = load_kdd_cup(DATA_DIR)  # 仅用于打印切分规模，避免重复读盘
        for key in keys:
            spec = SPECS[key]
            out = load_outcome(key, seed) if reuse else None
            elapsed = 0.0
            if out is None:
                started = time.perf_counter()
                out = run_once(seed, spec)
                save_outcome(out)
                elapsed = time.perf_counter() - started
            outcomes[(key, seed)] = out

            row = {"seed": seed, "config": key, "label": spec.label}
            row.update({k: round(float(out.metrics[k]), 4) for k in KEY_METRICS})
            row["fn"] = int(out.metrics["fn"])
            row["fp"] = int(out.metrics["fp"])
            row["n_test"] = int(out.metrics["n_samples"])
            m05 = out.metrics_at_05
            row["recall_at_05"] = round(float(m05["recall"]), 4)
            row["cost_at_05"] = round(float(m05["cost_per_student"]), 2)
            row["elapsed_s"] = round(elapsed, 1)
            rows.append(row)

            print(
                f"  seed={seed:<5} {key:<8} thr={out.threshold:.3f}  "
                f"AUC={out.metrics['roc_auc']:.4f}  rec={out.metrics['recall']:.3f}  "
                f"cost={out.metrics['cost_per_student']:6.2f}  "
                f"Brier={out.metrics['brier']:.4f}"
                + (f"  ({elapsed:.0f}s)" if elapsed else "  (cached)"),
                flush=True,
            )
    return pd.DataFrame(rows), outcomes


def summarize_multiseed(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """按配置汇总为「均值 ± 标准差」，并给出极差与变异系数。"""
    rows = []
    for key in order:
        sub = df.loc[df["config"] == key]
        if sub.empty:
            continue
        row: dict[str, Any] = {"配置": sub["label"].iloc[0], "n": int(len(sub))}
        for metric in ("threshold", "roc_auc", "recall", "precision", "brier",
                       "cost_per_student", "recall_at_05", "cost_at_05"):
            vals = sub[metric].astype(float)
            row[f"{metric}"] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
        costs = sub["cost_per_student"].astype(float)
        row["代价极差"] = f"{costs.max() - costs.min():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(
    df: pd.DataFrame, seeds: list[int], base: str, order: list[str]
) -> pd.DataFrame:
    """逐 seed 求「配置 − 基准」的差值，仅当同一 seed 两边都有结果。"""
    from scipy import stats

    rows = []
    for key in order:
        if key == base:
            continue
        left = df.loc[df["config"] == key].set_index("seed")["cost_per_student"]
        right = df.loc[df["config"] == base].set_index("seed")["cost_per_student"]
        common = [s for s in seeds if s in left.index and s in right.index]
        if len(common) < 2:
            continue
        delta = (left.loc[common] - right.loc[common]).to_numpy(dtype="float64")
        t_stat, p_val = stats.ttest_rel(left.loc[common], right.loc[common])
        rows.append(
            {
                "对比": f"{SPECS[key].label} − {SPECS[base].label}",
                "均值差(元)": round(float(delta.mean()), 2),
                "标准差": round(float(delta.std(ddof=1)), 2),
                "方向一致的seed数": f"{int((delta > 0).sum())}/{len(delta)} 偏高",
                "配对t": round(float(t_stat), 2),
                "p值": round(float(p_val), 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E7 Bootstrap


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """测试集重采样的 95% 置信区间（百分位法）。

    点估计用**原始测试集**上的值，而不是重采样均值——后者会把区间向中心拉，
    与论文表格里报告的数值对不上，读者一算就发现不一致。
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    n = y_true.size

    point = classification_metrics(y_true, y_prob, threshold=threshold, cost_matrix=MATRIX)

    tracked = ("cost_per_student", "recall", "precision", "pr_auc", "brier")
    draws: dict[str, list[float]] = {k: [] for k in tracked}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():     # 重采样后只剩单一类别，指标无定义
            continue
        m = classification_metrics(yt, y_prob[idx], threshold=threshold, cost_matrix=MATRIX)
        for k in tracked:
            draws[k].append(float(m[k]))

    labels = {
        "cost_per_student": "人均代价(元)",
        "recall": "召回率",
        "precision": "精确率",
        "pr_auc": "PR-AUC",
        "brier": "Brier",
    }
    rows = []
    for k in tracked:
        arr = np.asarray(draws[k], dtype="float64")
        arr = arr[~np.isnan(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append(
            {
                "指标": labels[k],
                "点估计": round(float(point[k]), 4),
                "下限(2.5%)": round(float(lo), 4),
                "上限(97.5%)": round(float(hi), 4),
                "区间宽度": round(float(hi - lo), 4),
                "有效重采样": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    thr_a: float,
    prob_b: np.ndarray,
    thr_b: float,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """配对 Bootstrap：同一批重采样样本上比较两种决策的人均代价差。

    因为两个方案作用在同一批学生上，误差高度相关；分别求置信区间看是否重叠
    会严重低估差异的显著性，必须逐次配对求差再统计。
    差值为正表示 A 比 B 贵。
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    pred_a = (np.asarray(prob_a, dtype="float64") >= thr_a).astype(int)
    pred_b = (np.asarray(prob_b, dtype="float64") >= thr_b).astype(int)
    n = y_true.size

    deltas = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        deltas[i] = MATRIX.per_student_cost(yt, pred_a[idx]) - MATRIX.per_student_cost(
            yt, pred_b[idx]
        )

    obs = MATRIX.per_student_cost(y_true, pred_a) - MATRIX.per_student_cost(y_true, pred_b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "观测差(元)": round(float(obs), 2),
        "重采样均值差": round(float(deltas.mean()), 2),
        "下限": round(float(lo), 2),
        "上限": round(float(hi), 2),
        "跨零": "是" if lo <= 0 <= hi else "否",
        "P(差值>0)": round(float((deltas > 0).mean()), 4),
    }


# ---------------------------------------------------------------- E8 校准


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """等宽分箱的 ECE：各箱 |平均预测概率 − 实际正例率| 按样本量加权。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_calibrators(y_train: np.ndarray, oof: np.ndarray) -> dict[str, Any]:
    """在 OOF 预测上拟合校准器。OOF 来自训练集 K 折，测试集完全不参与。"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(_logit(oof).reshape(-1, 1), np.asarray(y_train).astype(int))

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(oof, dtype="float64"), np.asarray(y_train).astype(int))
    return {"platt": platt, "isotonic": iso}


def apply_calibrator(name: str, cal: dict[str, Any], prob: np.ndarray) -> np.ndarray:
    if name == "none":
        return np.asarray(prob, dtype="float64")
    if name == "platt":
        return cal["platt"].predict_proba(_logit(prob).reshape(-1, 1))[:, 1]
    return np.clip(
        cal["isotonic"].predict(np.asarray(prob, dtype="float64")), 0.0, 1.0
    )


CAL_LABELS = {
    "none": "未校准",
    "platt": "Platt（Sigmoid）",
    "isotonic": "保序回归（Isotonic）",
}


def experiment_calibration(
    outcomes: dict[tuple[str, int], Outcome], seeds: list[int], key: str
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    """对指定配置在每个种子上做校准，报告跨种子均值。

    同时给出「校准后按 0.5 决策」的一列：若校准真的把概率还原成了后验，
    0.5 就应当接近代价最优阈值，这一列是校准有效性的直接检验。
    """
    rows = []
    curves: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        out = outcomes.get((key, seed))
        if out is None:
            continue
        cal = fit_calibrators(out.y_train, out.oof_train)
        per_seed: dict[str, np.ndarray] = {}
        for name in ("none", "platt", "isotonic"):
            oof_c = apply_calibrator(name, cal, out.oof_train)
            prob_c = apply_calibrator(name, cal, out.prob_test)
            # 校准改变了概率尺度，阈值必须在同一尺度上重选
            thr = float(cost_optimal_threshold(out.y_train, oof_c, MATRIX)["threshold"])
            m = classification_metrics(
                out.y_test, prob_c, threshold=thr, cost_matrix=MATRIX
            )
            m05 = classification_metrics(
                out.y_test, prob_c, threshold=0.5, cost_matrix=MATRIX
            )
            rows.append(
                {
                    "seed": seed,
                    "校准方式": CAL_LABELS[name],
                    "cal": name,
                    "阈值": round(thr, 4),
                    "Brier": round(float(m["brier"]), 4),
                    "ECE": round(expected_calibration_error(out.y_test, prob_c), 4),
                    "召回率": round(float(m["recall"]), 4),
                    "精确率": round(float(m["precision"]), 4),
                    "人均代价": round(float(m["cost_per_student"]), 2),
                    "0.5阈值人均代价": round(float(m05["cost_per_student"]), 2),
                }
            )
            per_seed[name] = prob_c
        curves[seed] = per_seed
    return pd.DataFrame(rows), curves


def summarize_calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cal in ("none", "platt", "isotonic"):
        sub = df.loc[df["cal"] == cal]
        if sub.empty:
            continue
        row = {"校准方式": CAL_LABELS[cal]}
        for col in ("阈值", "Brier", "ECE", "召回率", "精确率", "人均代价",
                    "0.5阈值人均代价"):
            vals = sub[col].astype(float)
            row[col] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}" if len(vals) > 1 \
                else f"{vals.mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reliability(
    out: Outcome, curves: dict[str, np.ndarray], path: Path
) -> bool:
    """可靠性图：校准前后「预测概率」与「实际辍学率」的贴合程度。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
    except Exception:
        return False

    # 必须一次性给出候选列表：若在循环里逐次赋值，最终只剩最后一个元素，
    # 中文字形会全部回退成方框（本脚本曾因此出过一版废图）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)
    styles = {
        "none": ("未校准", "#6B7280", "--", "s"),
        "platt": ("Platt 校准", "#1F3864", "-", "o"),
        "isotonic": ("保序回归校准", "#C55A11", "-", "^"),
    }
    for name, (label, color, ls, marker) in styles.items():
        if name not in curves:
            continue
        frac, mean_pred = calibration_curve(
            out.y_test, curves[name], n_bins=8, strategy="quantile"
        )
        ax.plot(mean_pred, frac, ls, color=color, marker=marker, ms=4.5, lw=1.6, label=label)

    ax.plot([0, 1], [0, 1], ":", color="#9CA3AF", lw=1.3, label="理想校准")
    ax.set_xlabel("模型输出的辍学概率", fontsize=10.5)
    ax.set_ylabel("实际辍学比例", fontsize=10.5)
    ax.set_title("校准前后的可靠性曲线（测试集，分位分箱）", fontsize=11.5, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- 输出


def to_markdown(title: str, df: pd.DataFrame) -> str:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(str(c) for c in df.columns) + " |")
    lines.append("|" + "---|" * len(df.columns))
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> int:
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据目录（默认 data/synthetic）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024, 1234, 3407])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_ORDER, choices=list(SPECS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--reuse", action="store_true", help="复用 artifacts/extra 中已有预测")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    print(f"数据目录: {DATA_DIR}")

    for d in (REPORT_DIR, PRED_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    seeds, keys = args.seeds, args.configs
    print(f"[E6] 多种子重复  seeds={seeds}  configs={keys}")
    raw, outcomes = experiment_multiseed(seeds, keys, reuse=args.reuse)
    raw.to_csv(PRED_DIR / "multiseed_raw.csv", index=False, encoding="utf-8-sig")

    summary = summarize_multiseed(raw, keys)
    summary.to_csv(REPORT_DIR / "exp_multiseed.csv", index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))

    paired_df = paired_differences(raw, seeds, base="cb_cs", order=keys)
    if not paired_df.empty:
        paired_df.to_csv(REPORT_DIR / "exp_multiseed_paired.csv", index=False,
                         encoding="utf-8-sig")
        print("\n配对人价差（同一 seed 共享切分，差值 > 0 表示该配置比 CatBoost 贵）：")
        print(paired_df.to_string(index=False))

    ci_df = pd.DataFrame()
    paired_boot_df = pd.DataFrame()
    if not args.skip_bootstrap:
        main_out = outcomes.get(("cb_cs", MAIN_SEED))
        if main_out is not None:
            print(f"\n[E7] Bootstrap 置信区间（{args.n_boot} 次重采样，seed={MAIN_SEED}）")
            ci_df = bootstrap_ci(main_out.y_test, main_out.prob_test,
                                 main_out.threshold, n_boot=args.n_boot)
            ci_df.to_csv(REPORT_DIR / "exp_bootstrap.csv", index=False,
                         encoding="utf-8-sig")
            print(ci_df.to_string(index=False))

            records = [
                {"对比": f"阈值移动：{main_out.threshold:.2f} 对 0.50（同模型）",
                 **paired_bootstrap(main_out.y_test, main_out.prob_test, main_out.threshold,
                                    main_out.prob_test, 0.5, n_boot=args.n_boot)},
            ]
            for key in keys:
                other = outcomes.get((key, MAIN_SEED))
                if other is None or key == "cb_cs":
                    continue
                records.append(
                    {"对比": f"模型替换：CatBoost 代价敏感 对 {SPECS[key].label}",
                     **paired_bootstrap(main_out.y_test, main_out.prob_test, main_out.threshold,
                                        other.prob_test, other.threshold, n_boot=args.n_boot)}
                )
            paired_boot_df = pd.DataFrame(records)
            paired_boot_df.to_csv(REPORT_DIR / "exp_paired_bootstrap.csv", index=False,
                                  encoding="utf-8-sig")
            print("\n配对 Bootstrap（差值 > 0 表示前者更贵）：")
            print(paired_boot_df.to_string(index=False))

    cal_summary = pd.DataFrame()
    if not args.skip_calibration:
        print("\n[E8] 概率校准")
        cal_raw, curves = experiment_calibration(outcomes, seeds, "cb_cs")
        if not cal_raw.empty:
            cal_raw.to_csv(PRED_DIR / "calibration_raw.csv", index=False,
                           encoding="utf-8-sig")
            cal_summary = summarize_calibration(cal_raw)
            cal_summary.to_csv(REPORT_DIR / "exp_calibration.csv", index=False,
                               encoding="utf-8-sig")
            print(cal_summary.to_string(index=False))

            main_out = outcomes.get(("cb_cs", MAIN_SEED))
            if main_out is not None and MAIN_SEED in curves:
                fig_path = FIG_DIR / "fig5-4-calibration.png"
                ok = plot_reliability(main_out, curves[MAIN_SEED], fig_path)
                print(f"  可靠性图：{'已生成 ' + fig_path.name if ok else '跳过（matplotlib 不可用）'}")

    # ---- 汇总，便于直接贴进论文 ----
    parts = [
        "# 补充实验结果（多种子 / Bootstrap / 校准）",
        "",
        f"- 数据：`data/{DATA_DIR.name}`，主种子 {MAIN_SEED}，重复种子 {seeds}",
        f"- 超参：iterations={MODEL_PARAMS['iterations']}, "
        f"depth={MODEL_PARAMS['depth']}, lr={MODEL_PARAMS['learning_rate']}，5 折 OOF 选阈值",
        f"- 代价矩阵：C_FN={MATRIX.c_fn:.0f}，C_FP={MATRIX.c_fp:.0f}，"
        f"ρ={MATRIX.ratio:.0f}，Elkan 阈值={MATRIX.elkan_threshold:.4f}",
        "",
        to_markdown(f"E6 多种子稳定性（{len(seeds)} 次重复，均值 ± 标准差）", summary),
    ]
    if not paired_df.empty:
        parts += ["", to_markdown("E6b 逐种子配对差值（基准：CatBoost 代价敏感）", paired_df)]
    if not ci_df.empty:
        parts += ["", to_markdown("E7 测试集 Bootstrap 95% 置信区间", ci_df)]
    if not paired_boot_df.empty:
        parts += ["", to_markdown("E7b 配对 Bootstrap", paired_boot_df)]
    if not cal_summary.empty:
        parts += ["", to_markdown("E8 概率校准（阈值已在校准尺度上重选）", cal_summary)]
    (REPORT_DIR / "补充实验-多种子与校准.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("\n汇总 -> reports/补充实验-多种子与校准.md")
    return 0


if __name__ == "__main__":
    # 全局代价矩阵：与主实验同一口径，所有函数共用
    _settings = get_settings()
    MATRIX = CostMatrix(c_fn=_settings.cost_fn, c_fp=_settings.cost_fp)
    raise SystemExit(main())
