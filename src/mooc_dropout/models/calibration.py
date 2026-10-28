"""概率校准：把「代价敏感加权后的分数」还原成可被教师按字面理解的后验概率。

为什么必须做（论文 5.7.2 / E8 的工程落地）：
    类别加权会让模型的输出概率系统性偏高。加权后 p=0.3 并不代表「三成概率辍学」，
    只代表「排序上比另一部分人更危险」。教师若按字面理解，会把预警规模放大数倍。
    因此线上服务要么输出「风险分数」并明确声明未校准，要么做校准后输出概率。
    本项目选择后者。

为什么校准后必须重选阈值：
    校准是单调变换（保序回归单调不减、Platt 单调递增），AUC 不变，但阈值所在的
    数值尺度变了。沿用校准前的阈值等于用旧尺子量新布，代价会显著上升（E8 中 +69%）。
    所以本模块只负责「变换概率」，阈值重选由 Trainer 在同一尺度上完成。

方法选择：
    * platt（Platt Scaling）：对 logit 做单变量逻辑回归，参数少、样本少时更稳，
      但只能修正单调的尺度偏移，纠正不了 S 形畸变；
    * isotonic（保序回归）：非参数、单调，能纠正任意单调畸变，样本少时易过拟合，
      故只在 OOF 预测（而非测试集）上拟合。

工程约束：
    校准器只在训练集 OOF 预测上拟合，测试集全程不参与 —— 用测试集调校准参数
    是最常见的隐性数据穿越，会让论文里的校准指标虚高。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ModelNotReadyError

CALIBRATION_METHODS = ("none", "platt", "isotonic")

_LABELS = {
    "none": "未校准",
    "platt": "Platt（Sigmoid）",
    "isotonic": "保序回归（Isotonic）",
}


def _logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype="float64"), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class ProbabilityCalibrator:
    """可序列化的概率校准器；method="none" 时退化为恒等变换。"""

    method: str = "none"
    _fitted: bool = False
    _estimator: Any = field(default=None, repr=False)
    n_fit_samples: int = 0
    fitted_positive_rate: float | None = None
    input_scale: str = "probability"  # platt 内部用 logit，对外统一收概率

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS:
            raise ValueError(f"calibration 仅支持 {list(CALIBRATION_METHODS)}")

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.method == "none" or self._fitted

    @property
    def label(self) -> str:
        return _LABELS[self.method]

    def fit(self, oof_prob, y_true) -> "ProbabilityCalibrator":
        """在训练集 OOF 预测上拟合。测试集不得参与。"""
        prob = np.asarray(oof_prob, dtype="float64")
        y = np.asarray(y_true).astype(int)
        if prob.shape != y.shape:
            raise ValueError("oof_prob 与 y_true 长度不一致")
        if prob.size < 10:
            raise ValueError("样本过少（<10），校准不可靠")
        if np.unique(y).size < 2:
            raise ValueError("y_true 仅含单一类别，无法拟合校准器")

        self.n_fit_samples = int(prob.size)
        self.fitted_positive_rate = float(y.mean())

        if self.method == "none":
            self._fitted = True
            return self

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(solver="lbfgs", max_iter=1000)
            estimator.fit(_logit(prob).reshape(-1, 1), y)
        else:
            from sklearn.isotonic import IsotonicRegression

            estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            estimator.fit(prob, y)

        self._estimator = estimator
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, prob) -> np.ndarray:
        """把原始概率映射到校准后的概率尺度。"""
        values = np.asarray(prob, dtype="float64")
        if self.method == "none":
            return np.clip(values, 0.0, 1.0)
        if not self._fitted or self._estimator is None:
            raise ModelNotReadyError(
                "校准器尚未拟合，请先调用 fit() 或在产物中加载",
                details={"method": self.method},
            )
        if self.method == "platt":
            out = self._estimator.predict_proba(_logit(values).reshape(-1, 1))[:, 1]
        else:
            out = self._estimator.predict(values)
        return np.clip(np.asarray(out, dtype="float64"), 0.0, 1.0)

    def transform_one(self, prob: float) -> float:
        return float(self.transform(np.array([prob], dtype="float64"))[0])

    # ------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "fitted": bool(self._fitted),
            "n_fit_samples": int(self.n_fit_samples),
            "fitted_positive_rate": (
                round(self.fitted_positive_rate, 4)
                if self.fitted_positive_rate is not None
                else None
            ),
            "note": "校准器在训练集 OOF 预测上拟合；阈值已在同一尺度上重选",
        }

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        import joblib

        return joblib.load(Path(path))
