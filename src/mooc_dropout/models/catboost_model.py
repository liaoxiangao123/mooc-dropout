"""代价敏感 CatBoost：本课题的核心模型。

两条代价敏感路径（论文第 4 章的实体）：
    1. 类别加权（weighting="class_weight"）：按 C_FN / C_FP 给正类加权。
       等价于把决策阈值从 0.5 移到 C_FP/(C_FP+C_FN)，因此加权与阈值调整在理论上同源，
       这也是论文里必须做消融实验说明二者差异的原因。
    2. 无加权（weighting="none"）：作为对照组，用于证明收益来自代价机制而非模型本身。

CatBoost 的原生类别特征处理（Ordered Target Statistics）使我们无需对
course_id / username 这类高基数特征做 One-Hot，避免了维度爆炸与目标泄漏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ModelNotReadyError
from ..logging_config import get_logger
from .base import BaseRiskModel

logger = get_logger(__name__)

try:  # pragma: no cover - 依赖缺失时走降级分支
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment]
    CATBOOST_AVAILABLE = False


class CatBoostRiskModel(BaseRiskModel):
    name = "catboost"
    supports_categorical = True

    def __init__(
        self,
        *,
        iterations: int = 600,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        cost_ratio: float = 10.0,
        weighting: str = "class_weight",
        use_cs_loss: bool = False,
        random_seed: int = 42,
        thread_count: int = -1,
        verbose: bool = False,
        categorical_features: list[str] | None = None,
        **extra_params,
    ) -> None:
        if weighting not in {"class_weight", "none"}:
            raise ValueError("weighting 仅支持 'class_weight' 或 'none'")
        if cost_ratio <= 0:
            raise ValueError("cost_ratio 必须为正数")

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.cost_ratio = float(cost_ratio)
        self.weighting = weighting
        self.use_cs_loss = bool(use_cs_loss)
        if self.use_cs_loss and weighting == "class_weight":
            # 自定义损失已包含代价系数，再叠加类别权重会造成代价重复计入
            raise ValueError("use_cs_loss=True 时必须将 weighting 设为 'none'")
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose
        self._categorical_features = list(categorical_features or [])
        self._cat_features: list[str] = []
        self.extra_params = extra_params
        self.model_: "CatBoostClassifier | None" = None

    @property
    def categorical_features(self) -> list[str]:
        return self._cat_features or self._categorical_features

    @staticmethod
    def _resolve_cat_features(X: pd.DataFrame, configured: list[str]) -> list[str]:
        """显式配置优先；未配置时自动识别非数值列，避免字符串直接进入数值特征。"""
        cols = [c for c in configured if c in X.columns]
        if not cols:
            cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return cols

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CatBoostRiskModel":
        if not CATBOOST_AVAILABLE:
            raise ModelNotReadyError(
                "未安装 catboost，无法训练核心模型；请执行 pip install catboost"
            )
        self._check_input(X)
        y = np.asarray(y).astype(int)
        if len(X) != len(y):
            raise ValueError("X 与 y 样本数不一致")
        if len(np.unique(y)) < 2:
            raise ValueError("训练集仅包含单一类别，无法训练分类模型")

        cat_features = self._resolve_cat_features(X, self._categorical_features)
        self._cat_features = cat_features
        for column in cat_features:
            X = X.assign(**{column: X[column].astype(str)})

        class_weights = None
        loss_function = "Logloss"
        if self.use_cs_loss:
            # 路线 C：代价敏感自定义目标（此时不再叠加类别权重）
            from .cost_sensitive_loss import CostSensitiveLogloss

            loss_function = CostSensitiveLogloss(cost_fn=self.cost_ratio, cost_fp=1.0)
        elif self.weighting == "class_weight":
            # 正类权重的理论依据：加权后 0.5 阈值等价于 C_FP/(C_FP+C_FN)
            class_weights = [1.0, self.cost_ratio]

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=loss_function,
            eval_metric="Logloss",
            class_weights=class_weights,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            verbose=self.verbose,
            allow_writing_files=False,
            **self.extra_params,
        )
        self.model_.fit(X, y, cat_features=cat_features)
        logger.info(
            "CatBoost 训练完成",
            extra={
                "model_name": self.name,
                "event": "model.trained",
                "weighting": self.weighting,
                "cost_ratio": self.cost_ratio,
                "use_cs_loss": self.use_cs_loss,
            },
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ModelNotReadyError("模型尚未训练")
        for column in self.categorical_features:
            if column in X.columns:
                X = X.assign(**{column: X[column].astype(str)})
        proba = self.model_.predict_proba(X)
        return np.asarray(proba)[:, 1].astype("float64")

    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            raise ModelNotReadyError("模型尚未训练")
        names = list(self.model_.feature_names_)
        values = np.asarray(self.model_.get_feature_importance())
        return pd.Series(values, index=names, name="importance").sort_values(ascending=False)

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "iterations": self.iterations,
                "depth": self.depth,
                "learning_rate": self.learning_rate,
                "weighting": self.weighting,
                "cost_ratio": self.cost_ratio,
                "use_cs_loss": self.use_cs_loss,
            }
        )
        return info
