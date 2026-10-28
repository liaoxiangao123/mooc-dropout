"""sklearn 基线模型：逻辑回归 / 随机森林 / 直方图 GBDT。

用途：
    * 作为对照组，验证 CatBoost 是否真的更优；
    * 在未安装 catboost 的环境中作为可运行的降级实现（保证服务始终可用）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .base import BaseRiskModel

_SUPPORTED = ("logreg", "random_forest", "hist_gbdt")


class SklearnRiskModel(BaseRiskModel):
    name = "sklearn"

    def __init__(
        self,
        kind: str = "hist_gbdt",
        *,
        cost_ratio: float = 1.0,
        random_seed: int = 42,
        categorical_features: list[str] | None = None,
        **params,
    ) -> None:
        if kind not in _SUPPORTED:
            raise ValueError(f"不支持的 kind={kind}，可选 {_SUPPORTED}")
        self.kind = kind
        self.name = kind
        self.cost_ratio = float(cost_ratio)
        self.random_seed = random_seed
        self._categorical_features = list(categorical_features or [])
        self.params = params
        self.pipeline_: Pipeline | None = None
        self._cat_cols: list[str] = []

    @property
    def categorical_features(self) -> list[str]:
        return self._cat_cols or self._categorical_features

    def _resolve_cat_cols(self, X: pd.DataFrame) -> list[str]:
        """显式指定优先；未指定时按 dtype 自动识别字符串列，避免把文本喂给数值模型。"""
        cols = [c for c in self._categorical_features if c in X.columns]
        if not cols:
            cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return cols

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        cat_cols = self._cat_cols
        num_cols = [c for c in X.columns if c not in cat_cols]
        params = dict(self.params)

        numeric_steps: list = [("passthrough", "passthrough")]
        if self.kind == "logreg":
            numeric_steps = [("scale", StandardScaler())]

        transformers = [("num", Pipeline(numeric_steps), num_cols)]
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    cat_cols,
                )
            )
        preprocessor = ColumnTransformer(transformers, remainder="drop")

        if self.kind == "logreg":
            estimator = LogisticRegression(
                max_iter=2000,
                random_state=self.random_seed,
                **params,
            )
        elif self.kind == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=params.pop("n_estimators", 300),
                max_depth=params.pop("max_depth", None),
                min_samples_leaf=params.pop("min_samples_leaf", 2),
                n_jobs=-1,
                random_state=self.random_seed,
                **params,
            )
        else:
            estimator = HistGradientBoostingClassifier(
                max_iter=params.pop("max_iter", 300),
                learning_rate=params.pop("learning_rate", 0.05),
                max_depth=params.pop("max_depth", None),
                random_state=self.random_seed,
                **params,
            )
        return Pipeline([("prep", preprocessor), ("model", estimator)])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SklearnRiskModel":
        self._check_input(X)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("训练集仅包含单一类别，无法训练分类模型")

        X = X.copy()
        self._cat_cols = self._resolve_cat_cols(X)
        for column in self._cat_cols:
            X[column] = X[column].astype(str)

        self.pipeline_ = self._build_pipeline(X)
        # 用样本权重表达代价敏感：正类权重 = C_FN / C_FP
        sample_weight = np.where(y == 1, self.cost_ratio, 1.0)
        self.pipeline_.fit(X, y, model__sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("模型尚未训练")
        X = X.copy()
        for column in self._cat_cols:
            if column in X.columns:
                X[column] = X[column].astype(str)
        return np.asarray(self.pipeline_.predict_proba(X))[:, 1].astype("float64")

    def feature_importance(self) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("模型尚未训练")
        estimator = self.pipeline_.named_steps["model"]
        names = list(self.pipeline_.named_steps["prep"].get_feature_names_out())

        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype="float64")
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype="float64")).ravel()
        else:
            # HistGradientBoosting 没有原生特征重要性；返回空 Series，
            # 由调用方决定是否改用置换重要性（需要 y，训练期不在模型内保存）
            return pd.Series(dtype="float64", name="importance")

        # 名称形如 "num__total_events"，去掉前缀便于阅读
        clean = [n.split("__", 1)[-1] for n in names]
        return pd.Series(values, index=clean, name="importance").sort_values(ascending=False)
