"""可解释性：优先用 CatBoost 原生 SHAP，缺失时降级为全局特征重要性。

为什么不用 shap 包：CatBoost 自带 SHAP 计算且支持类别特征，
可以省掉一个大依赖，也避免类别特征在 TreeExplainer 中的编码问题。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..logging_config import get_logger

logger = get_logger(__name__)


class RiskExplainer:
    def __init__(self, model, feature_names: list[str], categorical_features: list[str] | None = None):
        self.model = model
        self.feature_names = list(feature_names)
        self.categorical_features = list(categorical_features or [])
        self._global_importance = self._safe_global_importance()

    def _safe_global_importance(self) -> dict[str, float]:
        try:
            importance = self.model.feature_importance()
            return {str(k): float(v) for k, v in importance.items()}
        except Exception:  # pragma: no cover - 模型无重要性时降级
            logger.warning("无法获取全局特征重要性", extra={"event": "explain.no_importance"})
            return {}

    def global_importance(self, top_k: int = 10) -> list[dict[str, Any]]:
        ranked = sorted(self._global_importance.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"feature": name, "importance": round(value, 4)} for name, value in ranked[:top_k]
        ]

    def explain(self, features: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
        """返回单条样本的风险因子列表（按 |贡献| 降序）。"""
        if features.empty:
            return []

        contributions = self._shap_contributions(features)
        if contributions is None:
            return [
                {
                    "feature": name,
                    "value": _stringify(features.iloc[0].get(name)),
                    "contribution": None,
                    "direction": "unknown",
                    "source": "global_importance",
                }
                for name in list(self._global_importance)[:top_k]
            ]

        row = features.iloc[0]
        ranked = sorted(
            zip(self.feature_names, contributions),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        factors = []
        for name, value in ranked[:top_k]:
            factors.append(
                {
                    "feature": name,
                    "value": _stringify(row.get(name)),
                    "contribution": round(float(value), 4),
                    "direction": "risk_up" if value > 0 else "risk_down",
                    "source": "shap",
                }
            )
        return factors

    def _shap_contributions(self, features: pd.DataFrame) -> np.ndarray | None:
        inner = getattr(self.model, "model_", None)
        if inner is None or not hasattr(inner, "get_feature_importance"):
            return None
        try:  # pragma: no cover - 依赖 catboost 原生接口
            from catboost import Pool

            cat_features = [c for c in self.categorical_features if c in features.columns]
            data = features.copy()
            for column in cat_features:
                data[column] = data[column].astype(str)
            pool = Pool(data, cat_features=cat_features)
            values = np.asarray(inner.get_feature_importance(pool, type="ShapValues"))
            return values[0, :-1]  # 最后一列为期望值，剔除
        except Exception:  # pragma: no cover
            logger.warning("SHAP 计算失败，降级为全局重要性", extra={"event": "explain.shap_failed"})
            return None


def _stringify(value: Any) -> str | float:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(round(float(value), 4))
    return str(value)
