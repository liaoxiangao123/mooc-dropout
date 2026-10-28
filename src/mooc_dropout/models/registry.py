"""模型工厂：按名称构建模型，统一注入代价比与随机种子。"""

from __future__ import annotations

from .base import BaseRiskModel
from .catboost_model import CATBOOST_AVAILABLE, CatBoostRiskModel
from .baseline import SklearnRiskModel

MODEL_NAMES: tuple[str, ...] = ("catboost", "logreg", "random_forest", "hist_gbdt")


def is_available(name: str) -> bool:
    if name == "catboost":
        return CATBOOST_AVAILABLE
    return name in MODEL_NAMES


def build_model(
    name: str,
    *,
    cost_ratio: float = 10.0,
    weighting: str = "class_weight",
    random_seed: int = 42,
    categorical_features: list[str] | None = None,
    **params,
) -> BaseRiskModel:
    """构建模型实例。

    cost_ratio 只对使用代价敏感的模型生效；基线的默认值为 1.0（代价中立）。
    """
    if name == "catboost":
        return CatBoostRiskModel(
            cost_ratio=cost_ratio,
            weighting=weighting,
            random_seed=random_seed,
            categorical_features=categorical_features,
            **params,
        )
    if name in ("logreg", "random_forest", "hist_gbdt"):
        return SklearnRiskModel(
            kind=name,
            cost_ratio=cost_ratio,
            random_seed=random_seed,
            categorical_features=categorical_features,
            **params,
        )
    raise ValueError(f"未知模型 {name}，可选 {MODEL_NAMES}")
