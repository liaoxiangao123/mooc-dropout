"""模型抽象：统一 fit / predict_proba / 特征重要性 / 持久化 契约。

上层（训练、服务）只依赖该抽象，因此可无缝替换 CatBoost 与 sklearn 基线，
也便于单元测试用轻量替身。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseRiskModel(ABC):
    name: str = "base"
    supports_categorical: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseRiskModel":
        """训练；X 的列顺序必须与训练时一致。"""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回正类（辍学）概率的一维数组，取值 [0, 1]。"""

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """返回按重要性降序排列的 Series（index 为特征名）。"""

    @property
    def categorical_features(self) -> list[str]:
        """需要按类别特征处理的列名。"""
        return []

    # ---- 公共工具 ----
    @staticmethod
    def _check_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("模型输入必须是 pandas.DataFrame")
        if X.empty:
            raise ValueError("模型输入为空")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseRiskModel":
        return joblib.load(Path(path))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": self.__class__.__name__}
