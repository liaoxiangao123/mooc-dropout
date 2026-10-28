"""特征构建器：可 fit/transform、可持久化，保证训练与推理走同一条特征路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ..data.schema import validate_logs
from ..errors import ValidationFailure
from .behavior import summarize_events
from .registry import spec_names

CATEGORICAL_FEATURES: tuple[str, ...] = ("course_id", "username")

# behavior.summarize_events 产出的数值特征（顺序即列顺序）
NUMERIC_FEATURES: tuple[str, ...] = spec_names()[: -len(CATEGORICAL_FEATURES)]
ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class FeatureConfig:
    window_days: int = 14
    decay_lambda: float = 0.15
    session_gap_minutes: int = 30

    @classmethod
    def from_settings(cls, settings) -> "FeatureConfig":
        return cls(
            window_days=settings.feature_window_days,
            decay_lambda=settings.decay_lambda,
            session_gap_minutes=settings.session_gap_minutes,
        )


class FeatureBuilder:
    """训练与推理共用的特征构建器。

    fit 阶段校验数据契约并记录训练期元信息；transform 阶段只做确定性计算，
    不依赖任何全量统计量，因此单条样本与批量样本结果一致。
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.feature_names: list[str] = list(ALL_FEATURES)
        self.categorical_features: list[str] = list(CATEGORICAL_FEATURES)
        self.n_train_records_: int = 0
        self.is_fitted_: bool = False

    # ---- 训练侧 ----
    def fit(self, logs: pd.DataFrame) -> "FeatureBuilder":
        validated = validate_logs(logs)
        if validated.empty:
            raise ValidationFailure("训练日志为空")
        # 本构建器不依赖任何全量统计量，因此 1 条选课也可 fit；
        # 生产训练时由 Trainer 负责校验样本量是否足够。
        self.n_train_records_ = int(validated["enrollment_id"].nunique())
        self.is_fitted_ = True
        return self

    def transform(
        self,
        logs: pd.DataFrame,
        enrollments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValidationFailure("FeatureBuilder 尚未 fit，禁止直接 transform")

        validated = validate_logs(logs)
        features = summarize_events(
            validated,
            window_days=self.config.window_days,
            decay_lambda=self.config.decay_lambda,
            session_gap_minutes=self.config.session_gap_minutes,
        )

        # 类别特征优先取选课表，其次取日志自带列，最后填 unknown
        if enrollments is not None and not enrollments.empty:
            meta = (
                enrollments.drop_duplicates("enrollment_id")
                .set_index("enrollment_id")[list(CATEGORICAL_FEATURES)]
            )
            for column in CATEGORICAL_FEATURES:
                features[column] = meta[column].reindex(features.index)
        else:
            for column in CATEGORICAL_FEATURES:
                if column in validated.columns:
                    mapping = validated.drop_duplicates("enrollment_id").set_index(
                        "enrollment_id"
                    )[column]
                    features[column] = mapping.reindex(features.index)
                else:
                    features[column] = "unknown"

        for column in CATEGORICAL_FEATURES:
            features[column] = features[column].fillna("unknown").astype(str)

        features = features.reindex(columns=self.feature_names)
        # 逐列转换：兼容 pandas 3.x 对「列表键 -> 多列赋值」的行为变化
        for column in NUMERIC_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
        return features

    def fit_transform(
        self,
        logs: pd.DataFrame,
        enrollments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.fit(logs).transform(logs, enrollments)

    # ---- 持久化 ----
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "FeatureBuilder":
        return joblib.load(Path(path))
