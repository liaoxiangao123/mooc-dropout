"""仓储层：模型产物的唯一读写入口。

服务层不直接碰文件系统，便于测试替换为内存实现，也便于将来换成对象存储。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ModelNotReadyError
from ..explain.explainer import RiskExplainer
from ..features.builder import FeatureBuilder
from ..logging_config import get_logger
from ..models.base import BaseRiskModel
from ..models.calibration import ProbabilityCalibrator

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"
BUILDER_FILE = "feature_builder.joblib"
METADATA_FILE = "metadata.json"
# 可选产物：存在即表示服务应输出校准后概率
CALIBRATOR_FILE = "calibrator.joblib"
REQUIRED_FILES = (MODEL_FILE, BUILDER_FILE, METADATA_FILE)


@dataclass
class ModelArtifacts:
    model: BaseRiskModel
    builder: FeatureBuilder
    metadata: dict[str, Any]
    explainer: RiskExplainer
    calibrator: ProbabilityCalibrator | None = None

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", self.model.name))

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator is not None and self.calibrator.method != "none"


class ArtifactRepository:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self._cache: ModelArtifacts | None = None

    @property
    def is_available(self) -> bool:
        return all((self.model_dir / name).exists() for name in REQUIRED_FILES)

    def describe_files(self) -> dict[str, bool]:
        return {
            "model": (self.model_dir / MODEL_FILE).exists(),
            "feature_builder": (self.model_dir / BUILDER_FILE).exists(),
            "metadata": (self.model_dir / METADATA_FILE).exists(),
            "calibrator": (self.model_dir / CALIBRATOR_FILE).exists(),
        }

    def load(self, *, force_reload: bool = False) -> ModelArtifacts:
        if self._cache is not None and not force_reload:
            return self._cache

        files = self.describe_files()
        missing = [name for name in ("model", "feature_builder", "metadata") if not files[name]]
        if missing:
            raise ModelNotReadyError(
                "模型产物不完整，请先运行 `python scripts/train.py`",
                details={"missing": missing, "model_dir": str(self.model_dir)},
            )

        model = BaseRiskModel.load(self.model_dir / MODEL_FILE)
        builder = FeatureBuilder.load(self.model_dir / BUILDER_FILE)
        metadata = json.loads((self.model_dir / METADATA_FILE).read_text(encoding="utf-8"))
        explainer = RiskExplainer(
            model, feature_names=builder.feature_names, categorical_features=builder.categorical_features
        )
        calibrator = (
            ProbabilityCalibrator.load(self.model_dir / CALIBRATOR_FILE)
            if files["calibrator"]
            else None
        )
        self._cache = ModelArtifacts(
            model=model,
            builder=builder,
            metadata=metadata,
            explainer=explainer,
            calibrator=calibrator,
        )
        logger.info(
            "模型产物加载完成",
            extra={
                "event": "artifact.loaded",
                "model_name": str(metadata.get("model_name")),
                "calibrated": self._cache.is_calibrated,
            },
        )
        return self._cache

    def invalidate(self) -> None:
        self._cache = None
