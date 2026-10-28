"""模型元信息端点：供前端展示与论文附录（特征清单、阈值、代价矩阵）取数。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...features.registry import feature_dictionary
from ...schemas import ModelInfoResponse
from ...service.prediction_service import PredictionService
from ..dependencies import get_service

router = APIRouter(prefix="/v1/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse, summary="模型与阈值信息")
def model_info(service: PredictionService = Depends(get_service)) -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info())


@router.get("/features", summary="特征字典（分组 + 业务含义）")
def features() -> dict[str, Any]:
    items = feature_dictionary()
    return {"count": len(items), "items": items}
