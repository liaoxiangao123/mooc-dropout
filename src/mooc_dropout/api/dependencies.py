"""FastAPI 依赖注入：控制器只通过依赖拿到服务，不自行构造。"""

from __future__ import annotations

from fastapi import Request

from ..config import Settings
from ..errors import ModelNotReadyError
from ..service.prediction_service import PredictionService


def get_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ModelNotReadyError("推理服务未初始化")
    return service


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
