"""健康检查：/health 为存活探针，/ready 为就绪探针（模型是否可用）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from ...schemas import HealthResponse, ReadyResponse
from ..dependencies import get_app_settings, get_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="存活探针")
def health(settings=Depends(get_app_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version, env=settings.env)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="就绪探针：模型产物是否已加载",
    responses={503: {"description": "模型不可用"}},
)
def ready(
    response: Response,
    service=Depends(get_service),
    settings=Depends(get_app_settings),
) -> ReadyResponse:
    available = service.is_ready()
    if not available:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ready" if available else "not_ready",
        model_ready=available,
        detail={"model_dir": str(settings.model_dir)},
    )
