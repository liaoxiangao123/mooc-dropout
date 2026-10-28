"""应用工厂：装配中间件、路由、静态演示页与生命周期钩子。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, get_settings
from ..logging_config import configure_logging, get_logger
from ..repository.artifact_repository import ArtifactRepository
from ..service.prediction_service import PredictionService
from .errors_handler import register_exception_handlers
from .middleware import RequestContextMiddleware
from .routers import health, model, predict

logger = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, service: PredictionService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    service = service or PredictionService(ArtifactRepository(settings.model_dir))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service.is_ready():
            logger.info("模型就绪", extra={"event": "app.model_ready"})
        else:
            logger.warning(
                "模型产物缺失，/v1/predict 将返回 503；请先运行 scripts/train.py",
                extra={"event": "app.model_missing"},
            )
        yield
        logger.info("服务已优雅停机", extra={"event": "app.shutdown"})

    app = FastAPI(
        title="MOOC 辍学风险预测服务",
        description="基于代价敏感 CatBoost 的 MOOC 辍学风险预测毕业设计原型系统",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    # 最后添加 => 最外层，保证 CORS 预检也能带上 request_id
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(model.router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def demo_index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
