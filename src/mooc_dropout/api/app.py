"""应用工厂：装配中间件、路由、静态演示页与生命周期钩子。"""

from __future__ import annotations

import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, get_settings
from ..logging_config import configure_logging, get_logger
from ..repository.artifact_repository import ArtifactRepository
from ..service.prediction_service import PredictionService
from .errors_handler import register_exception_handlers
from .middleware import RequestContextMiddleware
from .routers import health, model, predict

# Windows 注册表可能把 .css 映射成 application/x-css 一类的怪值，浏览器会因 MIME
# 不匹配而直接丢弃样式表（表现为 /docs 有内容却没样式）。这里显式纠正一次。
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/html", ".html")

logger = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
# Swagger UI 的 js/css 随包内置，避免 /docs 依赖外网 CDN（离线/答辩现场可直接打开）
SWAGGER_VENDOR_DIR = STATIC_DIR / "vendor"
# 1x1 透明 PNG，省掉对 fastapi 官网 favicon 的请求
BLANK_FAVICON = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


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
        docs_url=None,
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

        # FastAPI 默认的 /docs 从 jsdelivr 拉 swagger-ui，现场断网就是一片空白；
        # 仓库带了 vendor 副本时改用本地路径，接口文档从此不依赖外网。
        has_local_swagger = (SWAGGER_VENDOR_DIR / "swagger-ui-bundle.js").exists() and (
            SWAGGER_VENDOR_DIR / "swagger-ui.css"
        ).exists()

        @app.get("/docs", include_in_schema=False)
        def swagger_ui_docs() -> HTMLResponse:
            # 显式标注 dict[str, Any]：形参类型是 dict[str, Any] | None，而 dict 在类型上是不变的，
            # 写成 dict[str, str] 会与形参不兼容（mypy arg-type）
            local: dict[str, Any] = (
                {
                    "swagger_js_url": "/static/vendor/swagger-ui-bundle.js",
                    "swagger_css_url": "/static/vendor/swagger-ui.css",
                    "swagger_favicon_url": BLANK_FAVICON,
                }
                if has_local_swagger
                else {}
            )
            return get_swagger_ui_html(
                openapi_url=app.openapi_url or "/openapi.json",
                title=app.title + " - Swagger UI",
                **local,
            )

        @app.get("/", include_in_schema=False)
        def demo_index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
