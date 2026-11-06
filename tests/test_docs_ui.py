"""接口文档页（/docs）的本地化与 MIME 修正。

这两件事都曾经出过线上问题，且都不在「预测链路」上，最容易被漏测：

1. FastAPI 默认的 `/docs` 从 `cdn.jsdelivr.net` 拉 swagger-ui 的 js/css。现场断网时
   页面能打开、但正文一片空白（脚本没加载上），看起来像服务坏了。
2. Windows 注册表可能把 `.css` 映射成 `application/x-css` 之类的怪值，浏览器做严格 MIME
   校验时会直接丢弃样式表，表现为「页面有内容但全是裸文字」。

因此这里断言两条硬约束：**页面不许出现任何外网地址**、**静态资源的 Content-Type 必须标准**。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mooc_dropout.api.app import STATIC_DIR, SWAGGER_VENDOR_DIR, create_app

VENDOR_JS = SWAGGER_VENDOR_DIR / "swagger-ui-bundle.js"
VENDOR_CSS = SWAGGER_VENDOR_DIR / "swagger-ui.css"

# 出现这些域名就说明页面在依赖外网：断网即白屏
EXTERNAL_HOSTS = ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com", "fonts.googleapis.com")


@pytest.fixture(scope="module")
def client(trained):
    settings, _ = trained
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_docs_page_served(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


@pytest.mark.parametrize("host", EXTERNAL_HOSTS)
def test_docs_page_has_no_external_dependency(client, host):
    body = client.get("/docs").text
    assert host not in body, f"/docs 仍在引用外网资源 {host}，断网会白屏"


@pytest.mark.skipif(not (VENDOR_JS.exists() and VENDOR_CSS.exists()), reason="仓库未内置 vendor 资源")
def test_docs_page_points_to_local_vendor(client):
    body = client.get("/docs").text
    assert "/static/vendor/swagger-ui-bundle.js" in body
    assert "/static/vendor/swagger-ui.css" in body


@pytest.mark.skipif(not VENDOR_CSS.exists(), reason="仓库未内置 vendor 资源")
def test_vendor_css_served_with_correct_mime(client):
    response = client.get("/static/vendor/swagger-ui.css")
    assert response.status_code == 200
    # 不带 charset 的 text/css 才是标准写法；application/x-css 会让浏览器丢弃样式表
    assert response.headers["content-type"].startswith("text/css"), response.headers["content-type"]
    assert len(response.content) > 1000


@pytest.mark.skipif(not VENDOR_JS.exists(), reason="仓库未内置 vendor 资源")
def test_vendor_js_served_with_correct_mime(client):
    response = client.get("/static/vendor/swagger-ui-bundle.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert len(response.content) > 100_000


def test_openapi_schema_lists_all_endpoints(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = set(response.json()["paths"])
    assert paths == {
        "/health",
        "/ready",
        "/v1/model/info",
        "/v1/model/features",
        "/v1/predict",
        "/v1/predict/batch",
        "/v1/intervention-list",
    }


def test_docs_and_index_are_not_in_schema(client):
    """两个页面都是 include_in_schema=False，不该混进接口契约里。"""
    paths = set(client.get("/openapi.json").json()["paths"])
    assert "/docs" not in paths
    assert "/" not in paths


def test_static_dir_exists():
    assert STATIC_DIR.is_dir(), "static 目录缺失，演示页与 /docs 都会 404"
    assert (STATIC_DIR / "index.html").is_file()
