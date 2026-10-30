"""一键启动演示：自动挑端口 -> 起服务 -> 健康检查通过后自动打开浏览器。

为什么不用 `serve.py`：
    `serve.py` 是给开发用的，端口被占会直接失败、也不会开浏览器。答辩现场需要的是
    「一条命令、立刻能看到页面」，这个脚本就是那一条命令。

用法：
    python scripts/demo.py
    python scripts/demo.py --artifacts-dir artifacts/realdata_official   # 换真实数据模型
    python scripts/demo.py --port 8080
    python scripts/demo.py --no-browser                                   # 只起服务
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from mooc_dropout.api.app import create_app  # noqa: E402
from mooc_dropout.config import get_settings  # noqa: E402

BANNER = "=" * 64


def find_free_port(start: int, tries: int = 20) -> int:
    """从 start 起找一个能 bind 的端口；现场最怕的就是 8000 被别的东西占着。"""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"{start}~{start + tries - 1} 之间没有空闲端口，请用 --port 指定")


def health_ok(base_url: str) -> bool:
    """探测 /health。

    必须显式清空 ProxyHandler：Windows 上 http_proxy 会让 127.0.0.1 也走代理，
    表现就是请求被怼回 "upstream connect failed"，看起来像服务没起来。
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base_url.rstrip("/") + "/health", timeout=1.0) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def open_when_ready(base_url: str, timeout: float = 30.0) -> None:
    """后台线程：等服务真能响应了再开浏览器，避免打开一个「无法访问」的白页。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health_ok(base_url):
            print(f"\n[演示] 服务已就绪，正在打开浏览器：{base_url}\n", flush=True)
            webbrowser.open(base_url)
            return
        time.sleep(0.3)
    print(f"\n[演示] {timeout:.0f}s 内未通过健康检查，请手动打开 {base_url}\n", flush=True)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="一键启动 MOOC 辍学风险预测演示")
    parser.add_argument("--port", type=int, default=settings.api_port, help="起始端口，被占用时自动顺延")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="模型产物根目录；演示真实数据模型可传 artifacts/realdata_official",
    )
    parser.add_argument("--no-browser", action="store_true", help="只起服务，不自动打开浏览器")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    if args.artifacts_dir:
        target = args.artifacts_dir if args.artifacts_dir.is_absolute() else ROOT / args.artifacts_dir
        metadata = target / "model" / "metadata.json"
        if not metadata.exists():
            print(f"未找到模型产物：{metadata}")
            print("先跑一次 python scripts/train.py --auto-synthetic 生成默认产物")
            return 2
        os.environ["MOOC_ARTIFACTS_DIR"] = str(target)
        settings = settings.model_copy(update={"artifacts_dir": target})
    else:
        target = settings.resolved_artifacts_dir

    port = find_free_port(args.port)
    base_url = f"http://{args.host}:{port}/"

    print(BANNER)
    print("  MOOC 辍学风险预测 · 原型演示")
    print(BANNER)
    print(f"  演示页    {base_url}")
    print(f"  接口文档  {base_url}docs")
    print(f"  模型产物  {target}")
    if port != args.port:
        print(f"  端口 {args.port} 被占用，已自动改用 {port}")
    if not (target / "model" / "metadata.json").exists():
        print("  ⚠ 模型产物缺失：/v1/predict 会返回 503，先跑 scripts/train.py --auto-synthetic")
    print(BANNER)
    print("  页面打不开时：把 127.0.0.1 加进浏览器代理例外，或临时关闭系统代理")
    print("  结束后按 Ctrl+C")
    print()

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(base_url,), daemon=True).start()

    uvicorn.run(create_app(settings), host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
