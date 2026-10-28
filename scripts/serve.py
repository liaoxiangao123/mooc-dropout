"""启动推理服务。

用法：
    python scripts/serve.py            # 默认 127.0.0.1:8000
    python scripts/serve.py --reload   # 开发热重载
    python scripts/serve.py --artifacts-dir artifacts/realdata_official
                                       # 换一套模型产物（答辩演示用真实数据模型）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from mooc_dropout.api.app import create_app  # noqa: E402
from mooc_dropout.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="启动 MOOC 辍学风险预测服务")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="模型产物根目录，覆盖 MOOC_ARTIFACTS_DIR；相对路径按项目根目录解析",
    )
    args = parser.parse_args()

    if args.artifacts_dir:
        target = args.artifacts_dir if args.artifacts_dir.is_absolute() else ROOT / args.artifacts_dir
        if not (target / "model" / "metadata.json").exists():
            print(f"未找到模型产物：{target / 'model' / 'metadata.json'}")
            return 2
        # --reload 会用导入字符串另起子进程，必须靠环境变量传递，否则覆盖会丢
        os.environ["MOOC_ARTIFACTS_DIR"] = str(target)
        settings = settings.model_copy(update={"artifacts_dir": target})
        print(f"模型产物: {target}")

    if args.reload:
        # 热重载模式需要以导入字符串启动
        sys.path.insert(0, str(ROOT / "src"))
        uvicorn.run(
            "mooc_dropout.api.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(ROOT / "src")],
        )
    else:
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
