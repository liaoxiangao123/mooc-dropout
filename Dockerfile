# 演示镜像：一条 docker run 就能看到页面，不依赖宿主机的 Python 环境。
#
#   docker build -t mooc-dropout .
#   docker run --rm -p 8000:8000 mooc-dropout
#   # 浏览器打开 http://127.0.0.1:8000
#
# 镜像内置的是**合成数据模型**（4 000 条选课，训练约 1 分钟），只为把界面跑起来；
# 论文里的指标来自真实数据模型，那需要 1.2 GB 的 KDD Cup 2015 数据，不适合塞进镜像。
# 要换成真实数据模型，把产物目录挂进来即可：
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts/realdata_official:/app/artifacts" mooc-dropout

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1：CatBoost 的 OpenMP 运行时，缺了会在 import 阶段报 libgomp.so.1 not found
# curl：仅用于 HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖层单独缓存：只改代码时不必重装 400 MB 的依赖
COPY requirements.lock.txt ./
RUN pip install -r requirements.lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

# 构建期就把演示模型训好，容器起来即可用（避免「跑起来但 /ready 是 503」）
RUN python scripts/train.py --auto-synthetic --iterations 250 --n-splits 5 \
    && test -f artifacts/model/metadata.json

ENV MOOC_API_HOST=0.0.0.0 \
    MOOC_API_PORT=8000 \
    MOOC_ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
