"""集中配置：全部来自环境变量，启动时集中校验并快速失败。

原则：
1. 代码中不出现任何硬编码的路径、端口、密钥、阈值；
2. 所有配置项在进程启动时一次性校验，非法配置直接抛错，不允许带病启动；
3. 生产环境额外做安全校验（例如 CORS 不允许通配符）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/mooc_dropout/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    """进程级配置。字段名即环境变量名（前缀 MOOC_）。"""

    model_config = SettingsConfigDict(
        env_prefix="MOOC_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 运行环境 ----
    env: Literal["dev", "test", "prod"] = "dev"
    app_name: str = "mooc-dropout-risk"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ---- HTTP 服务 ----
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    # ---- 目录 ----
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")

    # ---- 代价矩阵（元）----
    cost_fn: float = Field(default=1000.0, gt=0, description="漏报代价：辍学但未预警")
    cost_fp: float = Field(default=100.0, gt=0, description="误报代价：预警但未辍学")

    # ---- 特征工程 ----
    feature_window_days: int = Field(default=14, ge=1, le=60)
    decay_lambda: float = Field(default=0.15, gt=0, le=1.0)
    session_gap_minutes: int = Field(default=30, ge=1, le=240)

    # ---- 模型 ----
    random_seed: int = 42

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in _VALID_LOG_LEVELS:
            raise ValueError(f"非法日志级别 {value}，可选：{sorted(_VALID_LOG_LEVELS)}")
        return value

    @field_validator("cors_origins")
    @classmethod
    def _check_cors(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("cors_origins 不能为空")
        return value

    @model_validator(mode="after")
    def _check_env_specific(self) -> "Settings":
        if self.env == "prod" and "*" in self.cors_origin_list:
            raise ValueError("生产环境禁止使用通配符 CORS 来源")
        if self.cost_fp > self.cost_fn:
            # 不禁止，但必须在文档与实验里说明：误报比漏报更贵的场景同样成立
            pass
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    # ---- 路径解析：相对路径一律相对项目根，避免工作目录漂移 ----
    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def resolved_data_dir(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def resolved_artifacts_dir(self) -> Path:
        return self._resolve(self.artifacts_dir)

    @property
    def resolved_reports_dir(self) -> Path:
        return self._resolve(self.reports_dir)

    @property
    def model_dir(self) -> Path:
        """模型产物目录（模型、特征构建器、元数据）。"""
        return self.resolved_artifacts_dir / "model"

    def ensure_dirs(self) -> None:
        for directory in (
            self.resolved_data_dir,
            self.resolved_reports_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试中可用 get_settings.cache_clear() 重置。"""
    return Settings()
