"""HTTP 层数据契约（pydantic v2）。

原则：请求体严格校验（extra=forbid），响应结构固定，
错误信息只描述「哪里不合法」，不暴露内部字段与堆栈。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data.schema import KDD_EVENT_TYPES

RiskLevel = Literal["high", "medium", "low", "unknown"]


class EventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: datetime = Field(description="行为发生时间，ISO 8601")
    source: str = Field(max_length=32, description="事件类型，如 video/problem/discussion")
    event: str = Field(default="unknown", max_length=64)
    object: str = Field(default="unknown", max_length=128)

    @field_validator("source")
    @classmethod
    def _check_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in KDD_EVENT_TYPES:
            raise ValueError(f"source 必须是 {list(KDD_EVENT_TYPES)} 之一")
        return normalized


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "enrollment_id": "E1000001",
                "course_id": "C01",
                "username": "u000001",
                "events": [
                    {"time": "2014-03-01T20:00:00", "source": "video", "event": "play_video", "object": "video_C01_1"},
                    {"time": "2014-03-01T20:10:00", "source": "problem", "event": "problem_check", "object": "problem_C01_1"},
                ],
            }
        },
    )

    enrollment_id: str = Field(min_length=1, max_length=64)
    course_id: str = Field(default="unknown", max_length=128)
    username: str = Field(default="unknown", max_length=128)
    events: list[EventIn] = Field(min_length=1, max_length=50000)


class RiskFactor(BaseModel):
    feature: str
    value: str | float
    contribution: float | None = None
    direction: Literal["risk_up", "risk_down", "unknown"] = "unknown"
    source: str = "shap"


class PredictResponse(BaseModel):
    enrollment_id: str
    course_id: str
    risk_probability: float
    # 校准前分数：当 calibration=isotonic/platt 时与 risk_probability 不同
    raw_probability: float | None = None
    calibrated: bool = False
    risk_level: RiskLevel
    decision_threshold: float
    top_factors: list[RiskFactor] = Field(default_factory=list)
    model_name: str
    model_version: str


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PredictRequest] = Field(min_length=1, max_length=500)
    include_factors: bool = False


class BatchSummary(BaseModel):
    n_records: int
    high: int = 0
    medium: int = 0
    low: int = 0
    n_flagged: int = 0
    mean_probability: float = 0.0


class BatchPredictResponse(BaseModel):
    items: list[PredictResponse]
    summary: BatchSummary


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


class ReadyResponse(BaseModel):
    status: str
    model_ready: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: str | None = None
    threshold: float
    cost_matrix: dict[str, Any] = Field(default_factory=dict)
    metrics_test: dict[str, Any] = Field(default_factory=dict)
    metrics_test_uncalibrated: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    is_calibrated: bool = False
    n_train: int | None = None
    n_test: int | None = None
    feature_count: int
    categorical_features: list[str] = Field(default_factory=list)
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    features: list[dict[str, str]] = Field(default_factory=list)
