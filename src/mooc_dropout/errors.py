"""类型化错误体系。

所有可预期的业务错误都继承 AppError，由全局异常处理器统一转换为
规范化 JSON 响应；未预期异常一律降级为 500，且绝不向客户端泄漏堆栈。
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """应用级错误基类。"""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }


class ValidationFailure(AppError):
    status_code = 422
    code = "validation_error"
    message = "请求参数不合法"


class DataQualityError(AppError):
    """数据缺失列、类型错误、标签非法等。"""

    status_code = 422
    code = "data_quality_error"
    message = "数据质量不满足要求"


class CostMatrixError(ValidationFailure):
    code = "cost_matrix_error"
    message = "代价矩阵配置不合法"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "资源不存在"


class ModelNotReadyError(AppError):
    status_code = 503
    code = "model_not_ready"
    message = "模型尚未训练或产物缺失，请先执行训练脚本"


class ConfigError(RuntimeError):
    """配置错误，进程启动阶段即应失败。"""
