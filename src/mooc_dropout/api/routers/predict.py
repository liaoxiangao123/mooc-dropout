"""预测端点。控制器只做「解析请求 -> 调服务 -> 返回」，不含任何业务规则。"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, Response

from ...schemas import BatchPredictRequest, BatchPredictResponse, PredictRequest, PredictResponse
from ...service.prediction_service import PredictionService
from ..dependencies import get_service

router = APIRouter(prefix="/v1", tags=["predict"])

CSV_COLUMNS = [
    "priority",
    "enrollment_id",
    "course_id",
    "risk_probability",
    "risk_level",
    "decision_threshold",
    "top_factor",
    "top_factor_value",
    "model_version",
]


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="单条选课记录的辍学风险预测",
)
def predict(
    payload: PredictRequest,
    service: PredictionService = Depends(get_service),
) -> PredictResponse:
    result = service.predict(
        enrollment_id=payload.enrollment_id,
        course_id=payload.course_id,
        username=payload.username,
        events=[event.model_dump() for event in payload.events],
        top_k=5,
    )
    return PredictResponse(**result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="批量预测（最多 500 条），用于生成干预名单",
)
def predict_batch(
    payload: BatchPredictRequest,
    service: PredictionService = Depends(get_service),
) -> BatchPredictResponse:
    records = [record.model_dump() for record in payload.records]
    result = service.predict_batch(records, top_k=0)
    return BatchPredictResponse(**result)


@router.post(
    "/intervention-list",
    summary="导出干预名单（CSV，按风险降序，可直接派单）",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
)
def intervention_list(
    payload: BatchPredictRequest,
    only_flagged: bool = Query(True, description="仅导出达到决策阈值的学生"),
    min_level: str = Query("low", pattern="^(low|medium|high)$", description="最低风险等级"),
    service: PredictionService = Depends(get_service),
) -> Response:
    records = [record.model_dump() for record in payload.records]
    rows = service.build_intervention_list(records, only_flagged=only_flagged, min_level=min_level)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    # BOM 让 Excel 双击打开不乱码 —— 教师端最常见的打开方式
    content = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="intervention_list_{stamp}.csv"',
            "X-Total-Count": str(len(rows)),
        },
    )
