"""API 层：健康检查、预测端点、参数校验、错误结构。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mooc_dropout.api.app import create_app
from mooc_dropout.repository.artifact_repository import ArtifactRepository
from mooc_dropout.service.prediction_service import PredictionService


@pytest.fixture(scope="module")
def client(trained):
    settings, _ = trained
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def _payload(logs, enrollment_id: str, limit: int = 40) -> dict:
    subset = logs.loc[logs["enrollment_id"] == enrollment_id].head(limit)
    return {
        "enrollment_id": enrollment_id,
        "course_id": "C01",
        "username": "u_test",
        "events": [
            {
                "time": row.time.isoformat(),
                "source": row.source,
                "event": row.event,
                "object": row.object,
            }
            for row in subset.itertuples()
        ],
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["env"] == "test"
    assert response.headers["X-Request-ID"]


def test_request_id_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert response.headers["X-Request-ID"] == "trace-123"


def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_predict_returns_risk(client, logs):
    payload = _payload(logs, logs["enrollment_id"].iloc[0])
    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_level"] in {"high", "medium", "low"}
    assert body["decision_threshold"] > 0
    assert body["model_name"]
    assert isinstance(body["top_factors"], list)


def test_predict_rejects_unknown_event_type(client, logs):
    payload = _payload(logs, logs["enrollment_id"].iloc[0], limit=5)
    payload["events"][0]["source"] = "not_a_real_event"
    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["errors"]


def test_predict_rejects_empty_events(client, logs):
    payload = _payload(logs, logs["enrollment_id"].iloc[0], limit=3)
    payload["events"] = []
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_fields(client, logs):
    payload = _payload(logs, logs["enrollment_id"].iloc[0], limit=3)
    payload["unexpected"] = 1
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict(client, logs):
    ids = logs["enrollment_id"].drop_duplicates().head(3).tolist()
    payload = {
        "records": [_payload(logs, eid, limit=20) for eid in ids],
        "include_factors": False,
    }
    response = client.post("/v1/predict/batch", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["items"]) == len(ids)
    summary = body["summary"]
    assert summary["n_records"] == len(ids)
    assert summary["high"] + summary["medium"] + summary["low"] == len(ids)


def test_model_info(client):
    response = client.get("/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_count"] >= 40
    assert "course_id" in body["categorical_features"]
    assert body["threshold"] > 0
    assert len(body["features"]) == body["feature_count"]
    assert body["is_calibrated"] is False
    assert body["calibration"]["method"] == "none"


def test_intervention_list_csv(client, logs):
    ids = logs["enrollment_id"].drop_duplicates().head(4).tolist()
    payload = {"records": [_payload(logs, eid, limit=20) for eid in ids]}
    response = client.post("/v1/intervention-list", json=payload)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    text = response.content.decode("utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("priority,enrollment_id")
    assert int(response.headers["X-Total-Count"]) == len(lines) - 1


def test_intervention_list_respects_high_level(client, logs):
    ids = logs["enrollment_id"].drop_duplicates().head(5).tolist()
    payload = {"records": [_payload(logs, eid, limit=20) for eid in ids]}
    response = client.post("/v1/intervention-list?min_level=high", json=payload)

    assert response.status_code == 200
    text = response.content.decode("utf-8-sig")
    for line in text.splitlines()[1:]:
        if line.strip():
            assert ",high," in line


def test_feature_catalog(client):
    response = client.get("/v1/model/features")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 40
    assert all("group" in item for item in body["items"])


def test_demo_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "MOOC 辍学风险预测" in response.text


def test_not_ready_when_artifacts_missing(test_settings, tmp_path):
    empty = ArtifactRepository(tmp_path / "empty")
    app = create_app(test_settings, PredictionService(empty))
    with TestClient(app) as local_client:
        ready = local_client.get("/ready")
        assert ready.status_code == 503

        response = local_client.post(
            "/v1/predict",
            json={
                "enrollment_id": "E1",
                "events": [{"time": "2014-03-01T10:00:00", "source": "video"}],
            },
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_ready"
