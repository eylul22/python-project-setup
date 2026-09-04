"""Tests for the API endpoints."""

from fastapi.testclient import TestClient

from my_first_project.api import app


def test_health_reports_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_model_loaded() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.json()["model_loaded"] is True


def test_clean_collapses_whitespace() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/clean", json={"text": "hello    world", "max_length": 50}
        )
    assert response.status_code == 200
    assert response.json()["result"] == "hello world"


def test_clean_rejects_empty_text() -> None:
    with TestClient(app) as client:
        response = client.post("/clean", json={"text": "", "max_length": 50})
    assert response.status_code == 422


def test_predict_identifies_setosa() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == "setosa"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_negative_measurement() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length": -1.0,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
    assert response.status_code == 422