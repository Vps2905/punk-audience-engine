from fastapi.testclient import TestClient

from app.main import app


def test_health_route_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_route_local_ok(monkeypatch):
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)

    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_route_production_missing_env_returns_503(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)

    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "AUDIENCE_API_KEY" in response.json()["missing_required"]
    assert "ECHO_DATABASE_URL" in response.json()["missing_required"]
