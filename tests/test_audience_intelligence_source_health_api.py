from fastapi.testclient import TestClient

from app.main import app


def test_source_health_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    client = TestClient(app)
    response = client.get(
        "/api/audience-intelligence/source-health",
        headers={"x-audience-tenant-id": "tenant-a"},
    )

    assert response.status_code in {401, 403}


def test_source_health_reports_missing_source_db_without_leaking_secrets(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    for name in [
        "ECHO_DATABASE_URL",
        "DATABASE_URL",
        "POSTGRES_URL",
        "POSTGRES_DATABASE_URL",
        "SUPABASE_DB_URL",
        "DB_URL",
    ]:
        monkeypatch.delenv(name, raising=False)

    client = TestClient(app)
    response = client.get(
        "/api/audience-intelligence/source-health",
        headers={
            "x-audience-api-key": "test-key",
            "x-audience-tenant-id": "tenant-a",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["enabled"] is False
    assert data["status"] == "not_configured"
    assert data["reason"] == "no_source_database_url"
    assert data["source_table"] == "public.maid_extractions"

    payload_text = response.text
    assert "postgresql://" not in payload_text
    assert "password" not in payload_text.lower()
    assert "secret" not in payload_text.lower()
