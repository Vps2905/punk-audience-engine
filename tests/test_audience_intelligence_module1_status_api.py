from fastapi.testclient import TestClient

from app.main import app


def test_module_1_status_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    client = TestClient(app)

    response = client.get("/api/audience-intelligence/module-1/status")

    assert response.status_code in {401, 403}


def test_module_1_status_returns_safe_readiness_summary(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", "postgresql://secret-user:secret-pass@example/db")
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", "postgresql://secret-user:secret-pass@example/db")
    monkeypatch.setenv("AUDIENCE_PRIVACY_BUDGET_DATABASE_URL", "postgresql://secret-user:secret-pass@example/db")

    client = TestClient(app)

    response = client.get(
        "/api/audience-intelligence/module-1/status",
        headers={"x-audience-api-key": "test-key"},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["module"] == "module_1_ingestion_privacy_layer"
    assert payload["status"] == "ready_for_preproduction_review"
    assert payload["api_key_configured"] is True
    assert payload["database_configured"] is True

    assert payload["privacy_controls"]["salted_hashing"] is True
    assert payload["privacy_controls"]["contribution_bounding"] is True
    assert payload["privacy_controls"]["privacy_budget_ledger"] is True
    assert payload["privacy_controls"]["lineage_logging"] is True
    assert payload["privacy_controls"]["legacy_route_auth_protection"] is True

    text = str(payload)
    assert "secret-user" not in text
    assert "secret-pass" not in text
    assert "example/db" not in text

    assert "POST /api/audience-intelligence/ingest/csv" in payload["protected_endpoints"]
    assert "POST /synthetic/generate/{job_id}" in payload["legacy_routes_hardened"]
