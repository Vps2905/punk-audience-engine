from fastapi.testclient import TestClient

from app.main import app


def test_module_1_status_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    client = TestClient(app)

    response = client.get(
        "/api/audience-intelligence/module-1/status",
        headers={"x-audience-tenant-id": "tenant-a"},
    )

    assert response.status_code in {401, 403}


def test_module_1_status_returns_safe_readiness_summary(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", "postgresql://secret-user:secret-pass@example/db")
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", "postgresql://secret-user:secret-pass@example/db")
    monkeypatch.setenv("AUDIENCE_PRIVACY_BUDGET_DATABASE_URL", "postgresql://secret-user:secret-pass@example/db")

    client = TestClient(app)

    response = client.get(
        "/api/audience-intelligence/module-1/status",
        headers={
            "x-audience-api-key": "test-key",
            "x-audience-tenant-id": "tenant-a",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["module"] == "module_1_ingestion_privacy_layer"
    assert payload["status"] == "disabled_pending_deployment"
    assert payload["api_key_configured"] is True
    assert payload["database_configured"] is True

    assert payload["privacy_controls"]["hmac_sha256_tokenization"] is True
    assert payload["privacy_controls"][
        "entity_disjoint_privacy_partitions"
    ] is True
    assert payload["privacy_controls"][
        "deletion_and_opt_out_propagation"
    ] is True
    assert payload["privacy_controls"]["contribution_bounding"] is True
    assert payload["privacy_controls"]["privacy_budget_ledger"] is True
    assert payload["privacy_controls"]["lineage_logging"] is True
    assert payload["privacy_controls"]["legacy_route_auth_protection"] is True
    assert payload["privacy_controls"]["verified_tenant_request_boundary"] is True
    assert payload["privacy_controls"]["request_id_propagation"] is True

    text = str(payload)
    assert "secret-user" not in text
    assert "secret-pass" not in text
    assert "example/db" not in text

    assert "POST /api/audience-intelligence/ingest/csv" in payload["protected_endpoints"]
    assert "POST /synthetic/generate/{job_id}" in payload["legacy_routes_hardened"]
