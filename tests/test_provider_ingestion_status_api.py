from fastapi.testclient import TestClient

from app.main import app
from app.models.provider_ingestion_contracts import ProviderObjectDescriptor
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_scale_acceptance_service import (
    ProviderScaleAcceptanceService,
)


def test_provider_status_reports_disabled_without_live_data(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("PROVIDER_GATEWAY_ENABLED", "false")

    response = TestClient(app).get(
        "/api/audience-intelligence/provider-ingestion/status",
        headers={
            "x-audience-api-key": "test-key",
            "x-audience-tenant-id": "tenant_a",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "disabled"
    assert response.json()["live_provider_data_required"] is False


def test_provider_run_endpoint_returns_only_safe_operational_fields(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    db_url = f"sqlite:///{tmp_path / 'status.db'}"
    monkeypatch.setenv("PROVIDER_INGESTION_DATABASE_URL", db_url)
    state = ProviderIngestionStateService(database_url=db_url)
    descriptor = ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="dataset_a",
        bucket="provider-landing",
        key="delivery/object.csv",
        size_bytes=10,
        content_type="text/csv",
        version_id="version-1",
        server_side_encryption="AES256",
    )
    ingestion_id = state.claim_object(descriptor)["record"]["ingestion_id"]

    response = TestClient(app).get(
        (
            "/api/audience-intelligence/provider-ingestion/runs/"
            f"{ingestion_id}"
        ),
        headers={
            "x-audience-api-key": "test-key",
            "x-audience-tenant-id": "tenant_a",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ingestion_id"] == ingestion_id
    assert payload["status"] == "received"
    assert "metadata" not in payload
    assert "checksum_sha256" not in payload


def test_scale_readiness_is_tenant_scoped_and_fail_closed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    db_url = f"sqlite:///{tmp_path / 'scale-status.db'}"
    monkeypatch.setenv("PROVIDER_INGESTION_DATABASE_URL", db_url)
    ProviderScaleAcceptanceService(database_url=db_url)

    response = TestClient(app).get(
        "/api/audience-intelligence/provider-ingestion/scale-readiness",
        headers={
            "x-audience-api-key": "test-key",
            "x-audience-tenant-id": "tenant_a",
        },
    )

    assert response.status_code == 200
    assert response.json()["production_scale_certified"] is False
    assert response.json()["status"] == "scale_evidence_not_recorded"
