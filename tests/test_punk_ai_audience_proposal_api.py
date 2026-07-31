from fastapi.testclient import TestClient

from app.api import punk_ai_audience_proposals as api_module
from app.core.tenant_request_auth import tenant_signature
from app.main import app
from app.services.punk_ai_audience_proposal_store_service import (
    ProposalIdempotencyConflictError,
)


class FakeProposalService:
    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        return {
            "contract_version": "2026-07-25",
            "proposal_id": "proposal_1",
            "tenant_id": request["tenant_id"],
            "campaign_id": request["campaign_id"],
            "status": "historical_preview_ready",
            "candidate_cohorts": [],
            "downstream_export_enabled": False,
        }


class FakeFeatureStore:
    def get_feature_set(self, **kwargs):
        return {
            "tenant_id": kwargs["tenant_id"],
            "feature_set_id": "feature_set_1",
            "version": 1,
            "status": "ready_for_historical_preview",
            "source_mode": "legacy_postgres_safe_vectors",
            "data_use_mode": "historical_preview",
            "source_latest_at": "2026-07-08T08:40:00+00:00",
            "freshness_status": "stale",
            "feature_count": 85,
            "model_backend": "sklearn_hashing",
            "model_name": "sklearn_hashing_vectorizer",
            "model_version": "legacy_snapshot",
            "privacy_policy_version": "privacy_1",
            "rights_policy_id": "rights_1",
            "purpose": "internal_evaluation",
            "eligible_for_retrieval": True,
            "eligible_for_activation": False,
        }


def _payload():
    return {
        "tenant_id": "tenant-a",
        "campaign_id": "campaign-1",
        "objective": "store_visits",
        "audience_intent": "evening restaurant visitors",
        "locations": ["Montreal"],
        "categories": ["restaurant"],
        "dayparts": ["evening"],
        "budget": {"currency": "CAD", "daily": 100},
        "exclusions": [],
        "destination": "meta",
        "idempotency_key": "unique-request-0001",
        "execution_mode": "historical_preview",
    }


def test_punk_ai_contract_requires_matching_tenant_header(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")

    response = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "different-tenant",
        },
        json=_payload(),
    )

    assert response.status_code == 403


def test_punk_ai_contract_forwards_versioned_request_without_export(
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    service = FakeProposalService()
    monkeypatch.setattr(
        api_module,
        "build_audience_feature_proposal_service",
        lambda: service,
    )

    response = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "tenant-a",
        },
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json()["contract_version"] == "2026-07-25"
    assert response.json()["downstream_export_enabled"] is False
    assert service.requests[0]["tenant_id"] == "tenant-a"
    assert service.requests[0]["execution_mode"] == "historical_preview"


def test_punk_ai_contract_remains_api_key_protected(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")

    response = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={"X-Audience-Tenant-Id": "tenant-a"},
        json=_payload(),
    )

    assert response.status_code == 401


def test_punk_ai_contract_requires_valid_tenant_signature_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("REQUIRE_PUNK_AI_TENANT_SIGNATURE", "true")
    monkeypatch.setenv(
        "PUNK_AI_TENANT_AUTH_SECRET",
        "test-tenant-secret",
    )

    missing = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "tenant-a",
        },
        json=_payload(),
    )
    assert missing.status_code == 401

    invalid = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "tenant-a",
            "X-Audience-Tenant-Signature": "wrong-signature",
        },
        json=_payload(),
    )
    assert invalid.status_code == 403

    service = FakeProposalService()
    monkeypatch.setattr(
        api_module,
        "build_audience_feature_proposal_service",
        lambda: service,
    )
    valid = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "tenant-a",
            "X-Audience-Tenant-Signature": tenant_signature(
                "tenant-a",
                secret="test-tenant-secret",
            ),
        },
        json=_payload(),
    )
    assert valid.status_code == 200


def test_current_feature_set_status_exposes_no_embeddings_or_lineage(
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setattr(
        api_module,
        "build_audience_feature_store",
        lambda: FakeFeatureStore(),
    )

    response = TestClient(app).get(
        "/api/audience-intelligence/punk-ai/v1/feature-sets/current",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "tenant-a",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["freshness_status"] == "stale"
    assert payload["eligible_for_activation"] is False
    assert payload["downstream_export_enabled"] is False
    assert "embedding" not in payload
    assert "lineage" not in payload


def test_punk_ai_contract_returns_conflict_for_reused_key(
    monkeypatch,
):
    class ConflictService:
        def propose(self, _request):
            raise ProposalIdempotencyConflictError(
                "The idempotency key is already bound to a different request."
            )

    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setattr(
        api_module,
        "build_audience_feature_proposal_service",
        lambda: ConflictService(),
    )

    response = TestClient(app).post(
        "/api/audience-intelligence/punk-ai/v1/proposals",
        headers={
            "X-Audience-API-Key": "test-key",
            "X-Audience-Tenant-Id": "tenant-a",
        },
        json=_payload(),
    )

    assert response.status_code == 409
