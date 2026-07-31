from __future__ import annotations

from copy import deepcopy

import pytest

from app.services.durable_audience_feature_proposal_service import (
    DurableAudienceFeatureProposalService,
)
from app.services.punk_ai_audience_proposal_store_service import (
    ProposalIdempotencyConflictError,
)


def _request() -> dict:
    return {
        "tenant_id": "tenant-a",
        "campaign_id": "campaign-1",
        "objective": "store_visits",
        "audience_intent": "Evening restaurant visitors",
        "locations": ["Montreal", "Toronto"],
        "categories": ["restaurant"],
        "dayparts": ["evening"],
        "budget": {"currency": "CAD", "daily": 100},
        "exclusions": [],
        "destination": "meta",
        "idempotency_key": "unique-request-0001",
        "execution_mode": "historical_preview",
        "feature_set_id": "feature_set_1",
        "feature_set_version": 1,
        "top_k": 10,
    }


def _response(request: dict) -> dict:
    return {
        "contract_version": "2026-07-25",
        "proposal_id": "audience_proposal_1",
        "idempotency_key": request["idempotency_key"],
        "tenant_id": request["tenant_id"],
        "campaign_id": request["campaign_id"],
        "status": "historical_preview_ready",
        "execution_mode": "historical_preview",
        "feature_set": {
            "feature_set_id": "feature_set_1",
            "version": 1,
        },
        "approval_status": "blocked_historical_source",
        "activation_eligible": False,
        "safe_export_eligible": False,
        "downstream_export_enabled": False,
    }


class FakeDelegate:
    def __init__(self):
        self.calls = 0

    def propose(self, request):
        self.calls += 1
        return _response(request)


class FakeStore:
    def __init__(self):
        self.row = None

    def get(self, **_identity):
        return deepcopy(self.row)

    def record(self, *, request_fingerprint, response, **_identity):
        if self.row is None:
            self.row = {
                "proposal_id": response["proposal_id"],
                "request_fingerprint": request_fingerprint,
                "response": deepcopy(dict(response)),
                "inserted": True,
            }
        return deepcopy(self.row)


def test_identical_request_is_persisted_once_and_replayed():
    delegate = FakeDelegate()
    store = FakeStore()
    service = DurableAudienceFeatureProposalService(
        delegate=delegate,
        proposal_store=store,
    )

    first = service.propose(_request())
    second = service.propose(_request())

    assert first["proposal_persisted"] is True
    assert first["idempotency_replayed"] is False
    assert second["proposal_persisted"] is True
    assert second["idempotency_replayed"] is True
    assert first["proposal_id"] == second["proposal_id"]
    assert delegate.calls == 1


def test_semantically_equivalent_filter_order_replays():
    delegate = FakeDelegate()
    store = FakeStore()
    service = DurableAudienceFeatureProposalService(
        delegate=delegate,
        proposal_store=store,
    )
    first_request = _request()
    reordered = deepcopy(first_request)
    reordered["locations"] = ["Toronto", "Montreal", "Toronto"]

    service.propose(first_request)
    result = service.propose(reordered)

    assert result["idempotency_replayed"] is True
    assert delegate.calls == 1


def test_reused_key_with_changed_request_is_conflict():
    delegate = FakeDelegate()
    store = FakeStore()
    service = DurableAudienceFeatureProposalService(
        delegate=delegate,
        proposal_store=store,
    )
    changed = _request()
    changed["budget"]["daily"] = 200

    service.propose(_request())

    with pytest.raises(
        ProposalIdempotencyConflictError,
        match="different request",
    ):
        service.propose(changed)

    assert delegate.calls == 1


def test_response_identity_must_match_request():
    class WrongTenantDelegate:
        def propose(self, request):
            response = _response(request)
            response["tenant_id"] = "different-tenant"
            return response

    service = DurableAudienceFeatureProposalService(
        delegate=WrongTenantDelegate(),
        proposal_store=FakeStore(),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        service.propose(_request())
