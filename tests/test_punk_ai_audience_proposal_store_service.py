from __future__ import annotations

import json

import pytest

from app.services.punk_ai_audience_proposal_store_service import (
    ProposalIdempotencyConflictError,
    PunkAIAudienceProposalStore,
    UnsafeProposalDocumentError,
)


class FakeResult:
    def __init__(self, *, row=None, scalar=None):
        self.row = row
        self.scalar = scalar

    def mappings(self):
        return self

    def first(self):
        return self.row

    def one(self):
        return self.row

    def scalar_one_or_none(self):
        return self.scalar


class FakeConnection:
    def __init__(self, *, stored_fingerprint=None):
        self.calls = []
        self.params = None
        self.stored_fingerprint = stored_fingerprint

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        self.calls.append((sql, params))
        if "INSERT INTO" in sql:
            self.params = params
            return FakeResult(scalar=params["proposal_id"])
        if "FROM public.punk_ai_audience_proposals" in sql:
            if self.params is None:
                return FakeResult(row=None)
            return FakeResult(
                row={
                    "proposal_id": self.params["proposal_id"],
                    "request_fingerprint": (
                        self.stored_fingerprint
                        or self.params["request_fingerprint"]
                    ),
                    "response_document": json.loads(
                        self.params["response_document"]
                    ),
                }
            )
        return FakeResult()


class FakeContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class FakeEngine:
    def __init__(self, *, stored_fingerprint=None):
        self.connection = FakeConnection(
            stored_fingerprint=stored_fingerprint
        )

    def begin(self):
        return FakeContext(self.connection)

    def connect(self):
        return self.connection

    def dispose(self):
        return None


def _response() -> dict:
    return {
        "contract_version": "2026-07-25",
        "proposal_id": "audience_proposal_1",
        "idempotency_key": "unique-request-0001",
        "tenant_id": "tenant_a",
        "campaign_id": "campaign-1",
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
        "candidate_cohorts": [],
    }


def test_proposal_store_requires_explicit_database_without_fallback():
    store = PunkAIAudienceProposalStore(
        environment={
            "ECHO_DATABASE_URL": "postgresql://historical-source",
            "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://feature-reader",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="AUDIENCE_PROPOSAL_DATABASE_URL",
    ):
        store.get(
            tenant_id="tenant_a",
            campaign_id="campaign-1",
            idempotency_key="unique-request-0001",
        )


@pytest.mark.parametrize(
    "blocked_document",
    [
        {"candidate_cohorts": [{"raw_maids": ["blocked"]}]},
        {"candidate_cohorts": [{"device_id": "blocked"}]},
        {"candidate_cohorts": [{"embedding": [0.1, 0.2]}]},
        {"lineage": {"source": "blocked"}},
        {"database_url": "blocked"},
    ],
)
def test_proposal_store_rejects_unsafe_response_fields(
    blocked_document,
):
    response = _response()
    response.update(blocked_document)
    store = PunkAIAudienceProposalStore(
        database_url="postgresql://proposal-runtime"
    )

    with pytest.raises(
        UnsafeProposalDocumentError,
        match="blocked field",
    ):
        store.record(
            tenant_id="tenant_a",
            campaign_id="campaign-1",
            idempotency_key="unique-request-0001",
            request_fingerprint="a" * 64,
            response=response,
        )


def test_proposal_store_rejects_export_enabled_document():
    response = _response()
    response["downstream_export_enabled"] = True
    store = PunkAIAudienceProposalStore(
        database_url="postgresql://proposal-runtime"
    )

    with pytest.raises(
        UnsafeProposalDocumentError,
        match="cannot enable",
    ):
        store.record(
            tenant_id="tenant_a",
            campaign_id="campaign-1",
            idempotency_key="unique-request-0001",
            request_fingerprint="a" * 64,
            response=response,
        )


def test_proposal_store_rejects_credential_like_values():
    response = _response()
    response["explanation"] = (
        "Use postgresql://example.invalid/private"
    )
    store = PunkAIAudienceProposalStore(
        database_url="postgresql://proposal-runtime"
    )

    with pytest.raises(
        UnsafeProposalDocumentError,
        match="credential-like",
    ):
        store.record(
            tenant_id="tenant_a",
            campaign_id="campaign-1",
            idempotency_key="unique-request-0001",
            request_fingerprint="a" * 64,
            response=response,
        )


def test_proposal_store_inserts_with_tenant_context_and_no_update():
    engine = FakeEngine()
    store = PunkAIAudienceProposalStore(
        database_url="postgresql://proposal-runtime",
        engine_factory=lambda *_args, **_kwargs: engine,
    )

    result = store.record(
        tenant_id="tenant-a",
        campaign_id="campaign-1",
        idempotency_key="unique-request-0001",
        request_fingerprint="a" * 64,
        response=_response(),
    )

    assert result["inserted"] is True
    assert result["request_fingerprint"] == "a" * 64
    assert result["response"]["downstream_export_enabled"] is False
    sql = "\n".join(
        statement
        for statement, _params in engine.connection.calls
    )
    assert "set_config" in sql
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql
    assert "DO UPDATE" not in sql


def test_proposal_store_detects_concurrent_key_collision():
    engine = FakeEngine(stored_fingerprint="b" * 64)
    store = PunkAIAudienceProposalStore(
        database_url="postgresql://proposal-runtime",
        engine_factory=lambda *_args, **_kwargs: engine,
    )

    with pytest.raises(
        ProposalIdempotencyConflictError,
        match="different request",
    ):
        store.record(
            tenant_id="tenant-a",
            campaign_id="campaign-1",
            idempotency_key="unique-request-0001",
            request_fingerprint="a" * 64,
            response=_response(),
        )
