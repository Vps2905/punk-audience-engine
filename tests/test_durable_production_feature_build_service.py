from __future__ import annotations

import pytest

from app.services.durable_production_feature_build_service import (
    DurableProductionFeatureBuildService,
)
from tests.test_production_feature_embedding_service import _request, _rows


class FakeState:
    def __init__(self, claim):
        self.claim_result = claim
        self.transitions = []

    def claim(self, request):
        return self.claim_result

    def transition(self, **kwargs):
        self.transitions.append(dict(kwargs))
        return dict(kwargs)


class FakePipeline:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def execute(self, request, rows, *, phase_observer=None):
        self.calls += 1
        if phase_observer:
            phase_observer("embedding")
        if self.error is not None:
            raise self.error
        if phase_observer:
            phase_observer("publishing")
        return {
            "status": "completed",
            "tenant_id": request.source.tenant_id,
            "feature_build_id": request.build_id,
            "feature_set_id": "feature_set_safe",
            "feature_set_version": 1,
            "feature_count": len(rows),
            "eligible_for_activation": False,
            "downstream_export_enabled": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "idempotency_replayed": False,
        }


def _new_claim():
    request = _request()
    return {
        "duplicate": False,
        "record": {
            "tenant_id": request.source.tenant_id,
            "feature_build_id": request.build_id,
            "status": "claimed",
            "terminal": False,
        },
    }


def test_durable_build_records_ordered_lifecycle_and_verified_receipt():
    state = FakeState(_new_claim())
    service = DurableProductionFeatureBuildService(
        pipeline=FakePipeline(),
        state_service=state,
    )

    result = service.execute(_request(), _rows())

    assert [
        transition["status"]
        for transition in state.transitions
    ] == ["validating", "embedding", "publishing", "completed"]
    completed = state.transitions[-1]
    assert completed["processed_feature_count"] == 2
    assert completed["feature_set_id"] == "feature_set_safe"
    assert completed["result_receipt"]["raw_identifiers_read"] is False
    assert result["durable_state"] == "completed"
    assert result["durable_claim_replayed"] is False
    assert result["downstream_export_enabled"] is False


def test_completed_duplicate_returns_persisted_receipt_without_rebuild():
    pipeline = FakePipeline()
    state = FakeState(
        {
            "duplicate": True,
            "record": {
                "tenant_id": "tenant_a",
                "feature_build_id": _request().build_id,
                "status": "completed",
                "terminal": True,
                "result_receipt": {
                    "status": "completed",
                    "feature_set_id": "feature_set_existing",
                    "feature_set_version": 1,
                    "feature_count": 2,
                    "downstream_export_enabled": False,
                },
            },
        }
    )

    result = DurableProductionFeatureBuildService(
        pipeline=pipeline,
        state_service=state,
    ).execute(_request(), _rows())

    assert pipeline.calls == 0
    assert state.transitions == []
    assert result["feature_set_id"] == "feature_set_existing"
    assert result["idempotency_replayed"] is True
    assert result["durable_claim_replayed"] is True


def test_active_duplicate_does_not_repeat_embedding_or_publication():
    pipeline = FakePipeline()
    state = FakeState(
        {
            "duplicate": True,
            "record": {
                "tenant_id": "tenant_a",
                "feature_build_id": _request().build_id,
                "status": "embedding",
                "terminal": False,
            },
        }
    )

    result = DurableProductionFeatureBuildService(
        pipeline=pipeline,
        state_service=state,
    ).execute(_request(), _rows())

    assert pipeline.calls == 0
    assert result["status"] == "embedding"
    assert result["downstream_export_enabled"] is False


def test_invalid_canonical_input_is_quarantined_and_reraised():
    state = FakeState(_new_claim())
    service = DurableProductionFeatureBuildService(
        pipeline=FakePipeline(ValueError("invalid canonical row")),
        state_service=state,
    )

    with pytest.raises(ValueError, match="invalid canonical"):
        service.execute(_request(), _rows())

    assert state.transitions[-1]["status"] == "quarantined"
    assert state.transitions[-1]["reason_code"] == (
        "invalid_canonical_feature_input"
    )


def test_unapproved_model_is_blocked_and_reraised():
    state = FakeState(_new_claim())
    service = DurableProductionFeatureBuildService(
        pipeline=FakePipeline(
            RuntimeError("Embedding model approval failed")
        ),
        state_service=state,
    )

    with pytest.raises(RuntimeError, match="approval"):
        service.execute(_request(), _rows())

    assert state.transitions[-1]["status"] == "blocked"
    assert state.transitions[-1]["reason_code"] == (
        "embedding_model_not_approved"
    )
