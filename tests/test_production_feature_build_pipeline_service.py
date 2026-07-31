from __future__ import annotations

from tests.test_production_feature_embedding_service import (
    _request,
    _rows,
    _service,
)

import pytest

from app.services.production_feature_build_pipeline_service import (
    ProductionFeatureBuildPipelineService,
)


class FakeStore:
    def __init__(self):
        self.records = {}
        self.save_calls = 0

    def get_feature_set(
        self,
        *,
        tenant_id,
        feature_set_id=None,
        version=None,
        data_use_mode=None,
    ):
        key = (tenant_id, feature_set_id, version, data_use_mode)
        if key not in self.records:
            raise FileNotFoundError
        return dict(self.records[key])

    def save_feature_set(self, feature_set):
        self.save_calls += 1
        record = feature_set.to_record()
        record["lineage"] = dict(feature_set.lineage)
        key = (
            feature_set.tenant_id,
            feature_set.feature_set_id,
            feature_set.version,
            feature_set.data_use_mode,
        )
        self.records[key] = record
        return {
            "status": "completed",
            "feature_set_id": feature_set.feature_set_id,
            "feature_set_version": feature_set.version,
            "feature_count": feature_set.feature_count,
        }


class FakeRegistry:
    def __init__(self, *, approved=True):
        self.approved = approved
        self.calls = 0

    def require_approved(self, *, tenant_id, model):
        self.calls += 1
        return {
            "tenant_id": tenant_id,
            "model_fingerprint": model.fingerprint,
            "approved": self.approved,
        }


def test_pipeline_publishes_once_and_replays_content_addressed_build():
    store = FakeStore()
    registry = FakeRegistry()
    service = ProductionFeatureBuildPipelineService(
        embedding_service=_service(),
        feature_store=store,
        model_registry=registry,
    )

    first = service.execute(_request(), _rows())
    second = service.execute(_request(), _rows())

    assert first["status"] == "completed"
    assert first["idempotency_replayed"] is False
    assert second["idempotency_replayed"] is True
    assert first["feature_set_id"] == second["feature_set_id"]
    assert store.save_calls == 1
    assert first["eligible_for_activation"] is False
    assert first["downstream_export_enabled"] is False
    assert first["raw_identifiers_read"] is False
    assert first["raw_identifiers_stored"] is False
    assert registry.calls == 2


def test_pipeline_reports_ordered_embedding_and_publication_phases():
    store = FakeStore()
    service = ProductionFeatureBuildPipelineService(
        embedding_service=_service(),
        feature_store=store,
        model_registry=FakeRegistry(),
    )
    first_phases = []
    replay_phases = []

    service.execute(
        _request(),
        _rows(),
        phase_observer=first_phases.append,
    )
    service.execute(
        _request(),
        _rows(),
        phase_observer=replay_phases.append,
    )

    assert first_phases == ["embedding", "publishing"]
    assert replay_phases == ["embedding", "publishing"]


def test_pipeline_rejects_conflicting_existing_identity():
    store = FakeStore()
    service = ProductionFeatureBuildPipelineService(
        embedding_service=_service(),
        feature_store=store,
        model_registry=FakeRegistry(),
    )
    first = service.execute(_request(), _rows())
    key = next(iter(store.records))
    store.records[key]["model_version"] = "different-revision"

    with pytest.raises(RuntimeError, match="conflicts"):
        service.execute(_request(), _rows())

    assert first["feature_set_id"] == key[1]


def test_pipeline_verifies_store_receipt():
    class BadReceiptStore(FakeStore):
        def save_feature_set(self, feature_set):
            return {
                "status": "completed",
                "feature_set_id": "wrong",
                "feature_set_version": 1,
                "feature_count": feature_set.feature_count,
            }

    service = ProductionFeatureBuildPipelineService(
        embedding_service=_service(),
        feature_store=BadReceiptStore(),
        model_registry=FakeRegistry(),
    )
    with pytest.raises(RuntimeError, match="receipt"):
        service.execute(_request(), _rows())


def test_pipeline_fails_before_embedding_when_model_is_not_approved():
    class ForbiddenEmbedding:
        def build(self, request, rows):
            raise AssertionError("embedding must not run")

    service = ProductionFeatureBuildPipelineService(
        embedding_service=ForbiddenEmbedding(),
        feature_store=FakeStore(),
        model_registry=FakeRegistry(approved=False),
    )

    with pytest.raises(RuntimeError, match="approval"):
        service.execute(_request(), _rows())
