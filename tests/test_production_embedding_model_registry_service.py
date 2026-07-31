from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.production_embedding_model_registry_service import (
    ProductionEmbeddingModelRegistryService,
)
from tests.test_production_feature_embedding_service import _model


class FakeResult:
    def __init__(self, *, scalar=None, row=None):
        self._scalar = scalar
        self._row = row

    def scalar(self):
        return self._scalar

    def mappings(self):
        return self

    def first(self):
        return self._row


class FakeConnection:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append((sql, params or {}))
        if "to_regclass" in sql:
            return FakeResult(scalar=True)
        if "FROM audience_embedding_models" in sql:
            return FakeResult(row=self.row)
        return FakeResult()


class FakeEngine:
    def __init__(self, row):
        self.connection = FakeConnection(row)

    def connect(self):
        return self.connection


def _approved_row():
    model = _model()
    return {
        "tenant_id": "tenant_a",
        "model_fingerprint": model.fingerprint,
        "backend": model.backend,
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "embedding_dimension": model.dimension,
        "normalize_embeddings": model.normalize_embeddings,
        "document_prefix": model.document_prefix,
        "query_prefix": model.query_prefix,
        "status": "approved",
        "benchmark_status": "passed",
        "benchmark_report": {"nDCG@10": 0.82},
        "approved_by": "model_owner",
        "approved_at": datetime.now(timezone.utc),
    }


def test_registry_requires_exact_tenant_approved_model_revision():
    engine = FakeEngine(_approved_row())
    result = ProductionEmbeddingModelRegistryService(
        engine=engine
    ).require_approved(
        tenant_id="tenant-a",
        model=_model(),
    )

    assert result["approved"] is True
    assert result["model_fingerprint"] == _model().fingerprint
    calls = engine.connection.calls
    assert next(
        index for index, (sql, _) in enumerate(calls)
        if "set_config" in sql
    ) < next(
        index for index, (sql, _) in enumerate(calls)
        if "FROM audience_embedding_models" in sql
    )


def test_registry_blocks_unapproved_or_unregistered_model():
    row = _approved_row()
    row["status"] = "evaluation"
    row["benchmark_status"] = "pending"
    service = ProductionEmbeddingModelRegistryService(
        engine=FakeEngine(row)
    )
    with pytest.raises(RuntimeError, match="approval"):
        service.require_approved(
            tenant_id="tenant-a",
            model=_model(),
        )

    missing = ProductionEmbeddingModelRegistryService(
        engine=FakeEngine(None)
    )
    with pytest.raises(RuntimeError, match="not registered"):
        missing.require_approved(
            tenant_id="tenant-a",
            model=_model(),
        )


def test_registry_detects_fingerprint_record_conflict():
    row = _approved_row()
    row["model_revision"] = "different"
    service = ProductionEmbeddingModelRegistryService(
        engine=FakeEngine(row)
    )

    with pytest.raises(RuntimeError, match="conflicts"):
        service.require_approved(
            tenant_id="tenant-a",
            model=_model(),
        )
