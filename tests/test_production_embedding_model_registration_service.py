from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.services.production_embedding_model_registration_service import (
    ProductionEmbeddingModelRegistrationService,
)
from tests.test_production_embedding_benchmark_service import (
    production_benchmark_dataset,
    production_benchmark_report,
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

    def one(self):
        return self._row


class FakeConnection:
    def __init__(self, existing=None):
        self.record = existing
        self.calls = []

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        self.calls.append((sql, params))
        if "to_regclass" in sql:
            return FakeResult(scalar=True)
        if "INSERT INTO audience_embedding_models" in sql:
            if self.record is not None:
                return FakeResult(row=None)
            self.record = {
                "tenant_id": params["tenant_id"],
                "model_fingerprint": params["model_fingerprint"],
                "backend": params["backend"],
                "model_name": params["model_name"],
                "model_revision": params["model_revision"],
                "embedding_dimension": params["embedding_dimension"],
                "normalize_embeddings": params["normalize_embeddings"],
                "document_prefix": params["document_prefix"],
                "query_prefix": params["query_prefix"],
                "status": "approved",
                "benchmark_status": "passed",
                "benchmark_report": json.loads(
                    params["benchmark_report"]
                ),
                "approved_by": params["approved_by"],
                "approved_at": datetime.now(timezone.utc),
            }
            return FakeResult(
                row={
                    "model_fingerprint": params["model_fingerprint"]
                }
            )
        if "FROM audience_embedding_models" in sql:
            return FakeResult(row=self.record)
        return FakeResult()


class FakeContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class FakeDialect:
    name = "postgresql"


class FakeEngine:
    dialect = FakeDialect()

    def __init__(self, existing=None):
        self.connection = FakeConnection(existing)
        self.disposed = False

    def begin(self):
        return FakeContext(self.connection)

    def dispose(self):
        self.disposed = True


class FakePreflight:
    def run(self, **_kwargs):
        return {
            "status": "phase2_schema_ready",
            "same_database_as_source": False,
        }


def _environment():
    return {
        "ECHO_DATABASE_URL": "configured-source",
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": "configured-feature",
    }


def _benchmark():
    return production_benchmark_report()


def _existing_record():
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
        "benchmark_report": _benchmark(),
        "approved_by": "model_owner",
        "approved_at": datetime.now(timezone.utc),
    }


def test_registration_requires_two_explicit_operator_confirmations():
    service = ProductionEmbeddingModelRegistrationService(
        environment=_environment()
    )
    with pytest.raises(RuntimeError, match="target confirmation"):
        service.register_approved(
            tenant_id="tenant-a",
            model=_model(),
            benchmark_report=_benchmark(),
            benchmark_dataset=production_benchmark_dataset(),
            approved_by="model-owner",
            punk_owned_target_confirmed=False,
            benchmark_accepted=True,
        )
    with pytest.raises(RuntimeError, match="benchmark acceptance"):
        service.register_approved(
            tenant_id="tenant-a",
            model=_model(),
            benchmark_report=_benchmark(),
            benchmark_dataset=production_benchmark_dataset(),
            approved_by="model-owner",
            punk_owned_target_confirmed=True,
            benchmark_accepted=False,
        )


def test_registration_requires_benchmark_provenance_and_pass():
    service = ProductionEmbeddingModelRegistrationService(
        environment=_environment()
    )
    invalid = _benchmark()
    invalid["passed"] = False
    with pytest.raises(ValueError, match="pass"):
        service.register_approved(
            tenant_id="tenant-a",
            model=_model(),
            benchmark_report=invalid,
            benchmark_dataset=production_benchmark_dataset(),
            approved_by="model-owner",
            punk_owned_target_confirmed=True,
            benchmark_accepted=True,
        )


def test_approved_model_registration_is_insert_only_and_idempotent():
    engine = FakeEngine()
    service = ProductionEmbeddingModelRegistrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=FakePreflight,
    )
    first = service.register_approved(
        tenant_id="tenant-a",
        model=_model(),
        benchmark_report=_benchmark(),
        benchmark_dataset=production_benchmark_dataset(),
        approved_by="model-owner",
        punk_owned_target_confirmed=True,
        benchmark_accepted=True,
    )
    second = service.register_approved(
        tenant_id="tenant-a",
        model=_model(),
        benchmark_report=_benchmark(),
        benchmark_dataset=production_benchmark_dataset(),
        approved_by="model-owner",
        punk_owned_target_confirmed=True,
        benchmark_accepted=True,
    )

    assert first["status"] == "approved_model_registered"
    assert second["status"] == "approved_model_already_registered"
    assert first["model_fingerprint"] == _model().fingerprint
    assert first["features_built"] is False
    assert first["activation_or_export_performed"] is False
    sql = "\n".join(statement for statement, _ in engine.connection.calls)
    assert "ON CONFLICT DO NOTHING" in sql
    assert "UPDATE audience_embedding_models" not in sql


def test_existing_conflicting_registry_record_fails_closed():
    existing = _existing_record()
    existing["benchmark_report"] = {
        **_benchmark(),
        "dataset_version": "different",
    }
    service = ProductionEmbeddingModelRegistrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: FakeEngine(existing),
        preflight_factory=FakePreflight,
    )

    with pytest.raises(RuntimeError, match="conflicts"):
        service.register_approved(
            tenant_id="tenant-a",
            model=_model(),
            benchmark_report=_benchmark(),
            benchmark_dataset=production_benchmark_dataset(),
            approved_by="model-owner",
            punk_owned_target_confirmed=True,
            benchmark_accepted=True,
        )
