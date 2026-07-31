from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.services.production_feature_build_migration_service import (
    MIGRATION_VERSION,
    ProductionFeatureBuildMigrationService,
)


class FakeResult:
    def __init__(self, *, row=None):
        self._row = row

    def mappings(self):
        return self

    def first(self):
        return self._row

    def one(self):
        return self._row


class FakeConnection:
    def __init__(self, existing_checksum=None):
        self.existing_checksum = existing_checksum
        self.execute_calls = []
        self.driver_sql_calls = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.execute_calls.append((sql, params or {}))
        if "SELECT checksum" in sql:
            return FakeResult(
                row=(
                    {"checksum": self.existing_checksum}
                    if self.existing_checksum
                    else None
                )
            )
        if "model_registry_present" in sql:
            return FakeResult(
                row={
                    "model_registry_present": True,
                    "build_ledger_present": True,
                    "approved_model_immutability_present": True,
                    "feature_set_immutability_present": True,
                    "feature_vector_immutability_present": True,
                    "model_registry_rls_forced": True,
                    "build_ledger_rls_forced": True,
                }
            )
        return FakeResult()

    def exec_driver_sql(self, statement):
        self.driver_sql_calls.append(str(statement))
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

    def __init__(self, existing_checksum=None):
        self.connection = FakeConnection(existing_checksum)
        self.disposed = False

    def begin(self):
        return FakeContext(self.connection)

    def dispose(self):
        self.disposed = True


class FakePreflight:
    def __init__(self, result):
        self.result = result

    def run(self, **_kwargs):
        return dict(self.result)


def _environment():
    return {
        "ECHO_DATABASE_URL": "configured-source",
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": "configured-feature",
    }


def _ready_preflight():
    return {
        "status": "phase2_schema_ready",
        "same_database_as_source": False,
    }


def _checksum():
    return hashlib.sha256(
        (Path("migrations") / MIGRATION_VERSION).read_bytes()
    ).hexdigest()


def test_feature_build_migration_requires_explicit_confirmation():
    service = ProductionFeatureBuildMigrationService(
        environment=_environment()
    )

    with pytest.raises(RuntimeError, match="confirmation"):
        service.apply(punk_owned_target_confirmed=False)


def test_feature_build_migration_applies_only_0008_and_verifies():
    engine = FakeEngine()
    service = ProductionFeatureBuildMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=lambda: FakePreflight(_ready_preflight()),
    )

    result = service.apply(punk_owned_target_confirmed=True)

    migration_calls = [
        sql
        for sql in engine.connection.driver_sql_calls
        if "CREATE TABLE IF NOT EXISTS audience_embedding_models" in sql
    ]
    assert len(migration_calls) == 1
    assert "audience_feature_build_jobs" in migration_calls[0]
    assert "trg_approved_embedding_model_immutable" in migration_calls[0]
    assert result["status"] == "applied_and_verified"
    assert result["source_database_modified"] is False
    assert result["features_built"] is False
    assert result["activation_or_export_performed"] is False
    assert engine.disposed is True


def test_feature_build_migration_is_checksum_idempotent():
    engine = FakeEngine(existing_checksum=_checksum())
    service = ProductionFeatureBuildMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=lambda: FakePreflight(_ready_preflight()),
    )

    result = service.apply(punk_owned_target_confirmed=True)

    assert result["status"] == "already_applied_and_verified"
    assert not any(
        "CREATE TABLE IF NOT EXISTS audience_embedding_models" in sql
        for sql in engine.connection.driver_sql_calls
    )


def test_feature_build_migration_rejects_checksum_conflict():
    engine = FakeEngine(existing_checksum="wrong")
    service = ProductionFeatureBuildMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=lambda: FakePreflight(_ready_preflight()),
    )

    with pytest.raises(RuntimeError, match="checksum"):
        service.apply(punk_owned_target_confirmed=True)


def test_feature_build_migration_rejects_source_database_target():
    service = ProductionFeatureBuildMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: pytest.fail(
            "database must not be opened"
        ),
        preflight_factory=lambda: FakePreflight(
            {
                "status": "phase2_schema_ready",
                "same_database_as_source": True,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="separate"):
        service.apply(punk_owned_target_confirmed=True)
