from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from app.services.phase2_feature_migration_service import (
    PHASE2_MIGRATION_VERSION,
    Phase2FeatureMigrationService,
)


class FakeMappings:
    def __init__(self, row=None):
        self._row = row

    def mappings(self):
        return self

    def first(self):
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
            row = (
                {"checksum": self.existing_checksum}
                if self.existing_checksum
                else None
            )
            return FakeMappings(row)
        return FakeMappings()

    def exec_driver_sql(self, statement):
        self.driver_sql_calls.append(str(statement))
        return FakeMappings()


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection
        self.exited_with = None

    def __enter__(self):
        return self.connection

    def __exit__(self, exception_type, *_args):
        self.exited_with = exception_type
        return False


class FakeDialect:
    name = "postgresql"


class FakeEngine:
    dialect = FakeDialect()

    def __init__(self, existing_checksum=None):
        self.connection = FakeConnection(existing_checksum)
        self.begin_context = FakeBegin(self.connection)
        self.disposed = False

    def begin(self):
        return self.begin_context

    def dispose(self):
        self.disposed = True


class FakePreflight:
    def __init__(self, result):
        self.result = result

    def run(self, **_kwargs):
        return dict(self.result)


def _preflight_factory(
    *results: dict,
) -> Callable[[], FakePreflight]:
    pending = list(results)

    def factory():
        return FakePreflight(pending.pop(0))

    return factory


def _environment():
    return {
        "ECHO_DATABASE_URL": "configured-source",
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": (
            "configured-feature-target"
        ),
    }


def _migration_checksum() -> str:
    migration = Path("migrations") / PHASE2_MIGRATION_VERSION
    return hashlib.sha256(migration.read_bytes()).hexdigest()


def _ready_before():
    return {
        "status": "ready_for_approved_migration",
        "same_database_as_source": False,
    }


def _ready_after():
    return {
        "status": "phase2_schema_ready",
        "same_database_as_source": False,
    }


def test_migration_requires_explicit_target_confirmation():
    service = Phase2FeatureMigrationService(environment=_environment())

    with pytest.raises(RuntimeError, match="confirmation"):
        service.apply(punk_owned_target_confirmed=False)


def test_migration_rejects_same_database_before_connecting():
    def forbidden_engine(*_args, **_kwargs):
        pytest.fail("migration engine must not connect")

    service = Phase2FeatureMigrationService(
        environment=_environment(),
        engine_factory=forbidden_engine,
        preflight_factory=_preflight_factory(
            {
                "status": "ready_for_approved_migration",
                "same_database_as_source": True,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="separate"):
        service.apply(punk_owned_target_confirmed=True)


def test_only_migration_0004_is_applied_and_verified():
    engine = FakeEngine()
    service = Phase2FeatureMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=_preflight_factory(
            _ready_before(),
            _ready_after(),
        ),
    )

    result = service.apply(punk_owned_target_confirmed=True)

    migration_calls = [
        sql
        for sql in engine.connection.driver_sql_calls
        if "CREATE EXTENSION IF NOT EXISTS vector" in sql
    ]
    assert len(migration_calls) == 1
    assert "audience_feature_sets" in migration_calls[0]
    assert "audience_feature_vectors" in migration_calls[0]
    assert any(
        "pg_advisory_xact_lock" in sql
        for sql, _ in engine.connection.execute_calls
    )
    assert result["status"] == "applied_and_verified"
    assert result["migration_version"] == PHASE2_MIGRATION_VERSION
    assert result["source_database_modified"] is False
    assert result["historical_snapshot_imported"] is False
    assert result["activation_or_export_performed"] is False
    assert engine.disposed is True


def test_same_checksum_is_idempotently_skipped_and_verified():
    engine = FakeEngine(existing_checksum=_migration_checksum())
    service = Phase2FeatureMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=_preflight_factory(
            _ready_after(),
            _ready_after(),
        ),
    )

    result = service.apply(punk_owned_target_confirmed=True)

    assert result["status"] == "already_applied_and_verified"
    assert not any(
        "CREATE EXTENSION IF NOT EXISTS vector" in sql
        for sql in engine.connection.driver_sql_calls
    )


def test_checksum_mismatch_fails_without_executing_migration():
    engine = FakeEngine(existing_checksum="different-checksum")
    service = Phase2FeatureMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=_preflight_factory(_ready_after()),
    )

    with pytest.raises(RuntimeError, match="checksum"):
        service.apply(punk_owned_target_confirmed=True)

    assert not any(
        "CREATE EXTENSION IF NOT EXISTS vector" in sql
        for sql in engine.connection.driver_sql_calls
    )


def test_unready_preflight_blocks_before_database_write():
    def forbidden_engine(*_args, **_kwargs):
        pytest.fail("migration engine must not connect")

    service = Phase2FeatureMigrationService(
        environment=_environment(),
        engine_factory=forbidden_engine,
        preflight_factory=_preflight_factory(
            {
                "status": "blocked",
                "same_database_as_source": False,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="not migration-ready"):
        service.apply(punk_owned_target_confirmed=True)
