from __future__ import annotations

from pathlib import Path

import pytest

from app.services.phase2_proposal_ledger_migration_service import (
    PROPOSAL_LEDGER_MIGRATION_VERSION,
    Phase2ProposalLedgerMigrationService,
)


class FakeResult:
    def __init__(self, *, row=None):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row

    def one(self):
        return self.row


class FakeTransaction:
    def rollback(self):
        return None


class FakeConnection:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def begin(self):
        return FakeTransaction()

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append((sql, params or {}))
        if "SELECT checksum" in sql:
            return FakeResult(row=None)
        if "proposal_table_present" in sql:
            return FakeResult(
                row={
                    "proposal_table_present": True,
                    "proposal_rls_forced": True,
                    "proposal_tenant_policy_present": True,
                    "proposal_immutability_trigger_present": True,
                    "proposal_lookup_index_present": True,
                }
            )
        return FakeResult()

    def exec_driver_sql(self, statement):
        self.calls.append((str(statement), {}))
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

    def __init__(self):
        self.connection = FakeConnection()
        self.disposed = False

    def begin(self):
        return FakeContext(self.connection)

    def connect(self):
        return self.connection

    def dispose(self):
        self.disposed = True


class ReadyPreflight:
    def run(self, **_kwargs):
        return {
            "status": "phase2_schema_ready",
            "same_database_as_source": False,
        }


def _environment():
    return {
        "ECHO_DATABASE_URL": "postgresql://source",
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": (
            "postgresql://migration-admin"
        ),
    }


def test_proposal_ledger_migration_requires_explicit_approval():
    service = Phase2ProposalLedgerMigrationService(
        environment=_environment()
    )

    with pytest.raises(RuntimeError, match="confirmation"):
        service.apply(punk_owned_target_confirmed=False)


def test_proposal_ledger_migration_applies_only_approved_file():
    engine = FakeEngine()
    service = Phase2ProposalLedgerMigrationService(
        environment=_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        preflight_factory=ReadyPreflight,
        migration_path=Path("migrations") / PROPOSAL_LEDGER_MIGRATION_VERSION,
    )

    result = service.apply(punk_owned_target_confirmed=True)

    assert result["status"] == "applied_and_verified"
    assert result["proposal_table_present"] is True
    assert result["proposal_rls_forced"] is True
    assert result["proposal_immutability_trigger_present"] is True
    assert result["source_database_modified"] is False
    assert result["feature_rows_modified"] is False
    assert result["activation_or_export_performed"] is False
    assert result["credentials_exposed"] is False
    assert "migration-admin" not in str(result)
    rendered_sql = "\n".join(
        sql for sql, _params in engine.connection.calls
    )
    assert "punk_ai_audience_proposals" in rendered_sql
    assert PROPOSAL_LEDGER_MIGRATION_VERSION in str(
        engine.connection.calls
    )
