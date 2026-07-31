from pathlib import Path

from scripts import apply_database_migrations

MIGRATION = Path(
    "migrations/0005_punk_ai_audience_proposal_ledger.sql"
)


def test_proposal_ledger_migration_is_immutable_and_tenant_scoped():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "punk_ai_audience_proposals" in sql
    assert "UNIQUE (" in sql
    assert "idempotency_key" in sql
    assert "request_fingerprint" in sql
    assert "response_document JSONB" in sql
    assert "downstream_export_enabled = FALSE" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "current_setting(" in sql
    assert "'app.tenant_id'" in sql
    assert "BEFORE UPDATE OR DELETE" in sql
    assert "raw_maid" not in sql.lower()
    assert "embedding vector" not in sql.lower()


def test_generic_runner_excludes_operator_only_phase2_migrations(
    monkeypatch,
    tmp_path,
):
    for name in (
        "0003_provider_ingestion_gateway.sql",
        "0004_versioned_audience_features_pgvector.sql",
        "0005_punk_ai_audience_proposal_ledger.sql",
    ):
        (tmp_path / name).write_text("-- test\n", encoding="utf-8")
    monkeypatch.setattr(
        apply_database_migrations,
        "MIGRATIONS_DIR",
        tmp_path,
    )

    assert [
        path.name
        for path in apply_database_migrations.migration_files()
    ] == []
