from pathlib import Path


def test_phase2_migration_has_real_pgvector_hybrid_and_tenant_controls():
    sql = Path(
        "migrations/0004_versioned_audience_features_pgvector.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "embedding vector(384) NOT NULL" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "USING GIN (search_document)" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "current_setting(" in sql
    assert "eligible_for_activation = FALSE" in sql
    assert "freshness_status = 'fresh'" in sql


def test_historical_index_script_is_in_production_container():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "index_historical_audience_features.py" in dockerfile
    assert "benchmark_pgvector_recall.py" in dockerfile
    assert "preflight_phase2_database.py" in dockerfile
    assert "inventory_historical_vector_snapshots.py" in dockerfile
    assert "apply_phase2_feature_migration.py" in dockerfile
    assert "provision_phase2_feature_roles.py" in dockerfile
    assert "apply_phase2_proposal_ledger_migration.py" in dockerfile
    assert "provision_phase2_proposal_runtime_role.py" in dockerfile


def test_phase2_runbook_requires_read_only_preflight_before_migration():
    runbook = Path(
        "docs/phase2_historical_feature_retrieval_runbook.md"
    ).read_text(encoding="utf-8")

    assert "preflight_phase2_database.py" in runbook
    assert "--confirm-punk-owned-target" in runbook
    assert "does not apply a" in runbook


def test_phase2_runbook_requires_inventory_and_exact_snapshot_selection():
    runbook = Path(
        "docs/phase2_historical_feature_retrieval_runbook.md"
    ).read_text(encoding="utf-8")

    assert "inventory_historical_vector_snapshots.py" in runbook
    assert "largest_compatible_snapshot_job_id" in runbook
    assert "--job-id" in runbook
    assert "automatic snapshot selection" in runbook.lower()


def test_phase2_runner_is_restricted_to_migration_0004():
    runner = Path(
        "app/services/phase2_feature_migration_service.py"
    ).read_text(encoding="utf-8")

    assert "0004_versioned_audience_features_pgvector.sql" in runner
    assert "glob(" not in runner
    assert "same_database_as_source" in runner
    assert "pg_advisory_xact_lock" in runner
    assert "migration_checksum_sha256" in runner


def test_phase2_runbook_requires_separate_database_roles():
    runbook = Path(
        "docs/phase2_historical_feature_retrieval_runbook.md"
    ).read_text(encoding="utf-8")

    assert "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL" in runbook
    assert "AUDIENCE_FEATURE_WRITER_DATABASE_URL" in runbook
    assert "provision_phase2_feature_roles.py" in runbook
    assert "NOBYPASSRLS" in runbook
