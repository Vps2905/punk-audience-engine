from pathlib import Path


MIGRATION = Path(
    "migrations/0008_production_feature_build_registry.sql"
)


def test_feature_build_registry_has_model_approval_and_durable_jobs():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "audience_embedding_models" in sql
    assert "model_fingerprint char(64) not null" in sql
    assert "model_revision text not null" in sql
    assert "benchmark_status" in sql
    assert "approved_by is not null" in sql
    assert "audience_feature_build_jobs" in sql
    assert "request_fingerprint char(64) not null" in sql
    assert "canonical_source_fingerprint char(64) not null" in sql
    assert "result_receipt jsonb" in sql
    assert "unique (\n        tenant_id,\n        request_fingerprint" in sql
    assert "prevent_feature_build_identity_change" in sql
    assert "prevent_approved_embedding_model_mutation" in sql
    assert "trg_approved_embedding_model_immutable" in sql
    assert "prevent_feature_artifact_mutation" in sql
    assert "trg_audience_feature_sets_immutable" in sql
    assert "trg_audience_feature_vectors_immutable" in sql
    assert "enable row level security" in sql
    assert "force row level security" in sql
    assert "current_setting(" in sql


def test_general_migration_runner_excludes_operator_feature_registry():
    source = Path(
        "scripts/apply_database_migrations.py"
    ).read_text(encoding="utf-8")

    assert "0008_production_feature_build_registry.sql" in source
    assert "OPERATOR_ONLY_MIGRATIONS" in source


def test_pgvector_publication_uses_insert_only_conflict_handling():
    source = Path(
        "app/services/pgvector_audience_feature_store_service.py"
    ).read_text(encoding="utf-8")

    assert source.count("DO NOTHING") >= 2
    assert "DO UPDATE SET" not in source
