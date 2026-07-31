from pathlib import Path


def test_local_pgvector_compose_is_isolated_pinned_and_loopback_only():
    compose = Path(
        "docker-compose.phase2-pgvector.yml"
    ).read_text(encoding="utf-8")

    assert "pgvector/pgvector:0.8.5-pg18-trixie" in compose
    assert "127.0.0.1:${PHASE2_PGVECTOR_PORT:-55432}:5432" in compose
    assert "PHASE2_PGVECTOR_PASSWORD:?" in compose
    assert "phase2-pgvector-data:/var/lib/postgresql" in compose
    assert "ECHO_DATABASE_URL" not in compose
    assert "audience-engine" not in compose
    assert "provider-ingestion-worker" not in compose
    assert 'restart: "no"' in compose


def test_environment_template_contains_no_local_pgvector_password():
    template = Path(".env.example").read_text(encoding="utf-8")

    assert "PHASE2_PGVECTOR_PASSWORD=\n" in template
    assert "PHASE2_PGVECTOR_PORT=55432" in template


def test_historical_index_requires_explicit_target_and_snapshot():
    script = Path(
        "scripts/index_historical_audience_features.py"
    ).read_text(encoding="utf-8")

    assert '"--job-id",' in script
    assert "required=True" in script
    assert "AUDIENCE_FEATURE_WRITER_DATABASE_URL" in script
    assert "or source_database_url" not in script
    assert "Phase2DatabasePreflightService" in script
    assert '"phase2_schema_ready"' in script
    assert 'preflight.get("same_database_as_source")' in script


def test_generic_migration_runner_cannot_select_phase2_feature_database():
    script = Path(
        "scripts/apply_database_migrations.py"
    ).read_text(encoding="utf-8")

    assert "AUDIENCE_FEATURE_DATABASE_URL" not in script
