from pathlib import Path


def test_module1_postgres_schema_contains_required_tables_and_constraints():
    schema_path = Path("docs/module1_postgres_schema.sql")

    assert schema_path.exists()

    sql = schema_path.read_text().lower()

    required_tables = [
        "audience_ingestion_jobs",
        "audience_lineage_events",
        "audience_privacy_budget_ledger",
        "audience_run_history",
        "audience_run_cohorts",
        "audience_run_artifacts",
        "audience_run_warnings",
        "audience_run_events",
        "audience_run_approvals",
    ]

    for table in required_tables:
        assert f"create table if not exists {table}" in sql

    required_terms = [
        "jsonb",
        "timestamptz",
        "check (epsilon > 0)",
        "check (sensitivity > 0)",
        "check (decision in",
        "downstream_export_enabled boolean",
        "idx_audience_privacy_budget_scope",
        "idx_audience_lineage_job",
        "idx_audience_ingestion_jobs_job",
    ]

    for term in required_terms:
        assert term in sql


def test_module1_postgres_schema_documents_no_alembic_yet():
    sql = Path("docs/module1_postgres_schema.sql").read_text().lower()

    assert "currently does not include alembic" in sql
    assert "production schema reference" in sql
