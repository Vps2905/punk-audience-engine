from pathlib import Path


def test_cohort_hardening_migration_exists():
    path = Path(
        "migrations/"
        "0001_audience_run_cohorts_hardening.sql"
    )

    sql = path.read_text().lower()

    assert "export_cohort_id" in sql
    assert "set not null" in sql
    assert "create unique index" in sql
    assert "run_id" in sql
    assert "audience_run_cohorts" in sql


def test_database_migration_runner_exists():
    path = Path(
        "scripts/apply_database_migrations.py"
    )

    source = path.read_text()

    assert "audience_schema_migrations" in source
    assert "checksum" in source
    assert "apply_migrations" in source
