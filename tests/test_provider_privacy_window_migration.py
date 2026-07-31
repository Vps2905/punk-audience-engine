from pathlib import Path


MIGRATION = Path("migrations/0009_provider_privacy_windows_and_rights.sql")


def test_privacy_window_migration_has_production_closure_guards():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "provider_privacy_windows" in sql
    assert "provider_privacy_window_partitions" in sql
    assert "provider_canonical_partitions" in sql
    assert "provider_data_rights_requests" in sql
    assert "provider_scale_acceptance_runs" in sql
    assert "charged_epsilon" in sql
    assert "force row level security" in sql
    assert "current_setting('app.tenant_id', true)" in sql
    assert "unique (tenant_id, report_fingerprint)" in sql
