from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "migrations" / "0006_provider_distributed_scale_dispatch.sql"
)


def test_distributed_scale_migration_extends_ingestion_state_machine():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "provider_ingestion_objects_status_check" in sql
    assert "'dispatching'" in sql
    assert "'dispatched'" in sql
    assert "idx_provider_ingestion_dispatched" in sql
