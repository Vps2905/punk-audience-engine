from pathlib import Path

from app.services.production_module2_status_service import (
    ProductionModule2StatusService,
)


def test_status_is_fail_closed_and_never_returns_secret_paths():
    status = ProductionModule2StatusService(
        environment={
            "MODULE2_CERTIFICATION_REPORT_PATH": "/secret/path/report.json",
            "MODULE2_PRODUCTION_ROUTING_ENABLED": "true",
        }
    ).status()
    assert status["status"] == "unsafe_configuration_production_routing_blocked"
    assert status["production_certification_ready"] is False
    assert "/secret/path" not in str(status)
    assert status["safety"]["secret_values_returned"] is False


def test_module2_migration_has_atomic_active_index_and_privacy_safe_shadow_guards():
    sql = Path("migrations/0011_module2_certification_index_shadow.sql").read_text(
        encoding="utf-8"
    )
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_audience_retrieval_active_index" in sql
    assert "WHERE status = 'active'" in sql
    assert "raw_query_stored = FALSE" in sql
    assert "raw_identifiers_stored = FALSE" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "prevent_module2_index_identity_change" in sql
