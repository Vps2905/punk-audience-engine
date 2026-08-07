from pathlib import Path

from app.services.production_fresh_data_workflow_readiness_service import (
    ProductionFreshDataWorkflowReadinessService,
)
from app.services.production_fresh_data_workflow_status_service import (
    ProductionFreshDataWorkflowStatusService,
)


_REQUIRED_ENV = {
    "FRESH_DATA_WORKFLOW_ENABLED": "true",
    "PROVIDER_INGESTION_DATABASE_URL": "postgresql://provider",
    "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://reader",
    "AUDIENCE_FEATURE_WRITER_DATABASE_URL": "postgresql://writer",
}


def test_fresh_data_workflow_is_disabled_by_default():
    status = ProductionFreshDataWorkflowStatusService(
        environment={},
        readiness_probe=lambda: (_ for _ in ()).throw(
            AssertionError("disabled status must not probe databases")
        ),
    ).status()
    assert status["status"] == "fresh_data_workflow_disabled"
    assert status["enabled"] is False
    assert status["database_readiness"]["checked"] is False
    assert status["safety"]["approval_required"] is True
    assert status["safety"]["activation_or_export_performed"] is False
    assert status["safety"]["secret_values_returned"] is False


def test_enabled_workflow_requires_all_database_boundaries():
    status = ProductionFreshDataWorkflowStatusService(
        environment={"FRESH_DATA_WORKFLOW_ENABLED": "true"},
        readiness_probe=lambda: (_ for _ in ()).throw(
            AssertionError("incomplete config must not probe databases")
        ),
    ).status()
    assert status["status"] == "fresh_data_workflow_configuration_incomplete"


def test_runtime_status_does_not_require_migration_admin_credentials():
    probed = {"called": False}

    def probe():
        probed["called"] = True
        return {
            "ready": True,
            "read_only": True,
            "credentials_exposed": False,
            "components": {
                "provider_runtime_ready": True,
                "feature_reader_boundary_ready": True,
                "feature_writer_boundary_ready": True,
                "workflow_migration_ready": True,
            },
            "blockers": [],
        }

    status = ProductionFreshDataWorkflowStatusService(
        environment=_REQUIRED_ENV,
        readiness_probe=probe,
    ).status()

    assert probed["called"] is True
    assert status["status"] == "fresh_data_workflow_control_plane_ready"
    assert "feature_migration_database" not in status["configuration"]


def test_enabled_workflow_fails_closed_when_database_is_not_ready():
    status = ProductionFreshDataWorkflowStatusService(
        environment=_REQUIRED_ENV,
        readiness_probe=lambda: {
            "ready": False,
            "read_only": True,
            "credentials_exposed": False,
            "components": {"workflow_migration_ready": False},
            "blockers": ["workflow_migration_ready"],
        },
    ).status()
    assert status["status"] == "fresh_data_workflow_database_not_ready"
    assert "resolve database readiness blocker: workflow_migration_ready" in (
        status["remaining_external_gates"]
    )


def test_enabled_workflow_reports_control_plane_ready_only_after_live_probe():
    status = ProductionFreshDataWorkflowStatusService(
        environment=_REQUIRED_ENV,
        readiness_probe=lambda: {
            "ready": True,
            "read_only": True,
            "credentials_exposed": False,
            "components": {
                "provider_runtime_ready": True,
                "feature_reader_boundary_ready": True,
                "feature_writer_boundary_ready": True,
                "workflow_migration_ready": True,
            },
            "blockers": [],
        },
    ).status()
    assert status["status"] == "fresh_data_workflow_control_plane_ready"
    assert status["database_readiness"]["ready"] is True
    assert all(
        "migration 0014" not in gate
        and "role provisioning" not in gate
        for gate in status["remaining_external_gates"]
    )
    assert any(
        "trigger/queue worker" in gate
        for gate in status["remaining_external_gates"]
    )


def test_release_flags_fail_closed_without_database_probe():
    status = ProductionFreshDataWorkflowStatusService(
        environment={"AUDIENCE_DOWNSTREAM_EXPORT_ENABLED": "true"},
        readiness_probe=lambda: (_ for _ in ()).throw(
            AssertionError("unsafe release must fail before database probe")
        ),
    ).status()
    assert status["status"] == "unsafe_release_configuration_blocked"


def test_readiness_service_without_configuration_is_safe_and_read_only():
    report = ProductionFreshDataWorkflowReadinessService(
        environment={}
    ).inspect()
    assert report["ready"] is False
    assert report["read_only"] is True
    assert report["credentials_exposed"] is False
    assert report["blockers"] == [
        "feature_reader_boundary_ready",
        "feature_writer_boundary_ready",
        "provider_runtime_ready",
        "workflow_migration_ready",
    ]


def test_fresh_data_workflow_migration_is_durable_tenant_isolated_and_safe():
    sql = Path(
        "migrations/0014_production_fresh_data_workflows.sql"
    ).read_text(encoding="utf-8")
    for value in (
        "approval_required = TRUE",
        "activation_requested = FALSE",
        "export_requested = FALSE",
        "lookalike_generation_requested = FALSE",
        "lease_owner",
        "lease_expires_at",
        "awaiting_review",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "prevent_fresh_data_workflow_identity_change",
        "prevent_fresh_data_workflow_event_mutation",
        "REVOKE ALL ON audience_fresh_data_workflows FROM PUBLIC",
    ):
        assert value in sql


def test_router_and_safe_defaults_are_registered():
    main = Path("app/main.py").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "audience_intelligence_fresh_data_workflows_router" in main
    assert (
        "_include_audience_router("
        "audience_intelligence_fresh_data_workflows_router)"
        in main
    )
    assert "FRESH_DATA_WORKFLOW_ENABLED=false" in env
    assert "FRESH_DATA_WORKFLOW_AUTOMATIC_TRIGGER_ENABLED=false" in env
