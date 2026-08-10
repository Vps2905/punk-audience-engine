import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_infrastructure_contracts import (
    ProductionInfrastructureAssessmentRequest,
)
from app.models.production_preproduction_deployment_contracts import (
    PreproductionDeploymentCertificationRequest,
)
from app.services.production_infrastructure_governance_service import (
    REQUIRED_INFRASTRUCTURE_CONTROLS,
    ProductionInfrastructureAssessmentService,
    ProductionInfrastructureChangeSetReviewService,
)
from app.services.production_preproduction_deployment_certification_service import (
    OBSERVATION_FIELDS,
    ProductionPreproductionDeploymentCertificationService,
)
from app.services.preproduction_database_role_provisioning_service import (
    PreproductionDatabaseRoleProvisioningService,
)
from scripts import apply_database_migrations
from scripts.production_runtime_entrypoint import configured_environment


TENANT = "tenant_a"
IMAGE_DIGEST = "sha256:" + "a" * 64
REVIEW_FINGERPRINT = "b" * 64
MIGRATION_HEAD = "0027_preproduction_deployment_certification.sql"
TEMPLATE = Path("infrastructure/cloudformation/production_runtime_plane.yaml")


def request(
    *, review_fingerprint: str = REVIEW_FINGERPRINT,
) -> PreproductionDeploymentCertificationRequest:
    return PreproductionDeploymentCertificationRequest(
        tenant_id=TENANT,
        certification_id="preproduction-certification-1",
        environment_name="preproduction",
        infrastructure_review_fingerprint=review_fingerprint,
        candidate_image_digest=IMAGE_DIGEST,
        expected_migration_head=MIGRATION_HEAD,
    )


def passing_observations(
    *, review_fingerprint: str = REVIEW_FINGERPRINT,
) -> dict:
    return {
        "stack_status": "UPDATE_COMPLETE",
        "review_fingerprint": review_fingerprint,
        "running_image_digest": IMAGE_DIGEST,
        "api_desired_count": 2,
        "api_running_count": 2,
        "api_healthy_target_count": 2,
        "api_availability_zone_count": 2,
        "worker_desired_count": 2,
        "worker_running_count": 2,
        "worker_availability_zone_count": 2,
        "database_storage_encrypted": True,
        "database_multi_az": True,
        "database_publicly_accessible": False,
        "database_deletion_protection": True,
        "database_tls_required": True,
        "database_backup_retention_days": 35,
        "applied_migration_head": MIGRATION_HEAD,
        "pending_migration_count": 0,
        "migration_checksum_mismatch_count": 0,
        "runtime_database_roles_separated": True,
        "database_admin_credentials_in_runtime": False,
        "secret_injection_verified": True,
        "plaintext_secret_count": 0,
        "alarm_route_verified": True,
        "alarm_count": 6,
        "alarms_in_alarm_state_count": 0,
        "deployment_rollback_enabled": True,
        "production_traffic_enabled": False,
        "downstream_export_enabled": False,
        "live_provider_data_used": False,
    }


def infrastructure_evidence():
    assessment_request = ProductionInfrastructureAssessmentRequest(
        tenant_id=TENANT,
        assessment_id="infrastructure-assessment-1",
        environment_name="preproduction",
        execution_mode="preproduction",
    )
    assessment = ProductionInfrastructureAssessmentService().assess(
        request=assessment_request,
        attestations={name: True for name in REQUIRED_INFRASTRUCTURE_CONTROLS},
    )
    review = ProductionInfrastructureChangeSetReviewService().review(
        assessment_report=assessment,
        manual_review={
            "decision": "approved_for_preproduction_change_set",
            "review_reference": "iac-review-1",
        },
    )
    return assessment, review


def test_measured_preproduction_deployment_passes_without_fresh_data():
    report = ProductionPreproductionDeploymentCertificationService().certify(
        request=request(),
        observations=passing_observations(),
    )
    assert report["status"] == "preproduction_deployment_certified"
    assert report["preproduction_deployment_certified"] is True
    assert report["failed_control_count"] == 0
    assert report["fresh_provider_data_required"] is False
    assert report["live_production_certified"] is False
    assert report["safety"]["data_rows_read"] is False
    assert report["safety"]["production_traffic_enabled"] is False
    assert report["safety"]["downstream_export_performed"] is False


def test_observations_are_reduced_to_controls_without_resource_values():
    report = ProductionPreproductionDeploymentCertificationService().certify(
        request=request(), observations=passing_observations()
    )
    serialized = json.dumps(report)
    assert "stack_name" not in serialized
    assert "arn:aws" not in serialized
    assert "database_endpoint" not in serialized
    assert "secret_name" not in serialized
    assert set(passing_observations()) == set(OBSERVATION_FIELDS)


def test_missing_or_unknown_observations_are_rejected():
    missing = passing_observations()
    missing.pop("database_tls_required")
    with pytest.raises(ValueError, match="Missing preproduction"):
        ProductionPreproductionDeploymentCertificationService().certify(
            request=request(), observations=missing
        )
    unknown = passing_observations()
    unknown["pretend_pass"] = True
    with pytest.raises(ValueError, match="Unknown preproduction"):
        ProductionPreproductionDeploymentCertificationService().certify(
            request=request(), observations=unknown
        )


@pytest.mark.parametrize(
    ("field", "unsafe_value", "failed_control"),
    [
        ("running_image_digest", "sha256:" + "c" * 64, "immutable_candidate_image_running"),
        ("api_running_count", 1, "api_service_healthy_multi_az"),
        ("worker_availability_zone_count", 1, "provider_worker_healthy_multi_az"),
        ("database_publicly_accessible", True, "database_private_encrypted_multi_az"),
        ("pending_migration_count", 1, "database_migrations_complete_and_immutable"),
        ("database_admin_credentials_in_runtime", True, "runtime_database_roles_separated"),
        ("plaintext_secret_count", 1, "secret_injection_without_plaintext"),
        ("alarms_in_alarm_state_count", 1, "alarm_routes_and_runtime_alarms_verified"),
        ("deployment_rollback_enabled", False, "deployment_rollback_enabled"),
        ("production_traffic_enabled", True, "production_traffic_disabled"),
        ("downstream_export_enabled", True, "downstream_export_disabled"),
        ("live_provider_data_used", True, "no_live_provider_data_required"),
    ],
)
def test_unsafe_measured_condition_fails_closed(
    field, unsafe_value, failed_control
):
    observations = passing_observations()
    observations[field] = unsafe_value
    report = ProductionPreproductionDeploymentCertificationService().certify(
        request=request(), observations=observations
    )
    assert report["status"] == (
        "preproduction_deployment_certification_failed_closed"
    )
    assert report["preproduction_deployment_certified"] is False
    assert failed_control in report["failed_control_codes"]
    assert report["live_production_certified"] is False


def test_production_environment_cannot_be_certified_by_this_gate():
    with pytest.raises(ValueError, match="isolated preproduction"):
        PreproductionDeploymentCertificationRequest(
            tenant_id=TENANT,
            certification_id="invalid-production-certification",
            environment_name="production",
            infrastructure_review_fingerprint=REVIEW_FINGERPRINT,
            candidate_image_digest=IMAGE_DIGEST,
            expected_migration_head=MIGRATION_HEAD,
        )


def test_certification_report_is_tamper_evident():
    service = ProductionPreproductionDeploymentCertificationService()
    report = service.certify(
        request=request(), observations=passing_observations()
    )
    changed = deepcopy(report)
    changed["controls"][0]["passed"] = False
    with pytest.raises(ValueError, match="inconsistent|fingerprint"):
        service.validate_report(changed)
    unsafe = deepcopy(report)
    unsafe["safety"]["production_traffic_enabled"] = True
    with pytest.raises(ValueError, match="Unsafe"):
        service.validate_report(unsafe)


def test_infrastructure_status_accepts_matching_deployment_evidence(tmp_path):
    from app.services.production_infrastructure_status_service import (
        ProductionInfrastructureStatusService,
    )

    assessment, review = infrastructure_evidence()
    review_fingerprint = review["infrastructure_review_fingerprint"]
    deployment = ProductionPreproductionDeploymentCertificationService().certify(
        request=request(review_fingerprint=review_fingerprint),
        observations=passing_observations(
            review_fingerprint=review_fingerprint
        ),
    )
    environment = {"INFRASTRUCTURE_VALIDATION_ENABLED": "true"}
    for key, value in {
        "INFRASTRUCTURE_ASSESSMENT_EVIDENCE_PATH": assessment,
        "INFRASTRUCTURE_REVIEW_EVIDENCE_PATH": review,
        "PREPRODUCTION_DEPLOYMENT_CERTIFICATION_PATH": deployment,
    }.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionInfrastructureStatusService(
        environment=environment
    ).status()
    assert status["status"] == "preproduction_deployment_certified"
    assert status["preproduction_deployment_certified"] is True
    assert status["live_production_certified"] is False


def test_configured_invalid_deployment_evidence_fails_status_closed(tmp_path):
    from app.services.production_infrastructure_status_service import (
        ProductionInfrastructureStatusService,
    )

    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")
    status = ProductionInfrastructureStatusService(
        environment={
            "PREPRODUCTION_DEPLOYMENT_CERTIFICATION_PATH": str(path)
        }
    ).status()
    assert status["status"] == (
        "preproduction_deployment_certification_failed_closed"
    )
    assert status["preproduction_deployment_certified"] is False


def test_runtime_template_has_foundation_migration_runtime_sequence():
    template = TEMPLATE.read_text(encoding="utf-8")
    for fragment in (
        "DeploymentPhase:",
        "AllowedValues: [foundation, runtime]",
        "RuntimeEnabled: !Equals [!Ref DeploymentPhase, runtime]",
        "DesiredCount: !If [RuntimeEnabled, !Ref ApiDesiredCount, 0]",
        "DesiredCount: !If [RuntimeEnabled, !Ref WorkerDesiredCount, 0]",
        "MigrationTaskDefinition:",
        "scripts/production_runtime_entrypoint.py, migration",
        "MigrationTaskDefinitionArn:",
        "PREPRODUCTION_FULL_MIGRATION_ENABLED",
        "PREPRODUCTION_DATABASE_BOOTSTRAP_CONFIRMED",
        "PREPRODUCTION_TENANT_ID",
    ):
        assert fragment in template


def test_runtime_template_separates_admin_api_and_worker_database_secrets():
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "RuntimeDatabaseSecretArn:" in template
    assert "RuntimeDatabaseSecretArn}:api_username::" in template
    assert "RuntimeDatabaseSecretArn}:api_password::" in template
    assert "RuntimeDatabaseSecretArn}:worker_username::" in template
    assert "RuntimeDatabaseSecretArn}:worker_password::" in template
    api_task = template.split("  ApiTaskDefinition:", 1)[1].split(
        "  WorkerTaskDefinition:", 1
    )[0]
    worker_task = template.split("  WorkerTaskDefinition:", 1)[1].split(
        "  MigrationTaskDefinition:", 1
    )[0]
    assert "${DatabaseSecretArn}:username::" not in api_task
    assert "${DatabaseSecretArn}:username::" not in worker_task


def test_migration_task_can_bootstrap_separated_runtime_roles():
    template = TEMPLATE.read_text(encoding="utf-8")
    migration_task = template.split("  MigrationTaskDefinition:", 1)[1].split(
        "  AlbLogBucket:", 1
    )[0]
    for fragment in (
        "${DatabaseSecretArn}:username::",
        "${RuntimeDatabaseSecretArn}:api_username::",
        "${RuntimeDatabaseSecretArn}:api_password::",
        "${RuntimeDatabaseSecretArn}:worker_username::",
        "${RuntimeDatabaseSecretArn}:worker_password::",
    ):
        assert fragment in migration_task
    assert "scripts/production_runtime_entrypoint.py, migration" in migration_task


def test_full_preproduction_migration_includes_operator_owned_assets(monkeypatch):
    default_names = {
        path.name for path in apply_database_migrations.migration_files()
    }
    full_names = {
        path.name
        for path in apply_database_migrations.migration_files(
            include_operator_only=True
        )
    }
    assert not (
        default_names & apply_database_migrations.OPERATOR_ONLY_MIGRATIONS
    )
    assert apply_database_migrations.OPERATOR_ONLY_MIGRATIONS <= full_names
    monkeypatch.setenv("PREPRODUCTION_FULL_MIGRATION_ENABLED", "true")
    monkeypatch.delenv(
        "PREPRODUCTION_DATABASE_BOOTSTRAP_CONFIRMED", raising=False
    )
    with pytest.raises(RuntimeError, match="explicit bootstrap confirmation"):
        apply_database_migrations.apply_migrations()


def test_role_provisioning_rejects_non_staging_or_unsafe_role_names():
    with pytest.raises(RuntimeError, match="APP_ENV"):
        PreproductionDatabaseRoleProvisioningService(
            environment={"APP_ENV": "production"}
        ).provision()
    environment = {
        "APP_ENV": "preproduction",
        "PREPRODUCTION_DEPLOYMENT_PHASE": "foundation",
        "PREPRODUCTION_DATABASE_BOOTSTRAP_CONFIRMED": "true",
        "PREPRODUCTION_TENANT_ID": TENANT,
        "API_DATABASE_USER": "unsafe role;drop",
        "API_DATABASE_PASSWORD": "secret",
        "WORKER_DATABASE_USER": "worker_runtime",
        "WORKER_DATABASE_PASSWORD": "secret",
        "DATABASE_URL": "postgresql://admin:secret@db/punk",
    }
    with pytest.raises(ValueError, match="role name is invalid"):
        PreproductionDatabaseRoleProvisioningService(
            environment=environment
        ).provision()


def test_migration_runtime_mode_builds_tls_url_and_removes_credentials():
    result = configured_environment(
        "migration",
        {
            "DATABASE_HOST": "db.internal.example",
            "DATABASE_NAME": "punk_audience",
            "DATABASE_USER": "migration_admin",
            "DATABASE_PASSWORD": "secret",
        },
    )
    assert result["DATABASE_URL"].endswith("?sslmode=require")
    assert "DATABASE_USER" not in result
    assert "DATABASE_PASSWORD" not in result


def test_preproduction_deployment_migration_is_immutable_and_tenant_scoped():
    sql = Path(
        "migrations/0027_preproduction_deployment_certification.sql"
    ).read_text(encoding="utf-8")
    for fragment in (
        "inspection_read_only = TRUE",
        "secret_values_read = FALSE",
        "resource_identifiers_returned = FALSE",
        "live_provider_data_required = FALSE",
        "production_traffic_enabled = FALSE",
        "downstream_export_performed = FALSE",
        "live_production_certified = FALSE",
        "manual_production_approval_required = TRUE",
        "BEFORE UPDATE OR DELETE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting('app.tenant_id', true)",
        "REVOKE ALL",
    ):
        assert fragment in sql
