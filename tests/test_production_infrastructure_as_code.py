import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_infrastructure_contracts import (
    ProductionInfrastructureAssessmentRequest,
)
from app.services.production_infrastructure_governance_service import (
    REQUIRED_INFRASTRUCTURE_CONTROLS,
    ProductionInfrastructureAssessmentService,
    ProductionInfrastructureChangeSetReviewService,
)
from scripts.production_runtime_entrypoint import (
    build_database_url,
    configured_environment,
)


TENANT = "tenant_a"
TEMPLATE = Path(
    "infrastructure/cloudformation/production_runtime_plane.yaml"
)


def request():
    return ProductionInfrastructureAssessmentRequest(
        tenant_id=TENANT,
        assessment_id="infrastructure-assessment-1",
        environment_name="preproduction",
        execution_mode="preproduction",
    )


def passing_attestations():
    return {name: True for name in REQUIRED_INFRASTRUCTURE_CONTROLS}


def full_evidence():
    assessment = ProductionInfrastructureAssessmentService().assess(
        request=request(),
        attestations=passing_attestations(),
    )
    review = ProductionInfrastructureChangeSetReviewService().review(
        assessment_report=assessment,
        manual_review={
            "decision": "approved_for_preproduction_change_set",
            "review_reference": "iac-review-1",
        },
    )
    return assessment, review


def test_complete_infrastructure_assessment_passes_without_resource_values():
    assessment, _ = full_evidence()
    serialized = json.dumps(assessment).lower()
    assert assessment["assessment_status"] == "pass"
    assert assessment["failed_control_count"] == 0
    assert "arn:aws" not in serialized
    assert "database_url" not in serialized
    assert assessment["safety"]["cloud_resources_mutated"] is False
    assert assessment["safety"]["production_release_authorized"] is False


def test_missing_attestation_fails_closed():
    attestations = passing_attestations()
    attestations.pop("database_restore_plan_verified")
    assessment = ProductionInfrastructureAssessmentService().assess(
        request=request(), attestations=attestations
    )
    assert assessment["assessment_status"] == "fail_closed"
    assert assessment["failed_control_codes"] == [
        "database_restore_plan_verified"
    ]


def test_unknown_attestation_is_rejected_instead_of_ignored():
    attestations = passing_attestations()
    attestations["pretend_control"] = True
    with pytest.raises(ValueError, match="Unknown infrastructure"):
        ProductionInfrastructureAssessmentService().assess(
            request=request(), attestations=attestations
        )


def test_failed_infrastructure_controls_cannot_be_approved():
    assessment = ProductionInfrastructureAssessmentService().assess(
        request=request(), attestations={}
    )
    with pytest.raises(ValueError, match="cannot be approved"):
        ProductionInfrastructureChangeSetReviewService().review(
            assessment_report=assessment,
            manual_review={
                "decision": "approved_for_preproduction_change_set",
                "review_reference": "invalid-approval",
            },
        )


def test_preproduction_review_never_executes_or_authorizes_traffic():
    _, review = full_evidence()
    manual = review["manual_review"]
    assert manual["change_set_execution_authorized"] is False
    assert manual["production_traffic_authorized"] is False
    assert manual["production_release_authorized"] is False
    assert review["safety"]["automatic_deployment_performed"] is False


def test_infrastructure_evidence_is_tamper_evident():
    assessment, review = full_evidence()
    changed = deepcopy(assessment)
    changed["controls"][0]["passed"] = False
    with pytest.raises(ValueError, match="inconsistent|fingerprint"):
        ProductionInfrastructureAssessmentService().validate_report(changed)
    unsafe = deepcopy(review)
    unsafe["manual_review"]["change_set_execution_authorized"] = True
    with pytest.raises(ValueError, match="cannot set"):
        ProductionInfrastructureChangeSetReviewService().validate_report(unsafe)


def test_infrastructure_status_accepts_matching_safe_evidence(tmp_path):
    from app.services.production_infrastructure_status_service import (
        ProductionInfrastructureStatusService,
    )

    assessment, review = full_evidence()
    environment = {"INFRASTRUCTURE_VALIDATION_ENABLED": "true"}
    for key, value in {
        "INFRASTRUCTURE_ASSESSMENT_EVIDENCE_PATH": assessment,
        "INFRASTRUCTURE_REVIEW_EVIDENCE_PATH": review,
    }.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionInfrastructureStatusService(
        environment=environment
    ).status()
    assert status["status"] == "infrastructure_engineering_evidence_ready"
    assert status["infrastructure_engineering_evidence_ready"] is True
    assert status["live_production_certified"] is False
    assert all(status["components"].values())


@pytest.mark.parametrize(
    "flag",
    ["INFRASTRUCTURE_APPLY_ENABLED", "INFRASTRUCTURE_PRODUCTION_TRAFFIC_ENABLED"],
)
def test_infrastructure_status_blocks_mutation_and_traffic_flags(flag):
    from app.services.production_infrastructure_status_service import (
        ProductionInfrastructureStatusService,
    )

    status = ProductionInfrastructureStatusService(
        environment={flag: "true"}
    ).status()
    assert status["status"] == (
        "unsafe_configuration_infrastructure_mutation_blocked"
    )
    assert status["live_production_certified"] is False


def test_runtime_template_has_two_az_network_isolation():
    template = TEMPLATE.read_text(encoding="utf-8")
    for fragment in (
        "PublicSubnetA:",
        "PublicSubnetB:",
        "PrivateAppSubnetA:",
        "PrivateAppSubnetB:",
        "PrivateDbSubnetA:",
        "PrivateDbSubnetB:",
        "NatGatewayA:",
        "NatGatewayB:",
        "S3GatewayEndpoint:",
        "MapPublicIpOnLaunch: false",
        "AssignPublicIp: DISABLED",
    ):
        assert fragment in template
    assert "AssignPublicIp: ENABLED" not in template


def test_runtime_template_hardens_tls_and_ecs_tasks():
    template = TEMPLATE.read_text(encoding="utf-8")
    for fragment in (
        "AllowedPattern: \"^.+@sha256:[0-9a-f]{64}$\"",
        "Protocol: HTTPS",
        "SslPolicy: ELBSecurityPolicy-TLS13-1-2-2021-06",
        "StatusCode: HTTP_301",
        "routing.http.drop_invalid_header_fields.enabled",
        "routing.http.desync_mitigation_mode",
        "User: \"10001:10001\"",
        "ReadonlyRootFilesystem: true",
        "Drop: [ALL]",
        "EnableExecuteCommand: false",
        "DeploymentCircuitBreaker:",
        "Rollback: true",
    ):
        assert fragment in template


def test_runtime_template_uses_secret_injection_and_least_privilege():
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "Secrets:" in template
    assert "secretsmanager:GetSecretValue" in template
    assert "RuntimeSecretArn}:AUDIENCE_API_KEY::" in template
    assert "RuntimeSecretArn}:PUNK_AI_TENANT_AUTH_SECRET::" in template
    assert "DatabaseSecretArn}:username::" in template
    assert "DatabaseSecretArn}:password::" in template
    assert "ApiTaskRole:" in template
    assert "ProviderWorkerTaskRole:" in template
    assert "Resource: \"*\"" not in template
    assert "PRODUCTION_ALLOWED_HOSTS" in template
    assert "PRODUCTION_CORS_ORIGINS" in template
    assert "INFRASTRUCTURE_APPLY_ENABLED" in template
    assert "Value: \"false\"" in template


def test_runtime_bootstrap_builds_tls_url_and_encodes_credentials():
    environment = {
        "DATABASE_HOST": "db.internal.example",
        "DATABASE_PORT": "5432",
        "DATABASE_NAME": "punk_audience",
        "DATABASE_USER": "runtime user",
        "DATABASE_PASSWORD": "p@ss/word?#",
    }
    value = build_database_url(environment)
    assert value == (
        "postgresql://runtime%20user:p%40ss%2Fword%3F%23@"
        "db.internal.example:5432/punk_audience?sslmode=require"
    )


@pytest.mark.parametrize(
    ("mode", "target"),
    [
        ("api", "ECHO_DATABASE_URL"),
        ("worker", "PROVIDER_INGESTION_DATABASE_URL"),
    ],
)
def test_runtime_bootstrap_removes_database_component_secrets(mode, target):
    environment = {
        "DATABASE_HOST": "db.internal.example",
        "DATABASE_NAME": "punk_audience",
        "DATABASE_USER": "runtime",
        "DATABASE_PASSWORD": "secret",
        "SAFE_SETTING": "preserved",
    }
    result = configured_environment(mode, environment)
    assert result[target].endswith("?sslmode=require")
    assert "DATABASE_USER" not in result
    assert "DATABASE_PASSWORD" not in result
    assert result["SAFE_SETTING"] == "preserved"


def test_runtime_bootstrap_rejects_invalid_database_coordinates():
    environment = {
        "DATABASE_HOST": "db.example/unsafe",
        "DATABASE_NAME": "punk-audience",
        "DATABASE_USER": "runtime",
        "DATABASE_PASSWORD": "secret",
    }
    with pytest.raises(RuntimeError, match="DATABASE_HOST"):
        build_database_url(environment)


def test_runtime_database_is_private_encrypted_multi_az_and_recoverable():
    template = TEMPLATE.read_text(encoding="utf-8")
    for fragment in (
        "StorageEncrypted: true",
        "MultiAZ: true",
        "PubliclyAccessible: false",
        "DeletionProtection: true",
        "BackupRetentionPeriod: 35",
        "DeleteAutomatedBackups: false",
        "DeletionPolicy: Snapshot",
        "UpdateReplacePolicy: Snapshot",
        "rds.force_ssl: \"1\"",
        "EnablePerformanceInsights: true",
        "MonitoringInterval: 60",
    ):
        assert fragment in template
    assert "PubliclyAccessible: true" not in template


def test_runtime_template_has_autoscaling_logs_and_operator_alarms():
    template = TEMPLATE.read_text(encoding="utf-8")
    for fragment in (
        "AWS::ApplicationAutoScaling::ScalableTarget",
        "AWS::ApplicationAutoScaling::ScalingPolicy",
        "containerInsights",
        "RetentionInDays: 90",
        "UnhealthyTargetAlarm:",
        "LoadBalancer5xxAlarm:",
        "DatabaseHighCpuAlarm:",
        "DatabaseFreeStorageAlarm:",
        "AlarmActions: [!Ref AlertTopicArn]",
        "TreatMissingData: breaching",
    ):
        assert fragment in template


def test_infrastructure_migration_is_immutable_tenant_scoped_and_safe():
    sql = Path(
        "migrations/0026_production_infrastructure_evidence.sql"
    ).read_text(encoding="utf-8")
    for fragment in (
        "secret_values_stored = FALSE",
        "resource_identifiers_returned = FALSE",
        "cloud_resources_mutated = FALSE",
        "change_set_executed = FALSE",
        "automatic_deployment_performed = FALSE",
        "production_traffic_enabled = FALSE",
        "production_release_authorized = FALSE",
        "manual_approval_required = TRUE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting(''app.tenant_id'', true)",
        "BEFORE UPDATE OR DELETE",
        "REVOKE ALL",
    ):
        assert fragment in sql


def test_infrastructure_router_and_fail_closed_defaults_are_registered():
    main = Path("app/main.py").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "audience_intelligence_infrastructure_status_router" in main
    assert "INFRASTRUCTURE_VALIDATION_ENABLED=false" in env
    assert "INFRASTRUCTURE_APPLY_ENABLED=false" in env
    assert "INFRASTRUCTURE_PRODUCTION_TRAFFIC_ENABLED=false" in env
