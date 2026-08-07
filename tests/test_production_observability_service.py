import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_observability_contracts import (
    AggregateTelemetryCoverage,
    ProductionObservabilitySnapshotRequest,
    ServiceLevelObservation,
)
from app.services.production_observability_service import (
    ProductionIncidentReviewPlanningService,
    ProductionObservabilitySnapshotService,
)


TENANT = "tenant_a"
SOURCE = "a" * 64


def coverage(
    component,
    signal,
    *,
    emitted=100,
    accepted=100,
    correlation=0.99,
    export_success=0.99,
):
    return AggregateTelemetryCoverage(
        component_name=component,
        signal_type=signal,
        emitted_event_count=emitted,
        accepted_event_count=accepted,
        rejected_sensitive_attribute_count=2,
        correlation_coverage_rate=correlation,
        export_success_rate=export_success,
        minimum_correlation_coverage_rate=0.95,
        minimum_export_success_rate=0.95,
        source_evidence_fingerprint=SOURCE,
    )


def slo(
    name,
    value,
    *,
    direction="at_least",
    target=0.99,
    warning=0.97,
    critical=0.90,
    sample_count=1000,
    minimum_sample_count=100,
):
    return ServiceLevelObservation(
        service_name="audience_api",
        sli_name=name,
        objective_direction=direction,
        observed_value=value,
        target_value=target,
        warning_boundary=warning,
        critical_boundary=critical,
        sample_count=sample_count,
        minimum_sample_count=minimum_sample_count,
        window_seconds=3600,
        source_evidence_fingerprint=SOURCE,
    )


def full_evidence():
    snapshot = ProductionObservabilitySnapshotService().build(
        request=ProductionObservabilitySnapshotRequest(
            tenant_id=TENANT,
            snapshot_id="observation-1",
            execution_mode="production",
        ),
        telemetry_coverage=[
            coverage("audience_api", "metrics"),
            coverage(
                "audience_worker", "traces",
                correlation=0.80, export_success=0.90,
            ),
        ],
        service_levels=[
            slo("availability_rate", 0.995),
            slo(
                "latency_p95_ms", 650,
                direction="at_most", target=500, warning=600, critical=1000,
            ),
            slo("request_success_rate", 0.99, sample_count=20),
        ],
    )
    incidents = ProductionIncidentReviewPlanningService().plan(
        observability_snapshot=snapshot
    )
    return snapshot, incidents


def test_snapshot_evaluates_telemetry_coverage_and_slo_boundaries():
    snapshot, _ = full_evidence()
    coverage_by_key = {
        (value["component_name"], value["signal_type"]): value
        for value in snapshot["telemetry_coverage"]
    }
    slo_by_name = {
        value["sli_name"]: value for value in snapshot["service_levels"]
    }
    assert coverage_by_key[("audience_api", "metrics")][
        "coverage_status"
    ] == "healthy"
    assert coverage_by_key[("audience_worker", "traces")][
        "coverage_status"
    ] == "warning"
    assert slo_by_name["availability_rate"]["objective_status"] == "met"
    assert slo_by_name["latency_p95_ms"]["objective_status"] == "warning"
    assert slo_by_name["request_success_rate"][
        "objective_status"
    ] == "insufficient_data"
    assert snapshot["overall_status"] == "warning"


def test_missing_telemetry_signal_is_critical():
    snapshot = ProductionObservabilitySnapshotService().build(
        request=ProductionObservabilitySnapshotRequest(
            tenant_id=TENANT,
            snapshot_id="missing-signal",
            execution_mode="production",
        ),
        telemetry_coverage=[
            coverage("worker", "logs", emitted=0, accepted=0),
        ],
        service_levels=[slo("availability", 1.0)],
    )
    assert snapshot["critical_count"] == 1
    assert snapshot["overall_status"] == "critical"


def test_slo_contract_rejects_invalid_boundary_order():
    with pytest.raises(ValueError, match="critical <= warning <= target"):
        slo("invalid", 0.9, target=0.9, warning=0.95, critical=0.8)
    with pytest.raises(ValueError, match="target <= warning <= critical"):
        slo(
            "invalid_latency", 100,
            direction="at_most", target=500, warning=400, critical=1000,
        )


def test_snapshot_contains_only_aggregate_safe_observability_evidence():
    snapshot, _ = full_evidence()
    serialized = json.dumps(snapshot).lower()
    assert "original_prompt" not in serialized
    assert "request_body" not in serialized
    assert snapshot["safety"]["aggregate_telemetry_only"] is True
    assert snapshot["safety"]["prompt_content_stored"] is False
    assert snapshot["safety"]["sensitive_attributes_exported"] is False


def test_incident_plan_is_review_only_and_never_dispatches_or_remediates():
    _, incidents = full_evidence()
    assert incidents["review_required_count"] == 3
    for action in incidents["actions"]:
        assert action["external_alert_dispatched"] is False
        assert action["automatic_remediation_performed"] is False
        assert action["incident_declared_automatically"] is False
        assert action["manual_approval_required"] is True
        assert action["production_routing_enabled"] is False


def test_tampering_and_false_incident_actions_fail_closed():
    snapshot, incidents = full_evidence()
    tampered = deepcopy(snapshot)
    tampered["service_levels"][0]["observed_value"] = 0.0
    with pytest.raises(ValueError, match="assessment|fingerprint"):
        ProductionObservabilitySnapshotService().validate_report(tampered)

    unsafe = deepcopy(incidents)
    unsafe["actions"][0]["external_alert_dispatched"] = True
    with pytest.raises(ValueError, match="cannot set"):
        ProductionIncidentReviewPlanningService().validate_report(unsafe)

    false_action = deepcopy(incidents)
    changed = next(
        value
        for value in false_action["actions"]
        if value["recommended_action"] != "continue_monitoring"
    )
    changed["recommended_action"] = "continue_monitoring"
    with pytest.raises(ValueError, match="inconsistent"):
        ProductionIncidentReviewPlanningService().validate_report(false_action)


def test_incident_plan_tenant_tampering_fails_fingerprint_validation():
    _, incidents = full_evidence()
    tampered = deepcopy(incidents)
    tampered["tenant_id"] = "another_tenant"
    with pytest.raises(ValueError, match="fingerprint"):
        ProductionIncidentReviewPlanningService().validate_report(tampered)


def test_observability_status_accepts_complete_safe_evidence_chain(tmp_path):
    from app.services.production_observability_status_service import (
        ProductionObservabilityStatusService,
    )

    snapshot, incidents = full_evidence()
    values = {
        "OBSERVABILITY_SNAPSHOT_EVIDENCE_PATH": snapshot,
        "OBSERVABILITY_INCIDENT_PLAN_EVIDENCE_PATH": incidents,
    }
    environment = {}
    for key, value in values.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionObservabilityStatusService(
        environment=environment
    ).status()
    assert status["status"] == "observability_engineering_evidence_ready"
    assert status["observability_engineering_evidence_ready"] is True
    assert status["live_production_certified"] is False
    assert all(status["components"].values())


def test_observability_status_blocks_automatic_remediation():
    from app.services.production_observability_status_service import (
        ProductionObservabilityStatusService,
    )

    status = ProductionObservabilityStatusService(environment={
        "OBSERVABILITY_AUTOMATIC_REMEDIATION_ENABLED": "true",
    }).status()
    assert status["status"] == (
        "unsafe_configuration_release_affecting_feature_blocked"
    )


def test_observability_router_and_safe_defaults_are_registered():
    main = Path("app/main.py").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "audience_intelligence_observability_status_router" in main
    assert "OBSERVABILITY_AUTOMATIC_REMEDIATION_ENABLED=false" in env
    assert "OBSERVABILITY_PRODUCTION_ROUTING_ENABLED=false" in env


def test_observability_migration_is_immutable_tenant_scoped_and_safe():
    sql = Path(
        "migrations/0024_production_observability_slo_governance.sql"
    ).read_text(encoding="utf-8")
    for fragment in (
        "aggregate_telemetry_only = TRUE",
        "prompt_content_stored = FALSE",
        "request_payload_stored = FALSE",
        "response_payload_stored = FALSE",
        "sensitive_attributes_exported = FALSE",
        "external_alert_dispatched = FALSE",
        "automatic_remediation_performed = FALSE",
        "incident_declared_automatically = FALSE",
        "production_routing_enabled = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting(''app.tenant_id'', true)",
        "prevent_production_observability_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql
