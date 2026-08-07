import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_quality_monitoring_contracts import (
    AggregateQualityMetric,
    ProductionQualityDriftRequest,
    ProductionQualitySnapshotRequest,
)
from app.services.production_quality_monitoring_service import (
    ProductionQualityAlertPlanningService,
    ProductionQualityDriftService,
    ProductionQualitySnapshotService,
)


TENANT = "tenant_a"


def metric(
    name,
    value,
    *,
    domain="ingestion",
    source="a" * 64,
    target_min=0.8,
    target_max=1.0,
    critical_min=0.6,
    critical_max=1.0,
    warning_delta=0.1,
    critical_delta=0.2,
):
    return AggregateQualityMetric(
        metric_name=name,
        domain=domain,
        observed_value=value,
        target_min=target_min,
        target_max=target_max,
        critical_min=critical_min,
        critical_max=critical_max,
        drift_warning_delta=warning_delta,
        drift_critical_delta=critical_delta,
        sample_count=1000,
        source_evidence_fingerprint=source,
    )


def snapshot(snapshot_id, metrics):
    return ProductionQualitySnapshotService().build(
        request=ProductionQualitySnapshotRequest(
            tenant_id=TENANT,
            snapshot_id=snapshot_id,
            execution_mode="production",
        ),
        metrics=metrics,
    )


def full_evidence():
    baseline = snapshot("baseline", [
        metric("completeness_rate", 0.99, warning_delta=0.02, critical_delta=0.08),
        metric(
            "retrieval_recall", 0.92, domain="retrieval",
            warning_delta=0.05, critical_delta=0.15,
        ),
        metric("agent_success_rate", 0.99, domain="agent"),
        metric("legacy_metric", 0.9, domain="feature"),
    ])
    current = snapshot("current", [
        metric("completeness_rate", 0.96, warning_delta=0.02, critical_delta=0.08),
        metric(
            "retrieval_recall", 0.55, domain="retrieval",
            warning_delta=0.05, critical_delta=0.15,
        ),
        metric("agent_success_rate", 0.99, domain="agent"),
        metric("new_metric", 0.9, domain="privacy", source="b" * 64),
    ])
    drift = ProductionQualityDriftService().analyze(
        request=ProductionQualityDriftRequest(
            tenant_id=TENANT,
            baseline_snapshot_fingerprint=baseline[
                "quality_snapshot_fingerprint"
            ],
            current_snapshot_fingerprint=current[
                "quality_snapshot_fingerprint"
            ],
        ),
        baseline_snapshot=baseline,
        current_snapshot=current,
    )
    alerts = ProductionQualityAlertPlanningService().plan(
        drift_report=drift
    )
    return baseline, current, drift, alerts


def test_snapshot_classifies_healthy_warning_and_critical_aggregate_metrics():
    report = snapshot("classification", [
        metric("healthy_metric", 0.9),
        metric("warning_metric", 0.7),
        metric("critical_metric", 0.5),
    ])
    assert report["healthy_count"] == 1
    assert report["warning_count"] == 1
    assert report["critical_count"] == 1
    assert report["overall_health"] == "critical"
    assert report["safety"]["aggregate_metrics_only"] is True
    assert report["safety"]["individual_records_read"] is False


def test_metric_contract_rejects_invalid_ranges_and_non_finite_values():
    with pytest.raises(ValueError, match="thresholds"):
        metric("invalid", 0.9, target_min=0.9, target_max=0.8)
    with pytest.raises(ValueError, match="finite"):
        metric("invalid", float("nan"))


def test_drift_detects_degradation_new_missing_and_stable_metrics():
    _, _, drift, _ = full_evidence()
    by_key = {
        (value["domain"], value["metric_name"]): value
        for value in drift["drift_entries"]
    }
    assert by_key[("retrieval", "retrieval_recall")]["severity"] == "critical"
    assert by_key[("feature", "legacy_metric")]["change_type"] == "missing"
    assert by_key[("privacy", "new_metric")]["change_type"] == "new"
    assert by_key[("agent", "agent_success_rate")]["severity"] == "none"
    assert drift["drift_detected"] is True


def test_alert_plan_is_review_only_and_never_dispatches_or_remediates():
    _, _, _, alerts = full_evidence()
    for value in alerts["alerts"]:
        assert value["automatic_remediation_performed"] is False
        assert value["external_alert_dispatched"] is False
        assert value["manual_approval_required"] is True
        assert value["production_routing_enabled"] is False
    assert alerts["safety"]["activation_or_export_performed"] is False


def test_cross_tenant_drift_lineage_fails_closed():
    baseline, current, _, _ = full_evidence()
    with pytest.raises(ValueError, match="tenant"):
        ProductionQualityDriftService().analyze(
            request=ProductionQualityDriftRequest(
                tenant_id="another_tenant",
                baseline_snapshot_fingerprint=baseline[
                    "quality_snapshot_fingerprint"
                ],
                current_snapshot_fingerprint=current[
                    "quality_snapshot_fingerprint"
                ],
            ),
            baseline_snapshot=baseline,
            current_snapshot=current,
        )


def test_tampering_and_unsafe_alert_actions_fail_closed():
    _, current, drift, alerts = full_evidence()
    tampered_snapshot = deepcopy(current)
    tampered_snapshot["metric_observations"][0]["observed_value"] = 0.0
    with pytest.raises(ValueError, match="assessment|fingerprint"):
        ProductionQualitySnapshotService().validate_report(tampered_snapshot)

    tampered_drift = deepcopy(drift)
    changed_entry = next(
        value
        for value in tampered_drift["drift_entries"]
        if value["severity"] != "none"
    )
    changed_entry["severity"] = "none"
    with pytest.raises(ValueError, match="fingerprint"):
        ProductionQualityDriftService().validate_report(tampered_drift)

    unsafe_alerts = deepcopy(alerts)
    unsafe_alerts["alerts"][0]["automatic_remediation_performed"] = True
    with pytest.raises(ValueError, match="cannot set"):
        ProductionQualityAlertPlanningService().validate_report(unsafe_alerts)

    false_action = deepcopy(alerts)
    changed_alert = next(
        value
        for value in false_action["alerts"]
        if value["severity"] != "none"
    )
    changed_alert["recommended_action"] = "continue_monitoring"
    with pytest.raises(ValueError, match="action"):
        ProductionQualityAlertPlanningService().validate_report(false_action)


def test_quality_status_accepts_complete_safe_evidence_chain(tmp_path):
    from app.services.production_quality_monitoring_status_service import (
        ProductionQualityMonitoringStatusService,
    )

    _, current, drift, alerts = full_evidence()
    values = {
        "QUALITY_MONITORING_EVIDENCE_PATH": current,
        "QUALITY_DRIFT_EVIDENCE_PATH": drift,
        "QUALITY_ALERT_PLAN_EVIDENCE_PATH": alerts,
    }
    environment = {}
    for key, value in values.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionQualityMonitoringStatusService(
        environment=environment
    ).status()
    assert status["status"] == "quality_monitoring_engineering_evidence_ready"
    assert status["quality_monitoring_engineering_evidence_ready"] is True
    assert status["live_production_certified"] is False
    assert all(status["components"].values())


def test_quality_status_blocks_automatic_remediation_configuration():
    from app.services.production_quality_monitoring_status_service import (
        ProductionQualityMonitoringStatusService,
    )

    status = ProductionQualityMonitoringStatusService(environment={
        "QUALITY_AUTOMATIC_REMEDIATION_ENABLED": "true",
    }).status()
    assert status["status"] == (
        "unsafe_configuration_release_affecting_feature_blocked"
    )


def test_quality_status_router_and_safe_defaults_are_registered():
    main = Path("app/main.py").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "audience_intelligence_quality_status_router" in main
    assert "QUALITY_AUTOMATIC_REMEDIATION_ENABLED=false" in env
    assert "QUALITY_PRODUCTION_ROUTING_ENABLED=false" in env


def test_quality_migration_is_immutable_tenant_scoped_and_non_releasing():
    sql = Path(
        "migrations/0023_production_aggregate_quality_monitoring.sql"
    ).read_text(encoding="utf-8")
    for fragment in (
        "aggregate_metrics_only = TRUE",
        "raw_identifiers_stored = FALSE",
        "individual_records_read = FALSE",
        "automatic_remediation_performed = FALSE",
        "external_alert_dispatched = FALSE",
        "production_routing_enabled = FALSE",
        "activation_or_export_performed = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting(''app.tenant_id'', true)",
        "prevent_production_quality_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql
