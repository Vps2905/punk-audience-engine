import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_module4_evolution_contracts import Module4EvolutionSnapshotRequest
from app.services.production_module4_evolution_snapshot_service import ProductionModule4EvolutionSnapshotService


def request(mode="historical_preview"):
    return Module4EvolutionSnapshotRequest(
        tenant_id="tenant-a", source_run_id="run-1", execution_mode=mode
    )


def row(**overrides):
    value = {
        "export_cohort_id": "cohort-1",
        "management_quality_score": 0.90,
        "freshness_status": "fresh",
        "approval_status": "pending_approval",
        "data_safety_status": "safe_aggregate",
        "risk_decision": "allow_review",
    }
    value.update(overrides)
    return value


def test_historical_snapshot_is_monitoring_only_and_immutable():
    report = ProductionModule4EvolutionSnapshotService().build(
        request=request(), cohort_rows=[row()]
    ).to_record()
    assert report["historical_count"] == 1
    assert report["cohort_snapshots"][0]["monitoring_status"] == "historical_baseline"
    assert report["safety"]["automatic_evolution_performed"] is False
    assert report["safety"]["activation_or_export_performed"] is False


def test_production_snapshot_classifies_quality_stale_and_policy_states():
    rows = [
        row(export_cohort_id="stable"),
        row(export_cohort_id="quality", management_quality_score=0.2),
        row(export_cohort_id="stale", freshness_status="stale"),
        row(export_cohort_id="blocked", risk_decision="blocked_sensitive"),
    ]
    report = ProductionModule4EvolutionSnapshotService().build(
        request=request("production"), cohort_rows=rows
    ).to_record()
    assert report["monitoring_count"] == 1
    assert report["review_required_count"] == 1
    assert report["paused_count"] == 1
    assert report["blocked_count"] == 1


def test_duplicate_empty_or_identifier_bearing_input_fails_closed():
    service = ProductionModule4EvolutionSnapshotService()
    with pytest.raises(ValueError, match="at least one"):
        service.build(request=request(), cohort_rows=[])
    with pytest.raises(ValueError, match="unique"):
        service.build(request=request(), cohort_rows=[row(), row()])
    with pytest.raises(ValueError, match="identifier"):
        service.build(request=request(), cohort_rows=[row(raw_device_id="blocked")])


def test_report_validator_detects_tampering_and_unsafe_flags():
    service = ProductionModule4EvolutionSnapshotService()
    report = service.build(request=request(), cohort_rows=[row()]).to_record()
    assert service.validate_report(report) == report
    tampered = deepcopy(report)
    tampered["cohort_snapshots"][0]["quality_score"] = 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        service.validate_report(tampered)
    unsafe = deepcopy(report)
    unsafe["safety"]["automatic_evolution_performed"] = True
    with pytest.raises(ValueError, match="Unsafe Module 4.1"):
        service.validate_report(unsafe)


def test_module4_status_accepts_only_safe_evidence(tmp_path):
    from app.services.production_module4_status_service import ProductionModule4StatusService

    report = ProductionModule4EvolutionSnapshotService().build(
        request=request(), cohort_rows=[row()]
    ).to_record()
    path = tmp_path / "module4.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    status = ProductionModule4StatusService(environment={
        "MODULE4_EVOLUTION_SNAPSHOT_EVIDENCE_PATH": str(path)
    }).status()
    assert status["status"] == "module4_1_engineering_evidence_ready"
    assert status["module4_1_engineering_evidence_ready"] is True


def test_module4_release_affecting_flags_fail_closed():
    from app.services.production_module4_status_service import ProductionModule4StatusService

    status = ProductionModule4StatusService(environment={
        "MODULE4_AUTOMATIC_EVOLUTION_ENABLED": "true"
    }).status()
    assert status["status"] == "unsafe_configuration_release_affecting_feature_blocked"


def test_module4_migration_and_status_api_are_tenant_scoped_and_safe():
    sql = Path("migrations/0020_module4_governed_evolution_snapshots.sql").read_text()
    for fragment in (
        "automatic_evolution_performed = FALSE",
        "automatic_approval_performed = FALSE",
        "routing_enabled = FALSE",
        "eligible_for_activation = FALSE",
        "eligible_for_export = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting('app.tenant_id', true)",
        "prevent_module4_evolution_snapshot_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql
    main_text = Path("app/main.py").read_text()
    assert "audience_intelligence_module4_status_router" in main_text
    env_text = Path(".env.example").read_text()
    assert "MODULE4_AUTOMATIC_EVOLUTION_ENABLED=false" in env_text
    assert "MODULE4_PRODUCTION_ROUTING_ENABLED=false" in env_text
