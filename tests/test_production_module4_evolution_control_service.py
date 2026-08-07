import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_module4_evolution_contracts import (
    Module4EvolutionSnapshotRequest,
)
from app.models.production_module4_evolution_control_contracts import (
    Module4DriftAnalysisRequest,
)
from app.services.production_module4_evolution_snapshot_service import (
    ProductionModule4EvolutionSnapshotService,
)
from app.services.production_module4_evolution_control_service import (
    ProductionModule4ApprovalShadowService,
    ProductionModule4DriftDetectionService,
    ProductionModule4EvolutionRecommendationService,
    ProductionModule4RecoveryPlanningService,
)


TENANT = "tenant_a"


def cohort(cohort_id, *, quality=0.9, freshness="fresh", risk="allow_review"):
    return {
        "export_cohort_id": cohort_id,
        "management_quality_score": quality,
        "freshness_status": freshness,
        "approval_status": "pending_approval",
        "data_safety_status": "safe_aggregate",
        "risk_decision": risk,
    }


def snapshot(run_id, rows):
    return ProductionModule4EvolutionSnapshotService().build(
        request=Module4EvolutionSnapshotRequest(
            tenant_id=TENANT,
            source_run_id=run_id,
            execution_mode="production",
        ),
        cohort_rows=rows,
    ).to_record()


def drift(baseline, current):
    return ProductionModule4DriftDetectionService().analyze(
        request=Module4DriftAnalysisRequest(
            tenant_id=TENANT,
            baseline_snapshot_fingerprint=baseline["snapshot_fingerprint"],
            current_snapshot_fingerprint=current["snapshot_fingerprint"],
            execution_mode="production",
        ),
        baseline_snapshot=baseline,
        current_snapshot=current,
    )


def full_evidence():
    baseline = snapshot(
        "run-1",
        [cohort("stable"), cohort("drop"), cohort("missing")],
    )
    current = snapshot(
        "run-2",
        [cohort("stable"), cohort("drop", quality=0.5), cohort("new")],
    )
    drift_report = drift(baseline, current)
    recommendations = ProductionModule4EvolutionRecommendationService().recommend(
        drift_report=drift_report
    )
    decisions = []
    observations = []
    for index, value in enumerate(recommendations["recommendations"]):
        decision = (
            "approved_for_shadow_review" if index == 0 else "rejected"
        )
        decisions.append({
            "recommendation_fingerprint": value["recommendation_fingerprint"],
            "decision": decision,
            "decision_reference": f"review-{index}",
        })
        if decision == "approved_for_shadow_review":
            observations.append({
                "recommendation_fingerprint": value["recommendation_fingerprint"],
                "observation_status": "aligned",
                "routing_enabled": False,
            })
    approval = ProductionModule4ApprovalShadowService().evaluate(
        recommendation_report=recommendations,
        manual_review_records=decisions,
        shadow_observations=observations,
    )
    recovery = ProductionModule4RecoveryPlanningService().plan(
        recommendation_report=recommendations,
        approval_shadow_report=approval,
    )
    return current, drift_report, recommendations, approval, recovery


def test_drift_detects_quality_drop_new_missing_and_stable_cohorts():
    _, drift_report, _, _, _ = full_evidence()
    assert drift_report["drift_detected"] is True
    assert drift_report["new_count"] == 1
    assert drift_report["missing_count"] == 1
    assert drift_report["critical_count"] == 1
    by_id = {v["export_cohort_id"]: v for v in drift_report["drift_entries"]}
    assert by_id["drop"]["severity"] == "high"
    assert by_id["stable"]["change_type"] == "unchanged"


def test_recommendations_are_review_only_and_never_mutate():
    _, _, report, _, _ = full_evidence()
    actions = {v["recommended_action"] for v in report["recommendations"]}
    assert "rollback_review" in actions
    assert "pause_and_review" in actions
    for value in report["recommendations"]:
        assert value["manual_approval_required"] is True
        assert value["lifecycle_mutated"] is False
        assert value["eligible_for_export"] is False


def test_manual_shadow_evidence_does_not_route_or_activate():
    _, _, _, approval, _ = full_evidence()
    assert approval["shadow_validation_passed"] is True
    assert approval["approved_for_shadow_review_count"] == 1
    assert approval["safety"]["routing_enabled"] is False
    assert approval["safety"]["activation_or_export_performed"] is False


def test_recovery_plans_are_complete_but_never_executed():
    _, _, _, approval, recovery = full_evidence()
    assert recovery["recovery_coverage_complete"] is True
    assert recovery["recovery_plan_count"] == approval["manual_review_count"]
    assert all(v["recovery_executed"] is False for v in recovery["recovery_plans"])
    assert recovery["safety"]["recovery_executed"] is False


def test_shadow_requires_explicit_manual_approval():
    _, _, recommendations, _, _ = full_evidence()
    value = recommendations["recommendations"][0]
    with pytest.raises(ValueError, match="requires manual"):
        ProductionModule4ApprovalShadowService().evaluate(
            recommendation_report=recommendations,
            manual_review_records=[{
                "recommendation_fingerprint": value["recommendation_fingerprint"],
                "decision": "rejected",
                "decision_reference": "review-1",
            }],
            shadow_observations=[{
                "recommendation_fingerprint": value["recommendation_fingerprint"],
                "observation_status": "aligned",
                "routing_enabled": False,
            }],
        )


def test_tampering_and_unsafe_routing_fail_closed():
    _, drift_report, recommendations, approval, recovery = full_evidence()
    tampered = deepcopy(drift_report)
    tampered["drift_entries"][0]["severity"] = "none"
    with pytest.raises(ValueError, match="fingerprint"):
        ProductionModule4DriftDetectionService().validate_report(tampered)

    unsafe = deepcopy(recommendations)
    unsafe["recommendations"][0]["routing_enabled"] = True
    with pytest.raises(ValueError, match="fingerprint|Unsafe"):
        ProductionModule4EvolutionRecommendationService().validate_report(unsafe)

    unsafe_recovery = deepcopy(recovery)
    unsafe_recovery["safety"]["recovery_executed"] = True
    with pytest.raises(ValueError, match="Unsafe|cannot execute"):
        ProductionModule4RecoveryPlanningService().validate_report(unsafe_recovery)

    false_shadow_claim = deepcopy(approval)
    false_shadow_claim["shadow_validation_passed"] = False
    with pytest.raises(ValueError, match="shadow result"):
        ProductionModule4ApprovalShadowService().validate_report(
            false_shadow_claim
        )


def test_cross_tenant_snapshot_lineage_fails_closed():
    baseline = snapshot("run-1", [cohort("stable")])
    current = snapshot("run-2", [cohort("stable")])
    request = Module4DriftAnalysisRequest(
        tenant_id="another_tenant",
        baseline_snapshot_fingerprint=baseline["snapshot_fingerprint"],
        current_snapshot_fingerprint=current["snapshot_fingerprint"],
        execution_mode="production",
    )
    with pytest.raises(ValueError, match="tenant"):
        ProductionModule4DriftDetectionService().analyze(
            request=request,
            baseline_snapshot=baseline,
            current_snapshot=current,
        )


def test_module4_status_accepts_complete_safe_evidence_chain(tmp_path):
    from app.services.production_module4_status_service import (
        ProductionModule4StatusService,
    )

    current, drift_report, recommendations, approval, recovery = full_evidence()
    values = {
        "MODULE4_EVOLUTION_SNAPSHOT_EVIDENCE_PATH": current,
        "MODULE4_DRIFT_EVIDENCE_PATH": drift_report,
        "MODULE4_RECOMMENDATION_EVIDENCE_PATH": recommendations,
        "MODULE4_APPROVAL_SHADOW_EVIDENCE_PATH": approval,
        "MODULE4_RECOVERY_EVIDENCE_PATH": recovery,
    }
    environment = {}
    for key, value in values.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionModule4StatusService(environment=environment).status()
    assert status["status"] == "module4_engineering_evidence_ready"
    assert status["module4_engineering_evidence_ready"] is True
    assert all(status["components"].values())


def test_module4_control_migration_is_immutable_tenant_scoped_and_non_releasing():
    sql = Path("migrations/0021_module4_evolution_control_plane.sql").read_text()
    for fragment in (
        "automatic_evolution_performed = FALSE",
        "automatic_approval_performed = FALSE",
        "manual_approval_required = TRUE",
        "routing_enabled = FALSE",
        "eligible_for_activation = FALSE",
        "eligible_for_export = FALSE",
        "recovery_executed = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting(''app.tenant_id'', true)",
        "prevent_module4_evolution_control_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql
