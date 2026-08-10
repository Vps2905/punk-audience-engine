from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_bounded_autonomy_certification_contracts import (
    ShadowCertificationCase,
    ShadowCertificationPolicy,
)
from app.models.production_bounded_autonomy_contracts import AutonomyGoal
from app.models.production_module3_cohort_contracts import stable_fingerprint
from app.services.production_bounded_autonomy_certification_service import (
    ProductionBoundedAutonomyCertificationService,
)
from app.services.production_bounded_autonomy_shadow_service import (
    LegacyOrchestratorObservationAdapter,
    ProductionBoundedAutonomyShadowComparisonService,
    RealServiceCapabilityAdapterFactory,
)
from app.services.production_module5_status_service import (
    ProductionModule5StatusService,
)


def goal(index: int, *, tenant_id: str = "tenant_a") -> AutonomyGoal:
    return AutonomyGoal(
        tenant_id=tenant_id,
        request_id=f"request-{index}",
        goal_id=f"goal-{index}",
        objective=f"Unseen generalized business objective number {index}.",
        requested_outcomes=("governed_recommendation",),
        execution_mode="shadow",
    )


def pending_result(index: int) -> dict:
    return {
        "status": "completed",
        "run_id": f"run-{index}",
        "source_rows": 220,
        "freshness_status": "fresh",
        "approval_status": "pending_approval",
        "approval_required": True,
        "downstream_export_enabled": False,
        "prompt_selected_cohorts": 3,
        "embedding": {"vector_count": 90, "vector_dimension": 384},
        "safe_export": {
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "exported_cohorts": 1,
        },
    }


def stale_result(index: int) -> dict:
    value = pending_result(index)
    value["freshness_status"] = "stale"
    value["approval_status"] = "blocked_stale_source"
    value["safe_export"] = {
        **value["safe_export"],
        "approval_status": "blocked_stale_source",
        "block_export": True,
    }
    return value


def privacy_result(index: int) -> dict:
    return {
        "status": "skipped",
        "run_id": f"run-{index}",
        "source_rows": None,
        "freshness_status": "not_evaluated",
        "approval_status": "blocked_privacy_identifier_request",
        "approval_required": True,
        "downstream_export_enabled": False,
        "prompt_selected_cohorts": 0,
        "prompt_filter_report": {
            "filter_mode": "privacy_identifier_request_blocked",
        },
    }


def case(index: int, kind: str = "pending") -> ShadowCertificationCase:
    result = {
        "pending": pending_result,
        "stale": stale_result,
        "privacy": privacy_result,
    }[kind](index)
    return ShadowCertificationCase(
        goal=goal(index),
        legacy_result=result,
        legacy_latency_ms=5.0 + index / 100.0,
    )


def passing_policy(total: int = 6) -> ShadowCertificationPolicy:
    return ShadowCertificationPolicy(
        min_total_runs=total,
        min_unique_goal_hashes=total,
        min_terminal_safety_runs=total // 2,
        min_nonterminal_review_runs=total // 2,
        max_shadow_p95_latency_ms=1000.0,
        max_cases=max(total, 10),
    )


def passing_cases() -> list[ShadowCertificationCase]:
    return [
        case(0, "pending"),
        case(1, "pending"),
        case(2, "pending"),
        case(3, "stale"),
        case(4, "privacy"),
        case(5, "stale"),
    ]


def test_diverse_agreeing_cases_pass_only_staging_review_gate():
    report = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    ).run(tenant_id="tenant_a", cases=passing_cases())

    assert report["status"] == "engineering_preview_ready"
    assert report["summary"]["evaluated_run_count"] == 6
    assert report["summary"]["unique_goal_hash_count"] == 6
    assert report["summary"]["terminal_safety_run_count"] == 3
    assert report["summary"]["nonterminal_review_run_count"] == 3
    assert report["summary"]["route_agreement_rate"] == 1.0
    assert report["summary"]["critical_divergence_rate"] == 0.0
    assert report["review"]["eligible_for_staging_review"] is True
    assert report["review"]["live_cutover_authorized"] is False
    assert report["review"]["fresh_data_certified"] is False
    assert report["safety"]["activation_or_export_performed"] is False


def test_default_policy_blocks_an_insufficient_sample():
    report = ProductionBoundedAutonomyCertificationService().run(
        tenant_id="tenant_a",
        cases=[case(0, "pending")],
    )

    assert report["status"] == "engineering_preview_blocked"
    assert report["certification_gates"]["minimum_runs"]["passed"] is False
    assert report["review"]["eligible_for_staging_review"] is False


def test_prompts_and_legacy_payloads_never_enter_certification_evidence():
    secret = "confidential-unseen-goal-marker"
    cases = passing_cases()
    first = cases[0]
    cases[0] = ShadowCertificationCase(
        goal=AutonomyGoal(
            tenant_id="tenant_a",
            request_id="request-secret",
            goal_id="goal-secret",
            objective=secret,
            requested_outcomes=("governed_recommendation",),
            execution_mode="shadow",
        ),
        legacy_result={
            **dict(first.legacy_result),
            "source_payload": {"private_marker": secret},
        },
        legacy_latency_ms=first.legacy_latency_ms,
    )
    report = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    ).run(tenant_id="tenant_a", cases=cases)
    serialized = json.dumps(report)

    assert secret not in serialized
    assert "source_payload" not in serialized
    assert "private_marker" not in serialized


def test_cross_tenant_cases_are_rejected_before_execution():
    cases = passing_cases()
    cases[-1] = ShadowCertificationCase(
        goal=goal(99, tenant_id="tenant_b"),
        legacy_result=stale_result(99),
    )

    with pytest.raises(ValueError, match="one tenant boundary"):
        ProductionBoundedAutonomyCertificationService(
            policy=passing_policy()
        ).run(tenant_id="tenant_a", cases=cases)


def test_production_mode_case_is_rejected_by_contract():
    production_goal = AutonomyGoal(
        tenant_id="tenant_a",
        request_id="request-production",
        goal_id="goal-production",
        objective="A prohibited production certification case.",
        requested_outcomes=("governed_recommendation",),
        execution_mode="production",
    )

    with pytest.raises(ValueError, match="cannot use production mode"):
        ShadowCertificationCase(
            goal=production_goal,
            legacy_result=pending_result(1),
        )


def test_unsafe_delivery_divergence_blocks_certification():
    cases = passing_cases()
    unsafe = pending_result(2)
    unsafe.update(
        approval_status="approved",
        approval_required=False,
        downstream_export_enabled=True,
    )
    cases[2] = ShadowCertificationCase(goal=goal(2), legacy_result=unsafe)
    report = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    ).run(tenant_id="tenant_a", cases=cases)

    assert report["status"] == "engineering_preview_blocked"
    assert report["summary"]["critical_divergence_rate"] > 0.0
    assert report["certification_gates"]["critical_divergence"][
        "passed"
    ] is False
    assert report["review"]["live_cutover_authorized"] is False


class FailingComparisonService:
    def run(self, **_kwargs):
        raise RuntimeError("secret exception detail must never persist")

    def validate_report(self, report):
        return dict(report)


def test_comparator_exception_becomes_bounded_failure_without_error_text():
    report = ProductionBoundedAutonomyCertificationService(
        comparison_service=FailingComparisonService(),
        policy=ShadowCertificationPolicy(
            min_total_runs=1,
            min_unique_goal_hashes=1,
            min_terminal_safety_runs=1,
            min_nonterminal_review_runs=0,
            max_shadow_p95_latency_ms=1000,
            max_cases=1,
        ),
    ).run(tenant_id="tenant_a", cases=[case(0, "pending")])
    serialized = json.dumps(report)

    assert report["status"] == "engineering_preview_blocked"
    assert report["summary"]["contract_failure_count"] == 1
    assert report["certification_gates"]["contract_integrity"][
        "passed"
    ] is False
    assert "secret exception" not in serialized


def test_duplicate_comparison_fingerprints_do_not_inflate_sample_count():
    duplicate = case(0, "pending")
    report = ProductionBoundedAutonomyCertificationService(
        policy=ShadowCertificationPolicy(
            min_total_runs=1,
            min_unique_goal_hashes=1,
            min_terminal_safety_runs=0,
            min_nonterminal_review_runs=1,
            max_shadow_p95_latency_ms=1000,
            max_cases=2,
        )
    ).run(tenant_id="tenant_a", cases=[duplicate, duplicate])

    assert report["summary"]["submitted_case_count"] == 2
    assert report["summary"]["evaluated_run_count"] == 1
    assert report["summary"]["duplicate_case_count"] == 1
    assert report["certification_gates"]["duplicate_resistance"][
        "passed"
    ] is False


class SlowComparisonService:
    def __init__(self):
        self.delegate = ProductionBoundedAutonomyShadowComparisonService()

    def run(self, **kwargs):
        time.sleep(0.004)
        return self.delegate.run(**kwargs)

    def validate_report(self, report):
        return self.delegate.validate_report(report)


def test_p95_latency_gate_blocks_slow_shadow_control_plane():
    report = ProductionBoundedAutonomyCertificationService(
        comparison_service=SlowComparisonService(),
        policy=ShadowCertificationPolicy(
            min_total_runs=2,
            min_unique_goal_hashes=2,
            min_terminal_safety_runs=1,
            min_nonterminal_review_runs=1,
            max_shadow_p95_latency_ms=1.0,
            max_cases=2,
        ),
    ).run(
        tenant_id="tenant_a",
        cases=[case(0, "pending"), case(1, "stale")],
    )

    assert report["summary"]["shadow_latency_ms"]["p95"] > 1.0
    assert report["certification_gates"]["shadow_p95_latency"][
        "passed"
    ] is False
    assert report["status"] == "engineering_preview_blocked"


def test_case_budget_is_enforced_before_execution():
    policy = ShadowCertificationPolicy(
        min_total_runs=1,
        min_unique_goal_hashes=1,
        min_terminal_safety_runs=0,
        min_nonterminal_review_runs=0,
        max_cases=1,
    )

    with pytest.raises(ValueError, match="exceeds max_cases"):
        ProductionBoundedAutonomyCertificationService(policy=policy).run(
            tenant_id="tenant_a",
            cases=[case(0), case(1)],
        )


def test_sample_tampering_is_detected():
    service = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    )
    report = service.run(tenant_id="tenant_a", cases=passing_cases())
    tampered = deepcopy(report)
    tampered["samples"][0]["route_match"] = False

    with pytest.raises(ValueError, match="sample fingerprint mismatch"):
        service.validate_report(tampered)


def test_summary_tampering_is_detected_even_with_sample_fingerprints_intact():
    service = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    )
    report = service.run(tenant_id="tenant_a", cases=passing_cases())
    tampered = deepcopy(report)
    tampered["summary"]["route_agreement_rate"] = 0.5

    with pytest.raises(ValueError, match="summary mismatch"):
        service.validate_report(tampered)


def test_live_cutover_tampering_is_rejected():
    service = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    )
    report = service.run(tenant_id="tenant_a", cases=passing_cases())
    tampered = deepcopy(report)
    tampered["review"]["live_cutover_authorized"] = True

    with pytest.raises(ValueError, match="live_cutover_authorized"):
        service.validate_report(tampered)


def test_semantically_invalid_sample_is_rejected_after_fingerprints_are_rebuilt():
    service = ProductionBoundedAutonomyCertificationService(
        policy=passing_policy()
    )
    report = service.run(tenant_id="tenant_a", cases=passing_cases())
    tampered = deepcopy(report)
    tampered["samples"][0]["route_match"] = "true"
    sample_payload = dict(tampered["samples"][0])
    sample_payload.pop("sample_fingerprint")
    tampered["samples"][0]["sample_fingerprint"] = stable_fingerprint(
        sample_payload
    )
    tampered.pop("bounded_autonomy_certification_fingerprint")
    tampered["bounded_autonomy_certification_fingerprint"] = stable_fingerprint(
        tampered
    )

    with pytest.raises(TypeError, match="route_match must be boolean"):
        service.validate_report(tampered)


def test_environment_policy_is_loaded_and_invalid_values_fail_closed():
    service = ProductionBoundedAutonomyCertificationService(
        environment={
            "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_RUNS": "30",
            "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_UNIQUE_GOALS": "12",
            "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_CASES": "100",
        }
    )

    assert service.policy.min_total_runs == 30
    assert service.policy.min_unique_goal_hashes == 12
    with pytest.raises(ValueError, match="must be numeric"):
        ProductionBoundedAutonomyCertificationService(
            environment={
                "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_DIVERGENCE_RATE": "bad"
            }
        )


def test_250_case_batch_is_bounded_and_produces_complete_metrics():
    total = 250
    cases = [
        case(index, "pending" if index % 2 == 0 else "stale")
        for index in range(total)
    ]
    policy = ShadowCertificationPolicy(
        min_total_runs=total,
        min_unique_goal_hashes=total,
        min_terminal_safety_runs=total // 2,
        min_nonterminal_review_runs=total // 2,
        max_shadow_p95_latency_ms=1000,
        max_cases=total,
    )
    report = ProductionBoundedAutonomyCertificationService(policy=policy).run(
        tenant_id="tenant_a",
        cases=cases,
    )

    assert report["status"] == "engineering_preview_ready"
    assert report["summary"]["evaluated_run_count"] == total
    assert report["summary"]["duplicate_case_count"] == 0
    assert report["summary"]["shadow_latency_ms"]["p95"] > 0.0
    assert len(report["samples"]) == total


def test_module5_status_fails_closed_when_certification_enabled_without_evidence():
    status = ProductionModule5StatusService(
        environment={
            "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_ENABLED": "true"
        }
    ).status()

    assert status["status"] == (
        "unsafe_configuration_shadow_certification_not_ready"
    )
    assert status["components"][
        "module_5_8_repeated_shadow_certification"
    ] is False


def test_disabled_certification_configuration_cannot_break_status_path():
    status = ProductionModule5StatusService(
        environment={
            "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_ENABLED": "false",
            "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_DIVERGENCE_RATE": "invalid",
            "MODULE5_BOUNDED_AUTONOMY_MAX_VECTOR_COUNT_DELTA": "invalid",
        }
    ).status()

    assert status["status"] == "module5_1_evidence_pending"
    assert status["live_production_certified"] is False


def test_module5_status_accepts_full_bounded_comparison_certification_lineage(
    tmp_path,
):
    comparison_service = ProductionBoundedAutonomyShadowComparisonService()
    certification_service = ProductionBoundedAutonomyCertificationService(
        comparison_service=comparison_service,
        policy=passing_policy(),
    )
    certification = certification_service.run(
        tenant_id="tenant_a",
        cases=passing_cases(),
    )
    comparison = comparison_service.run(
        goal=goal(0),
        legacy_result=pending_result(0),
    )
    observation = LegacyOrchestratorObservationAdapter().adapt(
        goal=goal(0), legacy_result=pending_result(0)
    )
    adapters = RealServiceCapabilityAdapterFactory(observation)
    bounded = comparison_service.bounded_service.run(
        goal=goal(0), handlers=adapters.handlers()
    )

    bounded_path = tmp_path / "bounded.json"
    comparison_path = tmp_path / "comparison.json"
    certification_path = tmp_path / "certification.json"
    bounded_path.write_text(json.dumps(bounded), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    certification_path.write_text(json.dumps(certification), encoding="utf-8")
    status = ProductionModule5StatusService(
        environment={
            "MODULE5_BOUNDED_AUTONOMY_EVIDENCE_PATH": str(bounded_path),
            "MODULE5_BOUNDED_AUTONOMY_COMPARISON_EVIDENCE_PATH": str(
                comparison_path
            ),
            "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_EVIDENCE_PATH": str(
                certification_path
            ),
            "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_ENABLED": "true",
        }
    ).status()

    assert status["module5_bounded_autonomy_shadow_certified"] is True
    assert status["components"][
        "module_5_8_repeated_shadow_certification"
    ] is True
    assert status["live_production_certified"] is False


def test_module5_8_assets_preserve_the_non_authorizing_boundary():
    root = Path(__file__).resolve().parents[1]
    migration = (
        root / "migrations/0030_bounded_autonomy_shadow_certification.sql"
    ).read_text(encoding="utf-8")
    documentation = (
        root / "docs/module5_8_repeated_shadow_certification.md"
    ).read_text(encoding="utf-8")
    environment = (root / ".env.example").read_text(encoding="utf-8")

    assert "FORCE ROW LEVEL SECURITY" in migration
    assert "live_cutover_authorized = FALSE" in migration
    assert "fresh_data_certified = FALSE" in migration
    assert "activation_or_export_performed = FALSE" in migration
    assert "REVOKE ALL" in migration
    assert "Repeated Shadow Certification" in documentation
    assert (
        "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_ENABLED=false" in environment
    )
