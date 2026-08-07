import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_module5_governance_contracts import (
    Module5AgentExecutionRequest,
    Module5CircuitBreakerRequest,
)
from app.services.production_module5_governance_service import (
    ProductionModule5AgentExecutionService,
    ProductionModule5CircuitBreakerService,
    ProductionModule5HumanShadowReviewService,
    ProductionModule5PolicyEvaluationService,
    ProductionModule5RecoveryCertificationService,
)


TENANT = "tenant_a"


def supervisor_result(
    run_id="run-1",
    *,
    status="completed",
    route="completed_safe",
    recovery=None,
):
    return {
        "status": status,
        "run_id": run_id,
        "supervisor_route": route,
        "supervisor_stage": "autonomous_decision",
        "supervisor_reason_codes": ["safe_review_route"],
        "graph_terminal_status": route,
        "supervisor_graph_trace": [
            {"event": "graph_started", "prompt": "must-not-be-copied"},
            {"event": "graph_terminated", "secret": "must-not-be-copied"},
        ],
        "supervisor_recovery": recovery or {
            "attempt_count": 1,
            "max_attempts": 2,
            "retried": False,
            "exhausted": False,
            "last_error_category": None,
        },
        "downstream_export_enabled": False,
        "safe_export": {"downstream_export_enabled": False},
    }


def execution(run_id="run-1", **kwargs):
    return ProductionModule5AgentExecutionService().record(
        request=Module5AgentExecutionRequest(
            tenant_id=TENANT,
            request_id=f"request-{run_id}",
            run_id=run_id,
            execution_mode="production",
        ),
        supervisor_result=supervisor_result(run_id, **kwargs),
    )


def full_evidence():
    execution_report = execution()
    policy = ProductionModule5PolicyEvaluationService().evaluate(
        execution_report=execution_report
    )
    circuit = ProductionModule5CircuitBreakerService().evaluate(
        request=Module5CircuitBreakerRequest(
            tenant_id=TENANT,
            execution_report_fingerprints=(
                execution_report["execution_report_fingerprint"],
            ),
            minimum_sample_size=1,
        ),
        execution_reports=[execution_report],
    )
    review = ProductionModule5HumanShadowReviewService().evaluate(
        policy_report=policy,
        manual_review={
            "decision": "approved_for_shadow_review",
            "review_reference": "manual-review-1",
        },
        shadow_observation={
            "observation_status": "aligned",
            "routing_enabled": False,
            "activation_or_export_performed": False,
        },
    )
    recovery = ProductionModule5RecoveryCertificationService().plan(
        circuit_breaker_report=circuit,
        human_review_report=review,
    )
    return execution_report, policy, circuit, review, recovery


def test_execution_evidence_is_minimized_and_does_not_copy_trace_payloads():
    report = execution()
    serialized = json.dumps(report)
    assert "must-not-be-copied" not in serialized
    assert "original_prompt" not in serialized.lower()
    assert "tool_payload" not in serialized.lower()
    assert report["execution"]["trace_event_counts"] == {
        "graph_started": 1,
        "graph_terminated": 1,
    }
    assert report["safety"]["agent_execution_triggered"] is False


def test_execution_evidence_rejects_export_enabled_supervisor_output():
    unsafe = supervisor_result()
    unsafe["downstream_export_enabled"] = True
    with pytest.raises(ValueError, match="export disabled"):
        ProductionModule5AgentExecutionService().record(
            request=Module5AgentExecutionRequest(
                tenant_id=TENANT,
                request_id="request-1",
                run_id="run-1",
                execution_mode="production",
            ),
            supervisor_result=unsafe,
        )


def test_execution_evidence_rejects_unbounded_reason_metadata():
    unsafe = supervisor_result()
    unsafe["supervisor_reason_codes"] = ["secret value copied from prompt"]
    with pytest.raises(ValueError, match="safe metadata token"):
        ProductionModule5AgentExecutionService().record(
            request=Module5AgentExecutionRequest(
                tenant_id=TENANT,
                request_id="request-1",
                run_id="run-1",
                execution_mode="production",
            ),
            supervisor_result=unsafe,
        )


def test_policy_evaluation_is_review_only_and_never_authorizes_execution():
    _, policy, _, _, _ = full_evidence()
    decision = policy["decision"]
    assert decision["recommended_action"] == "shadow_review_only"
    assert decision["manual_approval_required"] is True
    assert decision["production_execution_authorized"] is False
    assert decision["agent_mutation_authorized"] is False
    assert decision["eligible_for_export"] is False


def test_circuit_breaker_opens_for_exhausted_recovery_without_retrying():
    healthy = execution("run-healthy")
    failed = execution(
        "run-failed",
        status="failed",
        route="failed",
        recovery={
            "attempt_count": 2,
            "max_attempts": 2,
            "retried": True,
            "exhausted": True,
            "last_error_category": "transient",
        },
    )
    reports = [healthy, failed]
    circuit = ProductionModule5CircuitBreakerService().evaluate(
        request=Module5CircuitBreakerRequest(
            tenant_id=TENANT,
            execution_report_fingerprints=tuple(
                value["execution_report_fingerprint"] for value in reports
            ),
            failure_rate_threshold=0.5,
            minimum_sample_size=2,
        ),
        execution_reports=reports,
    )
    assert circuit["assessment"]["circuit_state"] == "open"
    assert circuit["assessment"]["exhausted_count"] == 1
    assert circuit["safety"]["automatic_retry_triggered"] is False
    assert circuit["assessment"]["production_execution_authorized"] is False


def test_shadow_observation_requires_explicit_manual_approval():
    _, policy, _, _, _ = full_evidence()
    with pytest.raises(ValueError, match="requires manual"):
        ProductionModule5HumanShadowReviewService().evaluate(
            policy_report=policy,
            manual_review={
                "decision": "rejected",
                "review_reference": "manual-review-2",
            },
            shadow_observation={
                "observation_status": "aligned",
                "routing_enabled": False,
                "activation_or_export_performed": False,
            },
        )


def test_recovery_completes_engineering_chain_but_not_production_certification():
    _, _, _, review, recovery = full_evidence()
    assert review["shadow_validation_passed"] is True
    assert recovery["recovery_plan"]["engineering_evidence_complete"] is True
    assert recovery["recovery_plan"]["production_certified"] is False
    assert recovery["recovery_plan"]["recovery_executed"] is False
    assert recovery["safety"]["production_routing_enabled"] is False


def test_tampering_and_unsafe_authorization_fail_closed():
    execution_report, policy, circuit, review, recovery = full_evidence()
    tampered = deepcopy(execution_report)
    tampered["execution"]["supervisor_route"] = "failed"
    with pytest.raises(ValueError, match="fingerprint"):
        ProductionModule5AgentExecutionService().validate_report(tampered)

    unsafe_policy = deepcopy(policy)
    unsafe_policy["decision"]["production_execution_authorized"] = True
    with pytest.raises(ValueError, match="cannot authorize"):
        ProductionModule5PolicyEvaluationService().validate_report(unsafe_policy)

    false_circuit = deepcopy(circuit)
    false_circuit["assessment"]["circuit_state"] = "open"
    with pytest.raises(ValueError, match="circuit state"):
        ProductionModule5CircuitBreakerService().validate_report(false_circuit)

    false_review = deepcopy(review)
    false_review["shadow_validation_passed"] = False
    with pytest.raises(ValueError, match="shadow result"):
        ProductionModule5HumanShadowReviewService().validate_report(false_review)

    unsafe_recovery = deepcopy(recovery)
    unsafe_recovery["recovery_plan"]["production_certified"] = True
    with pytest.raises(ValueError, match="cannot set"):
        ProductionModule5RecoveryCertificationService().validate_report(
            unsafe_recovery
        )


def test_cross_tenant_circuit_breaker_lineage_fails_closed():
    report = execution()
    with pytest.raises(ValueError, match="tenant"):
        ProductionModule5CircuitBreakerService().evaluate(
            request=Module5CircuitBreakerRequest(
                tenant_id="another_tenant",
                execution_report_fingerprints=(
                    report["execution_report_fingerprint"],
                ),
                minimum_sample_size=1,
            ),
            execution_reports=[report],
        )


def test_module5_status_accepts_complete_safe_evidence_chain(tmp_path):
    from app.services.production_module5_status_service import (
        ProductionModule5StatusService,
    )

    execution_report, policy, circuit, review, recovery = full_evidence()
    values = {
        "MODULE5_EXECUTION_EVIDENCE_PATH": execution_report,
        "MODULE5_POLICY_EVIDENCE_PATH": policy,
        "MODULE5_CIRCUIT_BREAKER_EVIDENCE_PATH": circuit,
        "MODULE5_HUMAN_REVIEW_EVIDENCE_PATH": review,
        "MODULE5_RECOVERY_EVIDENCE_PATH": recovery,
    }
    environment = {}
    for key, value in values.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionModule5StatusService(environment=environment).status()
    assert status["status"] == "module5_engineering_evidence_ready"
    assert status["module5_engineering_evidence_ready"] is True
    assert status["live_production_certified"] is False
    assert all(status["components"].values())


def test_module5_status_blocks_release_affecting_configuration():
    from app.services.production_module5_status_service import (
        ProductionModule5StatusService,
    )

    status = ProductionModule5StatusService(environment={
        "MODULE5_AUTONOMOUS_MUTATION_ENABLED": "true",
    }).status()
    assert status["status"] == (
        "unsafe_configuration_release_affecting_feature_blocked"
    )
    assert status["module5_engineering_evidence_ready"] is False


def test_module5_status_route_is_registered():
    main = Path("app/main.py").read_text(encoding="utf-8")
    assert "audience_intelligence_module5_status_router" in main
    assert (
        "_include_audience_router(audience_intelligence_module5_status_router)"
        in main
    )


def test_module5_migration_is_immutable_tenant_scoped_and_non_releasing():
    sql = Path("migrations/0022_module5_governed_agent_control_plane.sql").read_text()
    for fragment in (
        "prompt_content_stored = FALSE",
        "tool_arguments_stored = FALSE",
        "agent_execution_triggered = FALSE",
        "automatic_retry_triggered = FALSE",
        "automatic_mutation_performed = FALSE",
        "automatic_approval_performed = FALSE",
        "production_execution_authorized = FALSE",
        "production_certified = FALSE",
        "recovery_executed = FALSE",
        "production_routing_enabled = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting(''app.tenant_id'', true)",
        "prevent_module5_agent_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql
