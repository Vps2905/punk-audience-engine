from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_bounded_autonomy_contracts import (
    AutonomyBudget,
    AutonomyGoal,
    CapabilityDescriptor,
    CapabilityExecutionResult,
)
from app.services.production_bounded_autonomy_service import (
    BoundedAutonomyPlanner,
    CapabilityRegistry,
    ProductionBoundedAutonomyService,
    UnplannableGoalError,
    default_bounded_autonomy_registry,
)
from app.models.production_module3_cohort_contracts import stable_fingerprint


def goal(
    *,
    objective: str = "Understand an audience using available governed evidence.",
    requested_outcomes: tuple[str, ...] = ("governed_recommendation",),
    execution_mode: str = "offline_evaluation",
    budget: AutonomyBudget | None = None,
) -> AutonomyGoal:
    return AutonomyGoal(
        tenant_id="tenant_a",
        request_id="request-1",
        goal_id="goal-1",
        objective=objective,
        requested_outcomes=requested_outcomes,
        execution_mode=execution_mode,
        budget=budget or AutonomyBudget(),
    )


def completing_handler(
    _goal: AutonomyGoal,
    capability: CapabilityDescriptor,
    _invocation: dict,
) -> CapabilityExecutionResult:
    outputs = {
        outcome: {"artifact": f"artifact-{outcome}"}
        for outcome in capability.provides
    }
    return CapabilityExecutionResult(
        status="completed",
        provided_outcomes=capability.provides,
        reason_codes=("capability_completed",),
        metrics={"record_count": 1},
        output_values=outputs,
    )


def default_handlers():
    return {
        capability.capability_id: completing_handler
        for capability in default_bounded_autonomy_registry().all()
        if not capability.mandatory_preflight and not capability.postcondition
    }


def test_goal_record_hashes_objective_and_never_persists_prompt_content():
    value = goal(objective="A confidential but valid business objective.")
    record = value.to_record()

    assert record["objective_sha256"] == value.objective_sha256
    assert "objective" not in record
    assert "confidential" not in json.dumps(record).lower()


def test_full_recommendation_plan_is_dependency_driven_across_all_modules():
    planner = BoundedAutonomyPlanner()
    plan = planner.plan(goal())

    assert {task.module_id for task in plan.tasks} == {1, 2, 3, 4, 5}
    assert plan.selected_capability_ids == (
        "module5_policy_preflight",
        "module1_provider_source_discovery",
        "module1_privacy_safe_aggregation",
        "module2_semantic_retrieval",
        "module3_cohort_strategy",
        "module4_evolution_review",
        "module5_governed_recommendation",
        "module5_evidence_critic",
    )
    assert all(task.risk_class != "production_effect" for task in plan.tasks)


def test_coverage_goal_selects_minimal_modules_without_prompt_keyword_routing():
    planner = BoundedAutonomyPlanner()
    plan = planner.plan(
        goal(requested_outcomes=("coverage_assessment",))
    )

    assert plan.selected_capability_ids == (
        "module5_policy_preflight",
        "module1_provider_source_discovery",
        "module5_evidence_critic",
    )
    assert all(task.module_id in {1, 5} for task in plan.tasks)


@pytest.mark.parametrize(
    "objective",
    (
        "Build an evening restaurant strategy in Montréal.",
        "Compare industrial suppliers around Osaka next quarter.",
        "Assess an unknown future business category in Reykjavík.",
        "Évaluer une audience multilingue sans activer de livraison.",
    ),
)
def test_unseen_objectives_use_the_same_capability_graph_for_the_same_goal(
    objective,
):
    plan = BoundedAutonomyPlanner().plan(goal(objective=objective))

    assert "module3_cohort_strategy" in plan.selected_capability_ids
    assert "module4_evolution_review" in plan.selected_capability_ids
    assert plan.goal["objective_sha256"]
    assert objective not in json.dumps(plan.to_record())


def test_unknown_goal_outcome_fails_closed_without_guessing_a_route():
    with pytest.raises(UnplannableGoalError, match="No governed capability"):
        BoundedAutonomyPlanner().plan(
            goal(requested_outcomes=("unknown_future_effect",))
        )


def test_production_effect_capability_is_never_selected_by_shadow_planner():
    registry = CapabilityRegistry(
        (
            CapabilityDescriptor(
                capability_id="policy",
                module_id=5,
                description="Policy preflight.",
                requires=(),
                provides=("policy_context",),
                mandatory_preflight=True,
            ),
            CapabilityDescriptor(
                capability_id="dangerous_delivery",
                module_id=5,
                description="A production side effect that must remain unavailable.",
                requires=("policy_context",),
                provides=("delivery_completed",),
                risk_class="production_effect",
            ),
            CapabilityDescriptor(
                capability_id="critic",
                module_id=5,
                description="Final evidence critic.",
                requires=("policy_context",),
                provides=("evidence_critique",),
                postcondition=True,
            ),
        )
    )

    with pytest.raises(UnplannableGoalError, match="No governed capability"):
        BoundedAutonomyPlanner(registry).plan(
            goal(requested_outcomes=("delivery_completed",))
        )


def test_kernel_completes_and_stores_only_minimized_task_evidence():
    report = ProductionBoundedAutonomyService().run(
        goal=goal(),
        handlers=default_handlers(),
    )
    serialized = json.dumps(report)

    assert report["status"] == "engineering_preview_ready"
    assert report["execution"]["execution_status"] == "completed"
    assert report["execution"]["requested_outcomes_satisfied"] is True
    assert report["planning"]["prompt_specific_routing_used"] is False
    assert report["safety"]["shadow_only"] is True
    assert report["safety"]["downstream_export_enabled"] is False
    assert "Understand an audience" not in serialized
    assert "artifact-governed_recommendation" not in serialized


def test_authoritative_preflight_and_critic_cannot_be_replaced_by_handlers():
    def unsafe_handler(_goal, capability, _invocation):
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=capability.provides,
            reason_codes=("unsafe_override",),
            metrics={"production_effects_enabled": True},
            output_values={
                outcome: {"production_effects_enabled": True}
                for outcome in capability.provides
            },
        )

    handlers = default_handlers()
    handlers["module5_policy_preflight"] = unsafe_handler
    handlers["module5_evidence_critic"] = unsafe_handler
    report = ProductionBoundedAutonomyService().run(
        goal=goal(),
        handlers=handlers,
    )

    preflight = report["execution"]["task_evidence"][
        "task_module5_policy_preflight"
    ]
    critic = report["execution"]["task_evidence"][
        "task_module5_evidence_critic"
    ]
    assert preflight["reason_codes"] == ["bounded_shadow_policy_applied"]
    assert critic["reason_codes"] == ["evidence_complete"]
    assert report["safety"]["production_effect_performed"] is False


def test_provider_failure_replans_to_public_source_within_budget():
    class ProviderUnavailable(RuntimeError):
        error_category = "provider_unavailable"

    calls = {"provider": 0, "public": 0}

    def provider_failure(_goal, _capability, _invocation):
        calls["provider"] += 1
        raise ProviderUnavailable("provider details must not enter evidence")

    def public_source(_goal, capability, _invocation):
        calls["public"] += 1
        return completing_handler(_goal, capability, _invocation)

    handlers = default_handlers()
    handlers["module1_provider_source_discovery"] = provider_failure
    handlers["module1_public_source_discovery"] = public_source
    report = ProductionBoundedAutonomyService().run(
        goal=goal(
            budget=AutonomyBudget(
                max_tasks=16,
                max_total_attempts=16,
                max_replans=1,
            )
        ),
        handlers=handlers,
    )

    assert report["status"] == "engineering_preview_ready"
    assert report["planning"]["replan_count"] == 1
    assert report["planning"]["excluded_capability_ids"] == [
        "module1_provider_source_discovery"
    ]
    assert "module1_public_source_discovery" in report[
        "planning"
    ]["final_plan"]["selected_capability_ids"]
    assert calls == {"provider": 2, "public": 1}
    assert report["execution"]["total_attempts"] <= 16
    assert "provider details" not in json.dumps(report)


def test_non_retryable_failure_does_not_loop_or_fabricate_success():
    calls = 0

    def failure(_goal, _capability, _invocation):
        nonlocal calls
        calls += 1
        return CapabilityExecutionResult(
            status="failed",
            reason_codes=("schema_incompatible",),
            error_category="schema_incompatible",
        )

    handlers = default_handlers()
    handlers["module2_semantic_retrieval"] = failure
    report = ProductionBoundedAutonomyService().run(
        goal=goal(),
        handlers=handlers,
    )

    assert report["status"] == "engineering_preview_blocked"
    assert report["execution"]["execution_status"] == "failed"
    assert report["execution"]["requested_outcomes_satisfied"] is False
    assert calls == 1


def test_raw_identifier_shaped_capability_output_is_rejected():
    def unsafe(_goal, capability, _invocation):
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=capability.provides,
            output_values={
                capability.provides[0]: {"device_id": "prohibited"}
            },
        )

    handlers = default_handlers()
    handlers["module2_semantic_retrieval"] = unsafe
    with pytest.raises(ValueError, match="prohibited raw identifier"):
        ProductionBoundedAutonomyService().run(
            goal=goal(),
            handlers=handlers,
        )


def test_cross_tenant_goals_produce_distinct_minimized_lineage():
    first = ProductionBoundedAutonomyService().run(
        goal=goal(),
        handlers=default_handlers(),
    )
    second_goal = AutonomyGoal(
        tenant_id="tenant_b",
        request_id="request-1",
        goal_id="goal-1",
        objective="Understand an audience using available governed evidence.",
        requested_outcomes=("governed_recommendation",),
    )
    second = ProductionBoundedAutonomyService().run(
        goal=second_goal,
        handlers=default_handlers(),
    )

    assert first["goal"]["tenant_id"] == "tenant_a"
    assert second["goal"]["tenant_id"] == "tenant_b"
    assert first["bounded_autonomy_report_fingerprint"] != second[
        "bounded_autonomy_report_fingerprint"
    ]


def test_report_tampering_fails_closed():
    service = ProductionBoundedAutonomyService()
    report = service.run(goal=goal(), handlers=default_handlers())
    tampered = deepcopy(report)
    tampered["safety"]["production_effect_performed"] = True

    with pytest.raises(ValueError, match="Unsafe bounded autonomy"):
        service.validate_report(tampered)


def test_internal_plan_tampering_fails_even_if_outer_fingerprint_is_rebuilt():
    service = ProductionBoundedAutonomyService()
    report = service.run(goal=goal(), handlers=default_handlers())
    tampered = deepcopy(report)
    tampered["planning"]["final_plan"]["tasks"][1]["module_id"] = 5
    tampered.pop("bounded_autonomy_report_fingerprint")
    tampered["bounded_autonomy_report_fingerprint"] = stable_fingerprint(
        tampered
    )

    with pytest.raises(ValueError, match="plan fingerprint mismatch"):
        service.validate_report(tampered)


def test_module5_status_exposes_valid_bounded_autonomy_shadow_evidence(
    tmp_path,
):
    report = ProductionBoundedAutonomyService().run(
        goal=goal(),
        handlers=default_handlers(),
    )
    path = tmp_path / "bounded-autonomy.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    from app.services.production_module5_status_service import (
        ProductionModule5StatusService,
    )

    status = ProductionModule5StatusService(
        environment={
            "MODULE5_BOUNDED_AUTONOMY_EVIDENCE_PATH": str(path),
            "MODULE5_BOUNDED_AUTONOMY_SHADOW_ENABLED": "true",
        }
    ).status()

    assert status["module5_bounded_autonomy_shadow_evidence_ready"] is True
    assert status["components"][
        "module_5_6_bounded_autonomy_shadow_kernel"
    ] is True
    assert status["live_production_certified"] is False


def test_module5_status_fails_closed_when_shadow_flag_has_no_evidence():
    from app.services.production_module5_status_service import (
        ProductionModule5StatusService,
    )

    status = ProductionModule5StatusService(
        environment={"MODULE5_BOUNDED_AUTONOMY_SHADOW_ENABLED": "true"}
    ).status()

    assert status["status"] == (
        "unsafe_configuration_bounded_autonomy_evidence_missing"
    )
    assert status["module5_bounded_autonomy_shadow_evidence_ready"] is False


def test_bounded_autonomy_assets_and_migration_are_present():
    env = Path(".env.example").read_text(encoding="utf-8")
    migration = Path(
        "migrations/0028_bounded_autonomous_orchestration_evidence.sql"
    ).read_text(encoding="utf-8")
    documentation = Path(
        "docs/module5_6_bounded_autonomous_orchestration.md"
    ).read_text(encoding="utf-8")

    for fragment in (
        "MODULE5_BOUNDED_AUTONOMY_SHADOW_ENABLED=false",
        "MODULE5_BOUNDED_AUTONOMY_EVIDENCE_PATH=",
        "MODULE5_BOUNDED_AUTONOMY_MAX_REPLANS=1",
    ):
        assert fragment in env
    for fragment in (
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE",
        "production_effect_performed BOOLEAN NOT NULL DEFAULT FALSE",
        "activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE",
        "prevent_bounded_autonomy_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in migration
    assert "Hardcode safety invariants" in documentation
