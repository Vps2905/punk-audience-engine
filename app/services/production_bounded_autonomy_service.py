from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.models.production_bounded_autonomy_contracts import (
    BOUNDED_AUTONOMY_EVIDENCE_VERSION,
    BOUNDED_AUTONOMY_PLAN_VERSION,
    AutonomyGoal,
    BoundedExecutionPlan,
    CapabilityDescriptor,
    CapabilityExecutionResult,
    PlanTask,
    required_metadata_token,
)
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)


CapabilityHandler = Callable[
    [AutonomyGoal, CapabilityDescriptor, Mapping[str, Any]],
    CapabilityExecutionResult,
]


class UnplannableGoalError(ValueError):
    pass


class CapabilityRegistry:
    """Typed, deterministic registry used by the planner instead of prompt branches."""

    def __init__(self, capabilities: Sequence[CapabilityDescriptor] = ()) -> None:
        self._capabilities: dict[str, CapabilityDescriptor] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: CapabilityDescriptor) -> None:
        if capability.capability_id in self._capabilities:
            raise ValueError(
                f"Duplicate capability_id: {capability.capability_id}."
            )
        self._capabilities[capability.capability_id] = capability

    def get(self, capability_id: str) -> CapabilityDescriptor:
        token = required_metadata_token(capability_id, label="capability_id")
        try:
            return self._capabilities[token]
        except KeyError as exc:
            raise KeyError(f"Unknown capability_id: {token}.") from exc

    def all(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            self._capabilities[key]
            for key in sorted(self._capabilities)
        )

    def preflight(self, mode: str) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            capability
            for capability in self.all()
            if capability.mandatory_preflight and mode in capability.allowed_modes
        )

    def postconditions(self, mode: str) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            capability
            for capability in self.all()
            if capability.postcondition and mode in capability.allowed_modes
        )

    def producers(
        self,
        outcome: str,
        *,
        mode: str,
        excluded_capability_ids: set[str],
        allow_production_effects: bool,
    ) -> tuple[CapabilityDescriptor, ...]:
        token = required_metadata_token(outcome, label="requested_outcome")
        candidates = [
            capability
            for capability in self.all()
            if token in capability.provides
            and mode in capability.allowed_modes
            and capability.capability_id not in excluded_capability_ids
            and not capability.postcondition
            and (
                allow_production_effects
                or capability.risk_class != "production_effect"
            )
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda value: (
                    value.priority,
                    value.module_id,
                    value.capability_id,
                ),
            )
        )

    def validate(self) -> None:
        if not self.preflight("offline_evaluation"):
            raise ValueError("A mandatory policy preflight capability is required.")
        if not self.postconditions("offline_evaluation"):
            raise ValueError("A final evidence postcondition capability is required.")
        for capability in self.all():
            for fallback_id in capability.fallback_capability_ids:
                fallback = self.get(fallback_id)
                if not set(capability.provides).intersection(fallback.provides):
                    raise ValueError(
                        "Fallback capabilities must provide at least one common outcome."
                    )


def default_bounded_autonomy_registry() -> CapabilityRegistry:
    """Describe the five modules through outcomes, never prompt/city keywords."""

    registry = CapabilityRegistry(
        (
            CapabilityDescriptor(
                capability_id="module5_policy_preflight",
                module_id=5,
                description="Evaluate authoritative privacy, tenant, mode and approval policy.",
                requires=(),
                provides=("policy_context",),
                risk_class="read_only",
                mandatory_preflight=True,
                priority=0,
            ),
            CapabilityDescriptor(
                capability_id="module1_provider_source_discovery",
                module_id=1,
                description="Discover governed provider sources from the tenant source catalog.",
                requires=("policy_context",),
                provides=("source_inventory", "coverage_assessment"),
                risk_class="read_only",
                max_attempts=2,
                retryable_error_categories=(
                    "provider_unavailable",
                    "timeout_error",
                ),
                fallback_capability_ids=(
                    "module1_public_source_discovery",
                ),
                priority=10,
            ),
            CapabilityDescriptor(
                capability_id="module1_public_source_discovery",
                module_id=1,
                description="Discover licensed public sources for offline or shadow evaluation.",
                requires=("policy_context",),
                provides=("source_inventory", "coverage_assessment"),
                allowed_modes=(
                    "historical_preview",
                    "offline_evaluation",
                    "shadow",
                ),
                risk_class="read_only",
                max_attempts=2,
                retryable_error_categories=("timeout_error",),
                priority=20,
            ),
            CapabilityDescriptor(
                capability_id="module1_privacy_safe_aggregation",
                module_id=1,
                description="Create or validate privacy-safe aggregate source evidence.",
                requires=("policy_context", "source_inventory"),
                provides=("privacy_safe_dataset", "privacy_assessment"),
                risk_class="read_only",
                max_attempts=1,
                priority=30,
            ),
            CapabilityDescriptor(
                capability_id="module2_semantic_retrieval",
                module_id=2,
                description="Select and rank evidence through governed semantic retrieval.",
                requires=("policy_context", "privacy_safe_dataset"),
                provides=("retrieval_evidence",),
                risk_class="read_only",
                max_attempts=2,
                retryable_error_categories=("transient_model_error",),
                priority=40,
            ),
            CapabilityDescriptor(
                capability_id="module3_cohort_strategy",
                module_id=3,
                description="Generate and compare privacy-safe aggregate cohort strategies.",
                requires=("policy_context", "retrieval_evidence"),
                provides=("audience_candidates",),
                risk_class="review_only",
                max_attempts=1,
                priority=50,
            ),
            CapabilityDescriptor(
                capability_id="module4_evolution_review",
                module_id=4,
                description="Compare candidate and historical evidence and propose review-only evolution.",
                requires=("policy_context", "audience_candidates"),
                provides=("evolution_assessment",),
                risk_class="review_only",
                max_attempts=1,
                priority=60,
            ),
            CapabilityDescriptor(
                capability_id="module5_governed_recommendation",
                module_id=5,
                description="Assemble a grounded recommendation under authoritative policy.",
                requires=(
                    "policy_context",
                    "audience_candidates",
                    "evolution_assessment",
                ),
                provides=("governed_recommendation",),
                risk_class="review_only",
                max_attempts=1,
                priority=70,
            ),
            CapabilityDescriptor(
                capability_id="module5_delivery_review",
                module_id=5,
                description="Produce a non-authorizing delivery readiness decision.",
                requires=("policy_context", "governed_recommendation"),
                provides=("delivery_review",),
                risk_class="review_only",
                max_attempts=1,
                priority=80,
            ),
            CapabilityDescriptor(
                capability_id="module5_evidence_critic",
                module_id=5,
                description="Critique evidence completeness and policy consistency without authorizing action.",
                requires=("policy_context",),
                provides=("evidence_critique",),
                risk_class="read_only",
                max_attempts=1,
                priority=10000,
                postcondition=True,
            ),
        )
    )
    registry.validate()
    return registry


class BoundedAutonomyPlanner:
    def __init__(
        self,
        registry: CapabilityRegistry | None = None,
        *,
        allow_production_effects: bool = False,
    ) -> None:
        self.registry = registry or default_bounded_autonomy_registry()
        self.allow_production_effects = bool(allow_production_effects)
        self.registry.validate()

    def plan(
        self,
        goal: AutonomyGoal,
        *,
        excluded_capability_ids: Sequence[str] = (),
    ) -> BoundedExecutionPlan:
        excluded = {
            required_metadata_token(value, label="excluded_capability_id")
            for value in excluded_capability_ids
        }
        selected: dict[str, CapabilityDescriptor] = {}
        producer_for_outcome: dict[str, str] = {}
        resolving: list[str] = []

        def select_outcome(outcome: str) -> str:
            token = required_metadata_token(outcome, label="required_outcome")
            if token in producer_for_outcome:
                return producer_for_outcome[token]
            if token in resolving:
                cycle = " -> ".join((*resolving, token))
                raise UnplannableGoalError(
                    f"Capability dependency cycle detected: {cycle}."
                )
            resolving.append(token)
            candidates = self.registry.producers(
                token,
                mode=goal.execution_mode,
                excluded_capability_ids=excluded,
                allow_production_effects=self.allow_production_effects,
            )
            if not candidates:
                raise UnplannableGoalError(
                    f"No governed capability can provide outcome: {token}."
                )
            capability = candidates[0]
            selected[capability.capability_id] = capability
            for provided in capability.provides:
                producer_for_outcome.setdefault(
                    provided,
                    capability.capability_id,
                )
            for requirement in capability.requires:
                select_outcome(requirement)
            resolving.pop()
            return capability.capability_id

        for outcome in goal.requested_outcomes:
            select_outcome(outcome)
        for capability in self.registry.preflight(goal.execution_mode):
            if capability.capability_id in excluded:
                raise UnplannableGoalError(
                    "A mandatory policy preflight capability cannot be excluded."
                )
            selected[capability.capability_id] = capability
            for provided in capability.provides:
                producer_for_outcome.setdefault(
                    provided,
                    capability.capability_id,
                )

        base_selected_ids = set(selected)
        postconditions = self.registry.postconditions(goal.execution_mode)
        for capability in postconditions:
            if capability.capability_id in excluded:
                raise UnplannableGoalError(
                    "A mandatory evidence postcondition cannot be excluded."
                )
            selected[capability.capability_id] = capability

        dependencies: dict[str, set[str]] = {
            capability_id: set()
            for capability_id in selected
        }
        required_outcomes: dict[str, set[str]] = defaultdict(set)
        for capability_id, capability in selected.items():
            if capability.postcondition:
                dependencies[capability_id].update(base_selected_ids)
                continue
            for requirement in capability.requires:
                producer = producer_for_outcome.get(requirement)
                if producer is None:
                    producer = select_outcome(requirement)
                if producer != capability_id:
                    dependencies[capability_id].add(producer)
                required_outcomes[capability_id].add(requirement)

        ordered_ids, waves = self._topological_order(
            dependencies,
            max_parallel=goal.budget.max_parallel_tasks,
        )
        if len(ordered_ids) > goal.budget.max_tasks:
            raise UnplannableGoalError(
                "The selected capability plan exceeds the goal task budget."
            )
        tasks = tuple(
            PlanTask(
                task_id=f"task_{capability_id}",
                capability_id=capability_id,
                module_id=selected[capability_id].module_id,
                depends_on=tuple(
                    f"task_{dependency}"
                    for dependency in sorted(dependencies[capability_id])
                ),
                required_outcomes=tuple(
                    sorted(required_outcomes[capability_id])
                ),
                risk_class=selected[capability_id].risk_class,
                max_attempts=selected[capability_id].max_attempts,
            )
            for capability_id in ordered_ids
        )
        wave_task_ids = tuple(
            tuple(f"task_{capability_id}" for capability_id in wave)
            for wave in waves
        )
        minimized_goal = goal.to_record()
        plan_payload = {
            "plan_version": BOUNDED_AUTONOMY_PLAN_VERSION,
            "goal": minimized_goal,
            "tasks": [task.to_record() for task in tasks],
            "requested_outcomes": list(goal.requested_outcomes),
            "selected_capability_ids": list(ordered_ids),
            "parallel_waves": [list(wave) for wave in wave_task_ids],
            "excluded_capability_ids": sorted(excluded),
        }
        fingerprint = stable_fingerprint(plan_payload)
        plan = BoundedExecutionPlan(
            plan_id=f"plan-{fingerprint[:24]}",
            plan_version=BOUNDED_AUTONOMY_PLAN_VERSION,
            goal=minimized_goal,
            tasks=tasks,
            requested_outcomes=goal.requested_outcomes,
            selected_capability_ids=tuple(ordered_ids),
            parallel_waves=wave_task_ids,
            excluded_capability_ids=tuple(sorted(excluded)),
            plan_fingerprint=fingerprint,
        )
        self.validate_plan(goal, plan)
        return plan

    def validate_plan(
        self,
        goal: AutonomyGoal,
        plan: BoundedExecutionPlan,
    ) -> BoundedExecutionPlan:
        if plan.plan_version != BOUNDED_AUTONOMY_PLAN_VERSION:
            raise ValueError("Unsupported bounded autonomy plan version.")
        if dict(plan.goal) != goal.to_record():
            raise ValueError("Bounded autonomy plan goal lineage mismatch.")
        if len(plan.tasks) > goal.budget.max_tasks:
            raise ValueError("Bounded autonomy plan exceeds max_tasks.")
        task_ids = {task.task_id for task in plan.tasks}
        if len(task_ids) != len(plan.tasks):
            raise ValueError("Bounded autonomy task identifiers must be unique.")
        selected_ids = {task.capability_id for task in plan.tasks}
        if selected_ids != set(plan.selected_capability_ids):
            raise ValueError("Bounded autonomy selected capability lineage mismatch.")
        for task in plan.tasks:
            capability = self.registry.get(task.capability_id)
            if capability.risk_class == "production_effect" and not (
                self.allow_production_effects
            ):
                raise ValueError("Production-effect capability planning is disabled.")
            if any(dependency not in task_ids for dependency in task.depends_on):
                raise ValueError("Bounded autonomy plan has a missing dependency.")
        provided = {
            outcome
            for capability_id in selected_ids
            for outcome in self.registry.get(capability_id).provides
        }
        if not set(goal.requested_outcomes).issubset(provided):
            raise ValueError("Bounded autonomy plan does not cover the goal outcomes.")
        if not any(
            self.registry.get(value).mandatory_preflight
            for value in selected_ids
        ):
            raise ValueError("Bounded autonomy plan omits policy preflight.")
        if not any(
            self.registry.get(value).postcondition
            for value in selected_ids
        ):
            raise ValueError("Bounded autonomy plan omits evidence criticism.")
        payload = {
            "plan_version": plan.plan_version,
            "goal": dict(plan.goal),
            "tasks": [task.to_record() for task in plan.tasks],
            "requested_outcomes": list(plan.requested_outcomes),
            "selected_capability_ids": list(plan.selected_capability_ids),
            "parallel_waves": [list(wave) for wave in plan.parallel_waves],
            "excluded_capability_ids": list(plan.excluded_capability_ids),
        }
        if stable_fingerprint(payload) != plan.plan_fingerprint:
            raise ValueError("Bounded autonomy plan fingerprint mismatch.")
        return plan

    def _topological_order(
        self,
        dependencies: Mapping[str, set[str]],
        *,
        max_parallel: int,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        remaining = {
            key: set(value)
            for key, value in dependencies.items()
        }
        ordered: list[str] = []
        waves: list[tuple[str, ...]] = []
        while remaining:
            ready = sorted(
                key for key, value in remaining.items() if not value
            )
            if not ready:
                raise UnplannableGoalError(
                    "Capability dependency graph contains a cycle."
                )
            for index in range(0, len(ready), max_parallel):
                waves.append(tuple(ready[index:index + max_parallel]))
            ordered.extend(ready)
            for key in ready:
                remaining.pop(key)
            for value in remaining.values():
                value.difference_update(ready)
        return tuple(ordered), tuple(waves)


@dataclass(frozen=True)
class _ExecutionOutcome:
    status: str
    task_evidence: Mapping[str, Mapping[str, Any]]
    available_outcomes: tuple[str, ...]
    failed_capability_ids: tuple[str, ...]
    total_attempts: int


class BoundedAutonomyExecutor:
    """Execute safe capability handlers while persisting metadata-only evidence."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        *,
        goal: AutonomyGoal,
        plan: BoundedExecutionPlan,
        handlers: Mapping[str, CapabilityHandler],
        attempt_budget: int | None = None,
    ) -> _ExecutionOutcome:
        task_status: dict[str, str] = {}
        task_evidence: dict[str, Mapping[str, Any]] = {}
        available_outcomes: set[str] = set()
        ephemeral_values: dict[str, Any] = {}
        failed_capability_ids: list[str] = []
        total_attempts = 0
        effective_attempt_budget = min(
            goal.budget.max_total_attempts,
            int(attempt_budget or goal.budget.max_total_attempts),
        )
        if effective_attempt_budget < 1:
            raise ValueError("attempt_budget must be positive.")

        for task in plan.tasks:
            capability = self.registry.get(task.capability_id)
            if capability.risk_class == "production_effect":
                result = CapabilityExecutionResult(
                    status="blocked",
                    reason_codes=("production_effect_disabled",),
                    error_category="policy_blocked",
                )
                task_status[task.task_id] = result.status
                task_evidence[task.task_id] = result.to_evidence(attempts=0)
                failed_capability_ids.append(capability.capability_id)
                continue
            if any(
                task_status.get(dependency) != "completed"
                for dependency in task.depends_on
            ):
                result = CapabilityExecutionResult(
                    status="skipped",
                    reason_codes=("dependency_not_completed",),
                    error_category="dependency_failed",
                )
                task_status[task.task_id] = result.status
                task_evidence[task.task_id] = result.to_evidence(attempts=0)
                failed_capability_ids.append(capability.capability_id)
                continue
            if total_attempts >= effective_attempt_budget:
                result = CapabilityExecutionResult(
                    status="blocked",
                    reason_codes=("attempt_budget_exhausted",),
                    error_category="budget_exhausted",
                )
                task_status[task.task_id] = result.status
                task_evidence[task.task_id] = result.to_evidence(attempts=0)
                failed_capability_ids.append(capability.capability_id)
                continue

            handler = handlers.get(capability.capability_id)
            if capability.mandatory_preflight:
                handler = self._policy_preflight
            if capability.postcondition:
                handler = self._evidence_critic(
                    requested_outcomes=set(goal.requested_outcomes),
                    available_outcomes=available_outcomes,
                )
            if handler is None:
                result = CapabilityExecutionResult(
                    status="failed",
                    reason_codes=("capability_handler_unavailable",),
                    error_category="handler_unavailable",
                )
                task_status[task.task_id] = result.status
                task_evidence[task.task_id] = result.to_evidence(attempts=0)
                failed_capability_ids.append(capability.capability_id)
                continue

            attempts = 0
            result: CapabilityExecutionResult | None = None
            while attempts < capability.max_attempts:
                if total_attempts >= effective_attempt_budget:
                    break
                attempts += 1
                total_attempts += 1
                invocation = {
                    "available_outcomes": tuple(sorted(available_outcomes)),
                    "input_values": {
                        requirement: ephemeral_values.get(requirement)
                        for requirement in capability.requires
                    },
                    "attempt": attempts,
                    "shadow_only": True,
                }
                try:
                    result = handler(goal, capability, invocation)
                    if not isinstance(result, CapabilityExecutionResult):
                        raise TypeError(
                            "Capability handlers must return CapabilityExecutionResult."
                        )
                except Exception as exc:  # noqa: BLE001 - converted to bounded metadata
                    category = self._exception_category(exc)
                    result = CapabilityExecutionResult(
                        status="failed",
                        reason_codes=("capability_exception",),
                        error_category=category,
                    )
                if result.status == "completed":
                    break
                if result.error_category not in (
                    capability.retryable_error_categories
                ):
                    break

            if result is None:
                result = CapabilityExecutionResult(
                    status="blocked",
                    reason_codes=("attempt_budget_exhausted",),
                    error_category="budget_exhausted",
                )
            result = self._validate_result(capability, result)
            task_status[task.task_id] = result.status
            task_evidence[task.task_id] = result.to_evidence(
                attempts=attempts
            )
            if result.status == "completed":
                available_outcomes.update(result.provided_outcomes)
                ephemeral_values.update(result.output_values)
            else:
                failed_capability_ids.append(capability.capability_id)

        requested_satisfied = set(goal.requested_outcomes).issubset(
            available_outcomes
        )
        postconditions_completed = all(
            task_status.get(task.task_id) == "completed"
            for task in plan.tasks
            if self.registry.get(task.capability_id).postcondition
        )
        if any(value == "blocked" for value in task_status.values()):
            status = "blocked"
        elif not requested_satisfied or not postconditions_completed:
            status = "failed"
        else:
            status = "completed"
        return _ExecutionOutcome(
            status=status,
            task_evidence=task_evidence,
            available_outcomes=tuple(sorted(available_outcomes)),
            failed_capability_ids=tuple(dict.fromkeys(failed_capability_ids)),
            total_attempts=total_attempts,
        )

    def _policy_preflight(
        self,
        goal: AutonomyGoal,
        capability: CapabilityDescriptor,
        invocation: Mapping[str, Any],
    ) -> CapabilityExecutionResult:
        del capability, invocation
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("policy_context",),
            reason_codes=("bounded_shadow_policy_applied",),
            metrics={
                "manual_approval_required": True,
                "production_effects_enabled": False,
            },
            output_values={
                "policy_context": {
                    "tenant_id": goal.tenant_id,
                    "execution_mode": goal.execution_mode,
                    "production_effects_enabled": False,
                }
            },
        )

    def _evidence_critic(
        self,
        *,
        requested_outcomes: set[str],
        available_outcomes: set[str],
    ) -> CapabilityHandler:
        def critic(
            goal: AutonomyGoal,
            capability: CapabilityDescriptor,
            invocation: Mapping[str, Any],
        ) -> CapabilityExecutionResult:
            del goal, capability, invocation
            missing = requested_outcomes.difference(available_outcomes)
            if missing:
                return CapabilityExecutionResult(
                    status="blocked",
                    reason_codes=("requested_outcome_missing",),
                    metrics={"missing_outcome_count": len(missing)},
                    error_category="evidence_incomplete",
                )
            return CapabilityExecutionResult(
                status="completed",
                provided_outcomes=("evidence_critique",),
                reason_codes=("evidence_complete",),
                metrics={
                    "requested_outcome_count": len(requested_outcomes),
                    "missing_outcome_count": 0,
                },
                output_values={"evidence_critique": {"complete": True}},
            )

        return critic

    def _validate_result(
        self,
        capability: CapabilityDescriptor,
        result: CapabilityExecutionResult,
    ) -> CapabilityExecutionResult:
        if not set(result.provided_outcomes).issubset(capability.provides):
            raise ValueError(
                "Capability result contains an undeclared provided outcome."
            )
        if not set(result.output_values).issubset(result.provided_outcomes):
            raise ValueError(
                "Capability output values must match declared provided outcomes."
            )
        assert_no_raw_identifier_fields(result.output_values)
        if result.status != "completed" and result.provided_outcomes:
            raise ValueError("Non-completed capabilities cannot provide outcomes.")
        return result

    def _exception_category(self, exc: Exception) -> str:
        explicit = getattr(exc, "error_category", None)
        if explicit:
            try:
                return required_metadata_token(
                    explicit,
                    label="error_category",
                )
            except ValueError:
                return "capability_error"
        name = re.sub(
            r"(?<!^)(?=[A-Z])",
            "_",
            type(exc).__name__,
        ).lower()
        try:
            return required_metadata_token(name, label="error_category")
        except ValueError:
            return "capability_error"


class ProductionBoundedAutonomyService:
    """Plan, execute, criticize and replan in a non-authorizing shadow boundary."""

    def __init__(
        self,
        registry: CapabilityRegistry | None = None,
    ) -> None:
        self.registry = registry or default_bounded_autonomy_registry()
        self.planner = BoundedAutonomyPlanner(self.registry)
        self.executor = BoundedAutonomyExecutor(self.registry)

    def run(
        self,
        *,
        goal: AutonomyGoal,
        handlers: Mapping[str, CapabilityHandler],
    ) -> dict[str, Any]:
        plan = self.planner.plan(goal)
        outcome = self.executor.execute(
            goal=goal,
            plan=plan,
            handlers=handlers,
        )
        plan_fingerprints = [plan.plan_fingerprint]
        execution_history = [
            {
                "plan_fingerprint": plan.plan_fingerprint,
                "execution_status": outcome.status,
                "total_attempts": outcome.total_attempts,
                "failed_capability_ids": list(
                    outcome.failed_capability_ids
                ),
            }
        ]
        attempts_across_plans = outcome.total_attempts
        excluded: list[str] = []
        replan_count = 0

        while (
            outcome.status == "failed"
            and replan_count < goal.budget.max_replans
            and outcome.failed_capability_ids
            and attempts_across_plans < goal.budget.max_total_attempts
        ):
            fallback_eligible = [
                capability_id
                for capability_id in outcome.failed_capability_ids
                if self.registry.get(capability_id).fallback_capability_ids
            ]
            if not fallback_eligible:
                break
            excluded.extend(fallback_eligible)
            try:
                plan = self.planner.plan(
                    goal,
                    excluded_capability_ids=excluded,
                )
            except UnplannableGoalError:
                break
            replan_count += 1
            plan_fingerprints.append(plan.plan_fingerprint)
            outcome = self.executor.execute(
                goal=goal,
                plan=plan,
                handlers=handlers,
                attempt_budget=(
                    goal.budget.max_total_attempts
                    - attempts_across_plans
                ),
            )
            attempts_across_plans += outcome.total_attempts
            execution_history.append(
                {
                    "plan_fingerprint": plan.plan_fingerprint,
                    "execution_status": outcome.status,
                    "total_attempts": outcome.total_attempts,
                    "failed_capability_ids": list(
                        outcome.failed_capability_ids
                    ),
                }
            )

        report = {
            "status": (
                "engineering_preview_ready"
                if outcome.status == "completed"
                else "engineering_preview_blocked"
            ),
            "policy_version": BOUNDED_AUTONOMY_EVIDENCE_VERSION,
            "goal": goal.to_record(),
            "planning": {
                "final_plan": plan.to_record(),
                "plan_fingerprints": plan_fingerprints,
                "replan_count": replan_count,
                "excluded_capability_ids": sorted(set(excluded)),
                "prompt_specific_routing_used": False,
                "production_effect_capabilities_selected": False,
            },
            "execution": {
                "execution_status": outcome.status,
                "task_evidence": dict(outcome.task_evidence),
                "available_outcomes": list(outcome.available_outcomes),
                "failed_capability_ids": list(outcome.failed_capability_ids),
                "total_attempts": attempts_across_plans,
                "execution_history": execution_history,
                "requested_outcomes_satisfied": set(
                    goal.requested_outcomes
                ).issubset(outcome.available_outcomes),
            },
            "safety": {
                "shadow_only": True,
                "prompt_content_stored": False,
                "tool_arguments_stored": False,
                "tool_results_stored": False,
                "raw_identifiers_returned": False,
                "automatic_approval_performed": False,
                "automatic_mutation_performed": False,
                "production_effect_performed": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "manual_approval_required": True,
            },
        }
        report["bounded_autonomy_report_fingerprint"] = stable_fingerprint(
            report
        )
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(report)))
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") not in {
            "engineering_preview_ready",
            "engineering_preview_blocked",
        }:
            raise ValueError("Unsupported bounded autonomy report status.")
        if payload.get("policy_version") != BOUNDED_AUTONOMY_EVIDENCE_VERSION:
            raise ValueError("Unsupported bounded autonomy evidence version.")
        goal = payload.get("goal")
        planning = payload.get("planning")
        execution = payload.get("execution")
        safety = payload.get("safety")
        if not all(
            isinstance(value, Mapping)
            for value in (goal, planning, execution, safety)
        ):
            raise ValueError("Bounded autonomy evidence sections are required.")
        if "objective" in goal or "prompt" in json.dumps(goal).lower():
            raise ValueError("Prompt content cannot be stored in autonomy evidence.")
        if planning.get("prompt_specific_routing_used") is not False:
            raise ValueError("Prompt-specific routing is prohibited.")
        if planning.get("production_effect_capabilities_selected") is not False:
            raise ValueError("Production-effect capability selection is prohibited.")
        final_plan = planning.get("final_plan")
        if not isinstance(final_plan, Mapping):
            raise ValueError("Final bounded autonomy plan evidence is required.")
        plan_payload = {
            "plan_version": final_plan.get("plan_version"),
            "goal": final_plan.get("goal"),
            "tasks": final_plan.get("tasks"),
            "requested_outcomes": final_plan.get("requested_outcomes"),
            "selected_capability_ids": final_plan.get(
                "selected_capability_ids"
            ),
            "parallel_waves": final_plan.get("parallel_waves"),
            "excluded_capability_ids": final_plan.get(
                "excluded_capability_ids"
            ),
        }
        final_plan_fingerprint = str(
            final_plan.get("plan_fingerprint") or ""
        )
        if stable_fingerprint(plan_payload) != final_plan_fingerprint:
            raise ValueError("Final bounded autonomy plan fingerprint mismatch.")
        if final_plan.get("plan_id") != (
            f"plan-{final_plan_fingerprint[:24]}"
        ):
            raise ValueError("Final bounded autonomy plan identifier mismatch.")
        plan_fingerprints = planning.get("plan_fingerprints") or []
        if (
            not plan_fingerprints
            or plan_fingerprints[-1] != final_plan_fingerprint
            or len(plan_fingerprints)
            != int(planning.get("replan_count") or 0) + 1
        ):
            raise ValueError("Bounded autonomy plan lineage is inconsistent.")
        selected = final_plan.get("selected_capability_ids") or []
        if len(selected) != len(set(selected)):
            raise ValueError("Selected capability evidence must be unique.")
        if int(planning.get("replan_count") or 0) > int(
            goal.get("budget", {}).get("max_replans") or 0
        ):
            raise ValueError("Bounded autonomy replan budget was exceeded.")
        if int(execution.get("total_attempts") or 0) > int(
            goal.get("budget", {}).get("max_total_attempts") or 0
        ):
            raise ValueError("Bounded autonomy attempt budget was exceeded.")
        if execution.get("execution_status") == "completed" and (
            execution.get("requested_outcomes_satisfied") is not True
        ):
            raise ValueError("Completed autonomy evidence must satisfy the goal.")
        required_false = (
            "prompt_content_stored",
            "tool_arguments_stored",
            "tool_results_stored",
            "raw_identifiers_returned",
            "automatic_approval_performed",
            "automatic_mutation_performed",
            "production_effect_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        )
        for key in required_false:
            if safety.get(key) is not False:
                raise ValueError(f"Unsafe bounded autonomy safety field: {key}.")
        if safety.get("shadow_only") is not True:
            raise ValueError("Bounded autonomy evidence must remain shadow-only.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Bounded autonomy requires manual approval.")
        supplied_fingerprint = payload.pop(
            "bounded_autonomy_report_fingerprint",
            None,
        )
        if stable_fingerprint(payload) != supplied_fingerprint:
            raise ValueError("Bounded autonomy report fingerprint mismatch.")
        payload["bounded_autonomy_report_fingerprint"] = supplied_fingerprint
        return payload
