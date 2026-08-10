from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from app.models.production_bounded_autonomy_contracts import (
    AutonomyGoal,
    CapabilityDescriptor,
    CapabilityExecutionResult,
)
from app.models.production_bounded_autonomy_shadow_contracts import (
    BOUNDED_AUTONOMY_SHADOW_COMPARISON_VERSION,
    AutonomousShadowDecision,
    CanonicalOrchestrationObservation,
    ShadowDivergencePolicy,
)
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)
from app.services.production_bounded_autonomy_service import (
    CapabilityHandler,
    ProductionBoundedAutonomyService,
    default_bounded_autonomy_registry,
)

_PRIVACY_BLOCK_MODES = {
    "privacy_identifier_request_blocked",
}
_BYPASS_BLOCK_MODES = {
    "approval_bypass_attempt_blocked",
}
_CLARIFICATION_MODES = {"broad_location_no_export"}
_EXISTING_APPROVAL_MODES = {"export_action_requires_existing_audience"}
_COVERAGE_BLOCK_MODES = {
    "location_category_gap_no_export",
}
_STALE_STATUSES = {"stale", "expired", "outdated"}
_RAW_IDENTIFIER_KEYS = {
    "maid",
    "maids",
    "raw_maid",
    "raw_maids",
    "device_id",
    "device_ids",
    "advertising_id",
    "advertising_ids",
    "raw_identifier",
    "raw_identifiers",
    "individual_id",
    "individual_ids",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _token(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(
        char if char.isalnum() or char in "_:-." else "_"
        for char in text
    ).strip("_")
    return (cleaned or default)[:128]


def _count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(parsed, 1_000_000_000))


def _first_count(*values: Any) -> int:
    for value in values:
        if value is not None:
            return _count(value)
    return 0


def _truthy_flag(payload: Any, names: set[str]) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = _token(key)
            if normalized in names:
                if isinstance(value, bool) and value:
                    return True
                if value not in (None, False, 0, "", [], {}, ()):
                    return True
            if _truthy_flag(value, names):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(_truthy_flag(value, names) for value in payload)
    return False


class LegacyOrchestratorObservationAdapter:
    """Convert a real legacy result into bounded facts without copying payloads."""

    def __init__(
        self,
        decision_service: AudienceSupervisorDecisionService | None = None,
    ) -> None:
        self.decision_service = (
            decision_service or AudienceSupervisorDecisionService()
        )

    def adapt(
        self,
        *,
        goal: AutonomyGoal,
        legacy_result: Mapping[str, Any],
    ) -> CanonicalOrchestrationObservation:
        result = dict(legacy_result)
        decision = self.decision_service.decide(result)
        safe_export = _mapping(result.get("safe_export"))
        prompt_filter = _mapping(result.get("prompt_filter_report"))
        v2 = _mapping(
            result.get("v2_autonomous")
            or result.get("v2")
            or result.get("autonomous_v2")
            or result.get("v2_result")
        )
        embedding = _mapping(v2.get("embedding_manifest"))
        legacy_embedding = _mapping(result.get("embedding"))
        freshness = _mapping(
            result.get("source_freshness")
            or safe_export.get("source_freshness")
            or v2.get("data_freshness")
        )
        source_rows_value = (
            result.get("source_rows_checked")
            if result.get("source_rows_checked") is not None
            else (
                result.get("source_rows")
                if result.get("source_rows") is not None
                else v2.get("source_rows_checked")
            )
        )
        source_evaluated = source_rows_value is not None
        filter_mode = _token(prompt_filter.get("filter_mode"), "not_reported")
        approval_status = _token(
            result.get("approval_status")
            or safe_export.get("approval_status")
            or safe_export.get("status"),
            "not_reported",
        )
        freshness_status = _token(
            result.get("freshness_status")
            or freshness.get("freshness_status")
            or freshness.get("status")
            or v2.get("freshness"),
            "not_evaluated" if not source_evaluated else "unknown",
        )
        raw_identifier_keys_present = _truthy_flag(
            result,
            _RAW_IDENTIFIER_KEYS,
        )
        raw_identifier_flag = _truthy_flag(
            result,
            {
                "raw_identifiers_returned",
                "raw_identifiers_exported",
                "individual_user_data_exported",
                "individual_level_data_returned",
            },
        )
        production_effect = _truthy_flag(
            result,
            {
                "activation_or_export_performed",
                "audience_activated_or_exported",
                "production_effect_performed",
            },
        )
        downstream = bool(
            result.get("downstream_export_enabled", False)
            or safe_export.get("downstream_export_enabled", False)
            or decision.get("downstream_export_enabled", False)
        )
        run_id = str(result.get("run_id") or goal.request_id)
        reason_codes = tuple(
            dict.fromkeys(
                _token(value)
                for value in (decision.get("reason_codes") or ())
                if str(value or "").strip()
            )
        )
        return CanonicalOrchestrationObservation(
            tenant_id=goal.tenant_id,
            request_id=goal.request_id,
            run_id=run_id,
            pipeline_status=_token(result.get("status"), "unknown"),
            approval_status=approval_status,
            filter_mode=filter_mode,
            decision_route=_token(decision.get("route"), "blocked"),
            decision_stage=_token(decision.get("stage"), "supervisor"),
            freshness_status=freshness_status,
            source_evaluated=source_evaluated,
            source_row_count=_count(source_rows_value),
            vector_count=_first_count(
                result.get("vector_count"),
                v2.get("vector_count"),
                embedding.get("vector_count"),
                legacy_embedding.get("vector_count"),
            ),
            ranked_match_count=_first_count(
                result.get("ranked_match_count"),
                v2.get("ranked_match_count"),
                v2.get("ranked_matches"),
            ),
            selected_cohort_count=_first_count(
                result.get("prompt_selected_cohorts"),
                result.get("selected_cohort_count"),
                safe_export.get("exported_cohorts"),
            ),
            prepared_candidate_count=_first_count(
                result.get("prepared_audience_candidates"),
                result.get("prepared_candidate_count"),
                result.get("created_audiences"),
                safe_export.get("exported_cohorts"),
            ),
            terminal=bool(decision.get("terminal", True)),
            approval_required=bool(
                decision.get("approval_required", True)
            ),
            downstream_export_enabled=downstream,
            raw_identifiers_returned=(
                raw_identifier_keys_present or raw_identifier_flag
            ),
            activation_or_export_performed=(production_effect or downstream),
            reason_codes=reason_codes,
        )


class RealServiceCapabilityAdapterFactory:
    """Bind the generic kernel to sanitized facts from the real orchestrator."""

    def __init__(self, observation: CanonicalOrchestrationObservation) -> None:
        self.observation = observation
        self.autonomous_decision: AutonomousShadowDecision | None = None

    def handlers(self) -> Mapping[str, CapabilityHandler]:
        handlers: dict[str, CapabilityHandler] = {}
        for descriptor in default_bounded_autonomy_registry().all():
            if descriptor.mandatory_preflight or descriptor.postcondition:
                continue
            handlers[descriptor.capability_id] = self._handler_for(
                descriptor.capability_id
            )
        return handlers

    def _handler_for(self, capability_id: str) -> CapabilityHandler:
        dispatch = {
            "module1_provider_source_discovery": self._source_discovery,
            "module1_public_source_discovery": self._source_discovery,
            "module1_privacy_safe_aggregation": self._privacy_assessment,
            "module2_semantic_retrieval": self._retrieval_assessment,
            "module3_cohort_strategy": self._cohort_assessment,
            "module4_evolution_review": self._evolution_assessment,
            "module5_governed_recommendation": self._recommendation,
            "module5_delivery_review": self._delivery_review,
        }
        try:
            return dispatch[capability_id]
        except KeyError as exc:
            raise ValueError(
                f"No real-service adapter exists for {capability_id}."
            ) from exc

    def _complete(
        self,
        capability: CapabilityDescriptor,
        *,
        metrics: Mapping[str, int | float | bool],
        values: Mapping[str, Any],
        reasons: tuple[str, ...],
    ) -> CapabilityExecutionResult:
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=capability.provides,
            reason_codes=reasons,
            metrics=metrics,
            output_values=values,
        )

    def _source_discovery(self, _goal, capability, _invocation):
        return self._complete(
            capability,
            metrics={
                "source_evaluated": self.observation.source_evaluated,
                "source_row_count": self.observation.source_row_count,
            },
            values={
                "source_inventory": {
                    "evaluated": self.observation.source_evaluated,
                    "row_count": self.observation.source_row_count,
                },
                "coverage_assessment": {
                    "selected_cohort_count": (
                        self.observation.selected_cohort_count
                    ),
                },
            },
            reasons=("real_orchestrator_source_observed",),
        )

    def _privacy_assessment(self, _goal, capability, _invocation):
        safe = not (
            self.observation.raw_identifiers_returned
            or self.observation.activation_or_export_performed
        )
        return self._complete(
            capability,
            metrics={
                "privacy_safe": safe,
                "raw_identifiers_returned": (
                    self.observation.raw_identifiers_returned
                ),
                "production_effect_observed": (
                    self.observation.activation_or_export_performed
                ),
            },
            values={
                "privacy_safe_dataset": {"safe": safe},
                "privacy_assessment": {"safe": safe},
            },
            reasons=(
                "privacy_invariants_observed"
                if safe
                else "unsafe_legacy_output_observed"
            ,),
        )

    def _retrieval_assessment(self, _goal, capability, _invocation):
        return self._complete(
            capability,
            metrics={
                "vector_count": self.observation.vector_count,
                "ranked_match_count": self.observation.ranked_match_count,
            },
            values={
                "retrieval_evidence": {
                    "vector_count": self.observation.vector_count,
                    "ranked_match_count": self.observation.ranked_match_count,
                }
            },
            reasons=("real_retrieval_metrics_observed",),
        )

    def _cohort_assessment(self, _goal, capability, _invocation):
        return self._complete(
            capability,
            metrics={
                "selected_cohort_count": (
                    self.observation.selected_cohort_count
                ),
                "prepared_candidate_count": (
                    self.observation.prepared_candidate_count
                ),
            },
            values={
                "audience_candidates": {
                    "selected_cohort_count": (
                        self.observation.selected_cohort_count
                    ),
                    "prepared_candidate_count": (
                        self.observation.prepared_candidate_count
                    ),
                }
            },
            reasons=("real_cohort_metrics_observed",),
        )

    def _evolution_assessment(self, _goal, capability, _invocation):
        return self._complete(
            capability,
            metrics={"review_only": True, "mutation_performed": False},
            values={
                "evolution_assessment": {
                    "review_only": True,
                    "mutation_performed": False,
                }
            },
            reasons=("historical_review_only",),
        )

    def _recommendation(self, _goal, capability, _invocation):
        decision = self._independent_decision()
        self.autonomous_decision = decision
        return self._complete(
            capability,
            metrics={
                "terminal": decision.terminal,
                "approval_required": decision.approval_required,
                "selected_cohort_count": decision.selected_cohort_count,
                "prepared_candidate_count": decision.prepared_candidate_count,
            },
            values={
                "governed_recommendation": decision.to_record(),
            },
            reasons=decision.reason_codes,
        )

    def _delivery_review(self, _goal, capability, _invocation):
        decision = self.autonomous_decision or self._independent_decision()
        return self._complete(
            capability,
            metrics={
                "manual_approval_required": True,
                "downstream_export_enabled": False,
                "activation_or_export_performed": False,
            },
            values={
                "delivery_review": {
                    "route": decision.route,
                    "manual_approval_required": True,
                    "downstream_export_enabled": False,
                }
            },
            reasons=("shadow_delivery_review_only",),
        )

    def _independent_decision(self) -> AutonomousShadowDecision:
        value = self.observation
        route = "pending_approval"
        stage = "approval"
        terminal = False
        reasons = ("manual_approval_required",)

        if value.raw_identifiers_returned:
            route, stage, terminal, reasons = (
                "blocked",
                "privacy",
                True,
                ("privacy_guardrail_blocked",),
            )
        elif value.activation_or_export_performed:
            route, stage, terminal, reasons = (
                "blocked",
                "governance",
                True,
                ("unexpected_production_effect_observed",),
            )
        elif value.filter_mode in _PRIVACY_BLOCK_MODES:
            route, stage, terminal, reasons = (
                "blocked",
                "privacy",
                True,
                ("privacy_guardrail_blocked",),
            )
        elif value.filter_mode in _BYPASS_BLOCK_MODES:
            route, stage, terminal, reasons = (
                "blocked",
                "approval",
                True,
                ("approval_bypass_attempt_blocked",),
            )
        elif value.pipeline_status in {"failed", "error"}:
            route, stage, terminal, reasons = (
                "failed",
                "pipeline",
                True,
                ("pipeline_failed",),
            )
        elif value.freshness_status in _STALE_STATUSES or "stale" in (
            value.freshness_status
        ):
            route, stage, terminal, reasons = (
                "blocked",
                "freshness",
                True,
                ("source_refresh_required",),
            )
        elif value.filter_mode in _CLARIFICATION_MODES:
            route, stage, terminal, reasons = (
                "needs_clarification",
                "intent",
                False,
                ("clarification_required",),
            )
        elif value.filter_mode in _EXISTING_APPROVAL_MODES:
            route, stage, terminal, reasons = (
                "needs_existing_approval",
                "approval",
                False,
                ("existing_approved_audience_required",),
            )
        elif value.approval_status == "blocked_requested_quality_unmet":
            route, stage, terminal, reasons = (
                "blocked",
                "coverage",
                True,
                ("requested_quality_unmet",),
            )
        elif value.approval_status == "blocked_v2_failure":
            route, stage, terminal, reasons = (
                "blocked",
                "v2_intelligence",
                True,
                ("v2_failure",),
            )
        elif (
            value.filter_mode in _COVERAGE_BLOCK_MODES
            or value.approval_status == "blocked_no_safe_exact_match"
            or value.selected_cohort_count == 0
        ):
            route, stage, terminal, reasons = (
                "blocked",
                "coverage",
                True,
                ("no_safe_exact_match",),
            )
        elif value.approval_status.startswith("blocked_"):
            route, stage, terminal, reasons = (
                "blocked",
                "guardrail",
                True,
                (value.approval_status,),
            )
        elif value.pipeline_status in {"completed", "success", "succeeded"} and (
            not value.approval_required
        ):
            route, stage, terminal, reasons = (
                "completed_safe",
                "completed",
                True,
                ("completed_without_delivery",),
            )

        return AutonomousShadowDecision(
            route=route,
            stage=stage,
            freshness_status=value.freshness_status,
            source_evaluated=value.source_evaluated,
            source_row_count=value.source_row_count,
            vector_count=value.vector_count,
            ranked_match_count=value.ranked_match_count,
            selected_cohort_count=value.selected_cohort_count,
            prepared_candidate_count=value.prepared_candidate_count,
            terminal=terminal,
            approval_required=True,
            downstream_export_enabled=False,
            raw_identifiers_returned=False,
            activation_or_export_performed=False,
            reason_codes=reasons,
        )


class ProductionBoundedAutonomyShadowComparisonService:
    """Dual-run comparator that can never change the authoritative route."""

    def __init__(
        self,
        *,
        bounded_service: ProductionBoundedAutonomyService | None = None,
        observation_adapter: LegacyOrchestratorObservationAdapter | None = None,
        policy: ShadowDivergencePolicy | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.bounded_service = bounded_service or ProductionBoundedAutonomyService()
        self.observation_adapter = (
            observation_adapter or LegacyOrchestratorObservationAdapter()
        )
        self.policy = policy or self._policy_from_environment(
            environment if environment is not None else os.environ
        )

    def _policy_from_environment(
        self,
        environment: Mapping[str, str],
    ) -> ShadowDivergencePolicy:
        def configured_delta(key: str) -> int:
            raw = str(environment.get(key) or "0").strip()
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer.") from exc

        return ShadowDivergencePolicy(
            max_source_row_delta=configured_delta(
                "MODULE5_BOUNDED_AUTONOMY_MAX_SOURCE_ROW_DELTA"
            ),
            max_vector_count_delta=configured_delta(
                "MODULE5_BOUNDED_AUTONOMY_MAX_VECTOR_COUNT_DELTA"
            ),
            max_ranked_match_delta=configured_delta(
                "MODULE5_BOUNDED_AUTONOMY_MAX_RANKED_MATCH_DELTA"
            ),
            max_selected_cohort_delta=configured_delta(
                "MODULE5_BOUNDED_AUTONOMY_MAX_SELECTED_COHORT_DELTA"
            ),
            max_prepared_candidate_delta=configured_delta(
                "MODULE5_BOUNDED_AUTONOMY_MAX_PREPARED_CANDIDATE_DELTA"
            ),
        )

    def run(
        self,
        *,
        goal: AutonomyGoal,
        legacy_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if goal.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "shadow",
        }:
            raise ValueError("Dual-run comparison is prohibited in production mode.")
        observation = self.observation_adapter.adapt(
            goal=goal,
            legacy_result=legacy_result,
        )
        adapters = RealServiceCapabilityAdapterFactory(observation)
        bounded_report = self.bounded_service.run(
            goal=goal,
            handlers=adapters.handlers(),
        )
        autonomous = adapters.autonomous_decision
        if autonomous is None:
            autonomous = AutonomousShadowDecision(
                route="blocked",
                stage="autonomy",
                freshness_status=observation.freshness_status,
                source_evaluated=observation.source_evaluated,
                source_row_count=0,
                vector_count=0,
                ranked_match_count=0,
                selected_cohort_count=0,
                prepared_candidate_count=0,
                terminal=True,
                approval_required=True,
                reason_codes=("autonomy_decision_unavailable",),
            )
        comparison = self._compare(observation, autonomous)
        critical_count = sum(
            1 for value in comparison.values() if value["critical"] and not value["match"]
        )
        divergence_count = sum(
            1 for value in comparison.values() if not value["match"]
        )
        bounded_ready = bounded_report.get("status") == "engineering_preview_ready"
        eligible = bool(
            bounded_ready
            and critical_count == 0
            and divergence_count == 0
            and not observation.raw_identifiers_returned
            and not observation.activation_or_export_performed
        )
        report = {
            "status": (
                "engineering_preview_ready"
                if eligible
                else "engineering_preview_blocked"
            ),
            "policy_version": BOUNDED_AUTONOMY_SHADOW_COMPARISON_VERSION,
            "tenant_id": goal.tenant_id,
            "request_id": goal.request_id,
            "goal_id": goal.goal_id,
            "objective_sha256": goal.objective_sha256,
            "legacy_observation": observation.to_record(),
            "autonomous_observation": autonomous.to_record(),
            "comparison": {
                "checks": comparison,
                "divergence_count": divergence_count,
                "critical_divergence_count": critical_count,
                "policy": dict(self.policy.to_record()),
            },
            "lineage": {
                "bounded_autonomy_report_fingerprint": bounded_report[
                    "bounded_autonomy_report_fingerprint"
                ],
                "final_plan_fingerprint": bounded_report["planning"][
                    "final_plan"
                ]["plan_fingerprint"],
            },
            "cutover": {
                "eligible_for_human_review": eligible,
                "automatic_cutover_performed": False,
                "production_routing_changed": False,
                "authoritative_legacy_route_preserved": True,
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
        report["shadow_comparison_report_fingerprint"] = stable_fingerprint(report)
        return self.validate_report(report)

    def _compare(
        self,
        legacy: CanonicalOrchestrationObservation,
        autonomous: AutonomousShadowDecision,
    ) -> dict[str, dict[str, Any]]:
        def exact(left: Any, right: Any, *, critical: bool) -> dict[str, Any]:
            return {
                "match": left == right,
                "critical": critical,
            }

        def delta(
            left: int,
            right: int,
            threshold: int,
            *,
            critical: bool,
        ) -> dict[str, Any]:
            difference = abs(int(left) - int(right))
            return {
                "match": difference <= threshold,
                "critical": critical,
                "absolute_delta": difference,
                "allowed_delta": threshold,
            }

        return {
            "route": exact(
                legacy.decision_route,
                autonomous.route,
                critical=self.policy.require_route_match,
            ),
            "stage": exact(
                legacy.decision_stage,
                autonomous.stage,
                critical=self.policy.require_stage_match,
            ),
            "terminal": exact(
                legacy.terminal,
                autonomous.terminal,
                critical=self.policy.require_terminal_match,
            ),
            "freshness": exact(
                legacy.freshness_status,
                autonomous.freshness_status,
                critical=self.policy.require_freshness_match,
            ),
            "source_evaluated": exact(
                legacy.source_evaluated,
                autonomous.source_evaluated,
                critical=True,
            ),
            "source_rows": delta(
                legacy.source_row_count,
                autonomous.source_row_count,
                self.policy.max_source_row_delta,
                critical=False,
            ),
            "vector_count": delta(
                legacy.vector_count,
                autonomous.vector_count,
                self.policy.max_vector_count_delta,
                critical=False,
            ),
            "ranked_matches": delta(
                legacy.ranked_match_count,
                autonomous.ranked_match_count,
                self.policy.max_ranked_match_delta,
                critical=False,
            ),
            "selected_cohorts": delta(
                legacy.selected_cohort_count,
                autonomous.selected_cohort_count,
                self.policy.max_selected_cohort_delta,
                critical=True,
            ),
            "prepared_candidates": delta(
                legacy.prepared_candidate_count,
                autonomous.prepared_candidate_count,
                self.policy.max_prepared_candidate_delta,
                critical=True,
            ),
            "downstream_disabled": exact(
                legacy.downstream_export_enabled,
                autonomous.downstream_export_enabled,
                critical=True,
            ),
            "raw_identifiers_absent": exact(
                legacy.raw_identifiers_returned,
                autonomous.raw_identifiers_returned,
                critical=True,
            ),
            "production_effect_absent": exact(
                legacy.activation_or_export_performed,
                autonomous.activation_or_export_performed,
                critical=True,
            ),
        }

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(report)))
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") not in {
            "engineering_preview_ready",
            "engineering_preview_blocked",
        }:
            raise ValueError("Unsupported shadow comparison status.")
        if payload.get("policy_version") != (
            BOUNDED_AUTONOMY_SHADOW_COMPARISON_VERSION
        ):
            raise ValueError("Unsupported shadow comparison policy version.")
        for section in (
            "legacy_observation",
            "autonomous_observation",
            "comparison",
            "lineage",
            "cutover",
            "safety",
        ):
            if not isinstance(payload.get(section), Mapping):
                raise TypeError(f"Shadow comparison {section} is required.")
        serialized = json.dumps(payload).lower()
        if '"prompt"' in serialized or '"objective"' in serialized:
            raise ValueError("Prompt/objective content cannot enter comparison evidence.")
        comparison = payload["comparison"]
        checks = comparison.get("checks")
        if not isinstance(checks, Mapping) or not checks:
            raise ValueError("Shadow divergence checks are required.")
        calculated_divergences = sum(
            1 for value in checks.values()
            if isinstance(value, Mapping) and value.get("match") is not True
        )
        calculated_critical = sum(
            1 for value in checks.values()
            if isinstance(value, Mapping)
            and value.get("critical") is True
            and value.get("match") is not True
        )
        if calculated_divergences != comparison.get("divergence_count"):
            raise ValueError("Shadow divergence count mismatch.")
        if calculated_critical != comparison.get("critical_divergence_count"):
            raise ValueError("Critical shadow divergence count mismatch.")
        cutover = payload["cutover"]
        if cutover.get("automatic_cutover_performed") is not False:
            raise ValueError("Automatic cutover is prohibited.")
        if cutover.get("production_routing_changed") is not False:
            raise ValueError("Shadow comparison cannot change production routing.")
        if cutover.get("authoritative_legacy_route_preserved") is not True:
            raise ValueError("The authoritative legacy route must be preserved.")
        if cutover.get("eligible_for_human_review") is True and (
            calculated_divergences or calculated_critical
        ):
            raise ValueError("Divergent evidence cannot be cutover-review eligible.")
        safety = payload["safety"]
        if safety.get("shadow_only") is not True:
            raise ValueError("Dual-run evidence must remain shadow-only.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Manual approval remains mandatory.")
        for key in (
            "prompt_content_stored",
            "tool_arguments_stored",
            "tool_results_stored",
            "raw_identifiers_returned",
            "automatic_approval_performed",
            "automatic_mutation_performed",
            "production_effect_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(key) is not False:
                raise ValueError(f"Unsafe shadow comparison field: {key}.")
        supplied = payload.pop("shadow_comparison_report_fingerprint", None)
        if stable_fingerprint(payload) != supplied:
            raise ValueError("Shadow comparison fingerprint mismatch.")
        payload["shadow_comparison_report_fingerprint"] = supplied
        return payload
