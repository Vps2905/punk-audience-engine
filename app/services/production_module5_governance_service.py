from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    stable_fingerprint,
)
from app.models.production_module5_governance_contracts import (
    MODULE5_CIRCUIT_BREAKER_POLICY_VERSION,
    MODULE5_EXECUTION_POLICY_VERSION,
    MODULE5_HUMAN_REVIEW_POLICY_VERSION,
    MODULE5_POLICY_POLICY_VERSION,
    MODULE5_RECOVERY_POLICY_VERSION,
    Module5AgentExecutionRequest,
    Module5CircuitBreakerRequest,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


_FALSE_SAFETY_FIELDS = (
    "raw_identifiers_read",
    "raw_identifiers_stored",
    "raw_identifiers_returned",
    "prompt_content_stored",
    "tool_arguments_stored",
    "tool_results_stored",
    "agent_execution_triggered",
    "automatic_retry_triggered",
    "automatic_mutation_performed",
    "automatic_approval_performed",
    "production_routing_enabled",
    "activation_or_export_performed",
    "downstream_export_enabled",
)


def _safety() -> dict[str, Any]:
    return {
        "raw_identifiers_read": False,
        "raw_identifiers_stored": False,
        "raw_identifiers_returned": False,
        "prompt_content_stored": False,
        "tool_arguments_stored": False,
        "tool_results_stored": False,
        "agent_execution_triggered": False,
        "automatic_retry_triggered": False,
        "automatic_mutation_performed": False,
        "automatic_approval_performed": False,
        "manual_approval_required": True,
        "monitoring_required": True,
        "production_routing_enabled": False,
        "activation_or_export_performed": False,
        "downstream_export_enabled": False,
    }


def _validate_safety(report: Mapping[str, Any], label: str) -> None:
    safety = report.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError(f"{label} safety evidence is required.")
    for field in _FALSE_SAFETY_FIELDS:
        if safety.get(field) is not False:
            raise ValueError(f"Unsafe {label} safety field: {field}.")
    if safety.get("manual_approval_required") is not True:
        raise ValueError(f"{label} manual approval must remain required.")
    if safety.get("monitoring_required") is not True:
        raise ValueError(f"{label} monitoring must remain required.")


def _safe_token(value: Any, *, label: str, default: str | None = None) -> str:
    text = str(value or default or "").strip().lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_:-.")
    if not text or len(text) > 128 or any(character not in allowed for character in text):
        raise ValueError(f"{label} must be a safe metadata token.")
    return text


class ProductionModule5AgentExecutionService:
    """Create privacy-minimized evidence from an already completed supervisor run."""

    _ALLOWED_ROUTES = {
        "blocked",
        "failed",
        "needs_clarification",
        "needs_existing_approval",
        "pending_approval",
        "completed_safe",
    }
    _ALLOWED_STATUSES = _ALLOWED_ROUTES | {
        "completed",
        "skipped",
        "blocked_stale_source",
        "blocked_no_safe_exact_match",
    }

    def record(
        self,
        *,
        request: Module5AgentExecutionRequest,
        supervisor_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(supervisor_result)
        if result.get("tenant_id") not in {None, request.tenant_id}:
            raise ValueError("Module 5 supervisor tenant mismatch.")
        if result.get("run_id") not in {None, request.run_id}:
            raise ValueError("Module 5 supervisor run mismatch.")
        if result.get("downstream_export_enabled") is not False:
            raise ValueError("Module 5 engineering evidence requires export disabled.")
        safe_export = result.get("safe_export")
        if isinstance(safe_export, Mapping) and (
            safe_export.get("downstream_export_enabled") is not False
        ):
            raise ValueError("Module 5 safe-export evidence must remain disabled.")

        route = str(result.get("supervisor_route") or "").strip()
        stage = _safe_token(
            result.get("supervisor_stage"),
            label="supervisor_stage",
            default="unknown",
        )
        status = str(result.get("status") or "").strip()
        if route not in self._ALLOWED_ROUTES:
            raise ValueError("Unsupported Module 5 supervisor route.")
        if status not in self._ALLOWED_STATUSES:
            raise ValueError("Unsupported Module 5 supervisor status.")
        reason_codes = sorted({
            _safe_token(value, label="supervisor_reason_code")
            for value in (result.get("supervisor_reason_codes") or [])
            if str(value).strip()
        })
        if not reason_codes:
            reason_codes = ["supervisor_route_recorded"]
        trace = result.get("supervisor_graph_trace") or result.get("supervisor_trace")
        event_counts = self._event_counts(trace)
        recovery = self._recovery_summary(result.get("supervisor_recovery"))
        execution = {
            "pipeline_status": status,
            "supervisor_route": route,
            "supervisor_stage": stage,
            "reason_codes": reason_codes,
            "terminal_status": _safe_token(
                result.get("graph_terminal_status"),
                label="graph_terminal_status",
                default=route,
            ),
            "trace_event_counts": event_counts,
            "recovery_summary": recovery,
        }
        fingerprint = stable_fingerprint({
            "policy_version": MODULE5_EXECUTION_POLICY_VERSION,
            "request": request.to_record(),
            "execution": execution,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE5_EXECUTION_POLICY_VERSION,
            "request": request.to_record(),
            "execution_report_fingerprint": fingerprint,
            "execution": execution,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 5.1 report status.")
        if payload.get("policy_version") != MODULE5_EXECUTION_POLICY_VERSION:
            raise ValueError("Unsupported Module 5.1 policy version.")
        request = Module5AgentExecutionRequest(**dict(payload.get("request") or {}))
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("Module 5.1 execution evidence is required.")
        if execution.get("supervisor_route") not in self._ALLOWED_ROUTES:
            raise ValueError("Module 5.1 supervisor route is invalid.")
        if execution.get("pipeline_status") not in self._ALLOWED_STATUSES:
            raise ValueError("Module 5.1 pipeline status is invalid.")
        if not execution.get("reason_codes"):
            raise ValueError("Module 5.1 reason codes are required.")
        _safe_token(execution.get("supervisor_stage"), label="supervisor_stage")
        _safe_token(execution.get("terminal_status"), label="terminal_status")
        for value in execution.get("reason_codes") or []:
            _safe_token(value, label="supervisor_reason_code")
        self._validate_event_counts(execution.get("trace_event_counts"))
        self._validate_recovery(execution.get("recovery_summary"))
        expected = stable_fingerprint({
            "policy_version": MODULE5_EXECUTION_POLICY_VERSION,
            "request": request.to_record(),
            "execution": dict(execution),
        })
        if payload.get("execution_report_fingerprint") != expected:
            raise ValueError("Module 5.1 execution fingerprint mismatch.")
        _validate_safety(payload, "Module 5.1")
        return payload

    def _event_counts(self, trace: Any) -> dict[str, int]:
        counts: Counter[str] = Counter()
        if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
            for entry in trace[:1000]:
                if isinstance(entry, Mapping):
                    event = _safe_token(
                        entry.get("event"),
                        label="trace_event",
                        default="unknown",
                    )
                    counts[event] += 1
        if not counts:
            counts["trace_not_available"] = 1
        return dict(sorted(counts.items()))

    def _validate_event_counts(self, value: Any) -> None:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("Module 5.1 trace event counts are required.")
        for key, count in value.items():
            _safe_token(key, label="trace_event")
            if not 1 <= int(count) <= 1000:
                raise ValueError("Module 5.1 trace event counts are invalid.")

    def _recovery_summary(self, value: Any) -> dict[str, Any]:
        source = dict(value) if isinstance(value, Mapping) else {}
        attempt_count = max(1, min(int(source.get("attempt_count") or 1), 100))
        max_attempts = max(attempt_count, min(int(source.get("max_attempts") or 1), 100))
        return {
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "retried": bool(source.get("retried")) or attempt_count > 1,
            "exhausted": bool(source.get("exhausted")),
            "last_error_category": _safe_token(
                source.get("last_error_category"),
                label="last_error_category",
                default="none",
            ),
        }

    def _validate_recovery(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise ValueError("Module 5.1 recovery summary is required.")
        attempt_count = int(value.get("attempt_count") or 0)
        max_attempts = int(value.get("max_attempts") or 0)
        if not 1 <= attempt_count <= max_attempts <= 100:
            raise ValueError("Module 5.1 recovery bounds are invalid.")
        if bool(value.get("retried")) != (attempt_count > 1):
            raise ValueError("Module 5.1 retry evidence is inconsistent.")
        _safe_token(value.get("last_error_category"), label="last_error_category")


class ProductionModule5PolicyEvaluationService:
    """Map execution evidence to a non-authorizing supervisor recommendation."""

    def evaluate(self, *, execution_report: Mapping[str, Any]) -> dict[str, Any]:
        execution = ProductionModule5AgentExecutionService().validate_report(
            execution_report
        )
        route = execution["execution"]["supervisor_route"]
        action = {
            "failed": "quarantine_and_review",
            "blocked": "hold_and_review",
            "needs_clarification": "request_manual_clarification",
            "needs_existing_approval": "await_existing_approval",
            "pending_approval": "await_manual_approval",
            "completed_safe": "shadow_review_only",
        }[route]
        decision = {
            "recommended_action": action,
            "source_supervisor_route": route,
            "manual_approval_required": True,
            "production_execution_authorized": False,
            "agent_mutation_authorized": False,
            "eligible_for_activation": False,
            "eligible_for_export": False,
        }
        fingerprint = stable_fingerprint({
            "policy_version": MODULE5_POLICY_POLICY_VERSION,
            "tenant_id": execution["request"]["tenant_id"],
            "source_execution_report_fingerprint": execution[
                "execution_report_fingerprint"
            ],
            "decision": decision,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE5_POLICY_POLICY_VERSION,
            "tenant_id": execution["request"]["tenant_id"],
            "source_execution_report_fingerprint": execution[
                "execution_report_fingerprint"
            ],
            "policy_report_fingerprint": fingerprint,
            "decision": decision,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 5.2 report status.")
        if payload.get("policy_version") != MODULE5_POLICY_POLICY_VERSION:
            raise ValueError("Unsupported Module 5.2 policy version.")
        decision = payload.get("decision")
        if not isinstance(decision, Mapping):
            raise ValueError("Module 5.2 decision evidence is required.")
        if decision.get("recommended_action") not in {
            "quarantine_and_review",
            "hold_and_review",
            "request_manual_clarification",
            "await_existing_approval",
            "await_manual_approval",
            "shadow_review_only",
        }:
            raise ValueError("Module 5.2 action is invalid.")
        for field in (
            "production_execution_authorized",
            "agent_mutation_authorized",
            "eligible_for_activation",
            "eligible_for_export",
        ):
            if decision.get(field) is not False:
                raise ValueError(f"Module 5.2 cannot authorize {field}.")
        if decision.get("manual_approval_required") is not True:
            raise ValueError("Module 5.2 manual approval is required.")
        source = required_sha256_digest(
            payload.get("source_execution_report_fingerprint"),
            label="source_execution_report_fingerprint",
        )
        tenant_id = required_slug(payload.get("tenant_id"), label="tenant_id")
        expected = stable_fingerprint({
            "policy_version": MODULE5_POLICY_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_execution_report_fingerprint": source,
            "decision": dict(decision),
        })
        if payload.get("policy_report_fingerprint") != expected:
            raise ValueError("Module 5.2 policy fingerprint mismatch.")
        _validate_safety(payload, "Module 5.2")
        return payload


class ProductionModule5CircuitBreakerService:
    """Assess aggregate supervisor health without triggering retries or routing."""

    def evaluate(
        self,
        *,
        request: Module5CircuitBreakerRequest,
        execution_reports: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        reports = [
            ProductionModule5AgentExecutionService().validate_report(value)
            for value in execution_reports
        ]
        fingerprints = [value["execution_report_fingerprint"] for value in reports]
        if fingerprints != list(request.execution_report_fingerprints):
            raise ValueError("Module 5 circuit-breaker fingerprint lineage mismatch.")
        if any(value["request"]["tenant_id"] != request.tenant_id for value in reports):
            raise ValueError("Module 5 circuit-breaker tenant mismatch.")
        failures = sum(
            value["execution"]["supervisor_route"] == "failed" for value in reports
        )
        blocked = sum(
            value["execution"]["supervisor_route"] == "blocked" for value in reports
        )
        retried = sum(
            value["execution"]["recovery_summary"]["retried"] for value in reports
        )
        exhausted = sum(
            value["execution"]["recovery_summary"]["exhausted"] for value in reports
        )
        sample_size = len(reports)
        failure_rate = round(failures / sample_size, 8)
        threshold_met = (
            sample_size >= request.minimum_sample_size
            and failure_rate >= request.failure_rate_threshold
        )
        if exhausted or threshold_met:
            state, action = "open", "quarantine_and_manual_recovery_review"
        elif failures or blocked or retried:
            state, action = "review_required", "hold_and_investigate"
        else:
            state, action = "closed", "continue_shadow_monitoring"
        assessment = {
            "sample_size": sample_size,
            "failure_count": failures,
            "blocked_count": blocked,
            "retried_count": retried,
            "exhausted_count": exhausted,
            "failure_rate": failure_rate,
            "threshold_met": threshold_met,
            "circuit_state": state,
            "recommended_action": action,
            "production_execution_authorized": False,
        }
        fingerprint = stable_fingerprint({
            "policy_version": MODULE5_CIRCUIT_BREAKER_POLICY_VERSION,
            "request": request.to_record(),
            "assessment": assessment,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE5_CIRCUIT_BREAKER_POLICY_VERSION,
            "request": request.to_record(),
            "circuit_breaker_report_fingerprint": fingerprint,
            "assessment": assessment,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 5.3 report status.")
        if payload.get("policy_version") != MODULE5_CIRCUIT_BREAKER_POLICY_VERSION:
            raise ValueError("Unsupported Module 5.3 policy version.")
        request = Module5CircuitBreakerRequest(**dict(payload.get("request") or {}))
        assessment = payload.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ValueError("Module 5.3 assessment is required.")
        sample_size = int(assessment.get("sample_size") or 0)
        if sample_size != len(request.execution_report_fingerprints):
            raise ValueError("Module 5.3 sample size is inconsistent.")
        failures = int(assessment.get("failure_count") or 0)
        expected_rate = round(failures / sample_size, 8)
        if float(assessment.get("failure_rate") or 0.0) != expected_rate:
            raise ValueError("Module 5.3 failure rate is inconsistent.")
        threshold_met = (
            sample_size >= request.minimum_sample_size
            and expected_rate >= request.failure_rate_threshold
        )
        if bool(assessment.get("threshold_met")) != threshold_met:
            raise ValueError("Module 5.3 threshold result is inconsistent.")
        exhausted = int(assessment.get("exhausted_count") or 0)
        blocked = int(assessment.get("blocked_count") or 0)
        retried = int(assessment.get("retried_count") or 0)
        expected_state = (
            "open" if exhausted or threshold_met
            else "review_required" if failures or blocked or retried
            else "closed"
        )
        if assessment.get("circuit_state") != expected_state:
            raise ValueError("Module 5.3 circuit state is inconsistent.")
        if assessment.get("production_execution_authorized") is not False:
            raise ValueError("Module 5.3 cannot authorize production execution.")
        expected = stable_fingerprint({
            "policy_version": MODULE5_CIRCUIT_BREAKER_POLICY_VERSION,
            "request": request.to_record(),
            "assessment": dict(assessment),
        })
        if payload.get("circuit_breaker_report_fingerprint") != expected:
            raise ValueError("Module 5.3 circuit-breaker fingerprint mismatch.")
        _validate_safety(payload, "Module 5.3")
        return payload


class ProductionModule5HumanShadowReviewService:
    """Record an explicit manual decision and non-routing shadow observation."""

    def evaluate(
        self,
        *,
        policy_report: Mapping[str, Any],
        manual_review: Mapping[str, Any],
        shadow_observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy = ProductionModule5PolicyEvaluationService().validate_report(
            policy_report
        )
        review = {
            "decision": str(manual_review.get("decision") or "").strip(),
            "review_reference": _safe_token(
                manual_review.get("review_reference"),
                label="review_reference",
            ),
        }
        if review["decision"] not in {
            "approved_for_shadow_review",
            "rejected",
            "needs_review",
        }:
            raise ValueError("Unsupported Module 5 manual review decision.")
        if not review["review_reference"]:
            raise ValueError("Module 5 manual review reference is required.")
        observation = None
        if shadow_observation is not None:
            if review["decision"] != "approved_for_shadow_review":
                raise ValueError("Module 5 shadow observation requires manual approval.")
            observation = {
                "observation_status": str(
                    shadow_observation.get("observation_status") or ""
                ).strip(),
                "routing_enabled": shadow_observation.get("routing_enabled"),
                "activation_or_export_performed": shadow_observation.get(
                    "activation_or_export_performed"
                ),
            }
            if observation["observation_status"] not in {
                "aligned",
                "degraded",
                "error",
            }:
                raise ValueError("Unsupported Module 5 shadow observation.")
            if observation["routing_enabled"] is not False:
                raise ValueError("Module 5 shadow review cannot enable routing.")
            if observation["activation_or_export_performed"] is not False:
                raise ValueError("Module 5 shadow review cannot activate or export.")
        passed = bool(
            review["decision"] == "approved_for_shadow_review"
            and observation
            and observation["observation_status"] == "aligned"
        )
        core = {
            "policy_report_fingerprint": policy["policy_report_fingerprint"],
            "manual_review": review,
            "shadow_observation": observation,
            "shadow_validation_passed": passed,
        }
        fingerprint = stable_fingerprint({
            "policy_version": MODULE5_HUMAN_REVIEW_POLICY_VERSION,
            "tenant_id": policy["tenant_id"],
            **core,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE5_HUMAN_REVIEW_POLICY_VERSION,
            "tenant_id": policy["tenant_id"],
            "source_policy_report_fingerprint": policy[
                "policy_report_fingerprint"
            ],
            "human_review_report_fingerprint": fingerprint,
            **core,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 5.4 report status.")
        if payload.get("policy_version") != MODULE5_HUMAN_REVIEW_POLICY_VERSION:
            raise ValueError("Unsupported Module 5.4 policy version.")
        review = payload.get("manual_review")
        if not isinstance(review, Mapping) or review.get("decision") not in {
            "approved_for_shadow_review",
            "rejected",
            "needs_review",
        }:
            raise ValueError("Module 5.4 manual review is invalid.")
        _safe_token(review.get("review_reference"), label="review_reference")
        observation = payload.get("shadow_observation")
        if observation is not None:
            if review.get("decision") != "approved_for_shadow_review":
                raise ValueError("Module 5.4 shadow evidence requires manual approval.")
            if not isinstance(observation, Mapping):
                raise ValueError("Module 5.4 shadow observation is invalid.")
            if observation.get("observation_status") not in {
                "aligned", "degraded", "error"
            }:
                raise ValueError("Module 5.4 shadow status is invalid.")
            if observation.get("routing_enabled") is not False:
                raise ValueError("Module 5.4 cannot enable routing.")
            if observation.get("activation_or_export_performed") is not False:
                raise ValueError("Module 5.4 cannot activate or export.")
        expected_pass = bool(
            review.get("decision") == "approved_for_shadow_review"
            and observation
            and observation.get("observation_status") == "aligned"
        )
        if payload.get("shadow_validation_passed") is not expected_pass:
            raise ValueError("Module 5.4 shadow result is inconsistent.")
        core = {
            "policy_report_fingerprint": required_sha256_digest(
                payload.get("policy_report_fingerprint"),
                label="policy_report_fingerprint",
            ),
            "manual_review": dict(review),
            "shadow_observation": dict(observation) if observation else None,
            "shadow_validation_passed": expected_pass,
        }
        tenant_id = required_slug(payload.get("tenant_id"), label="tenant_id")
        expected = stable_fingerprint({
            "policy_version": MODULE5_HUMAN_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            **core,
        })
        if payload.get("human_review_report_fingerprint") != expected:
            raise ValueError("Module 5.4 human review fingerprint mismatch.")
        if payload.get("source_policy_report_fingerprint") != core[
            "policy_report_fingerprint"
        ]:
            raise ValueError("Module 5.4 policy lineage mismatch.")
        _validate_safety(payload, "Module 5.4")
        return payload


class ProductionModule5RecoveryCertificationService:
    """Produce a manual recovery plan; never certify or execute production."""

    def plan(
        self,
        *,
        circuit_breaker_report: Mapping[str, Any],
        human_review_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        circuit = ProductionModule5CircuitBreakerService().validate_report(
            circuit_breaker_report
        )
        review = ProductionModule5HumanShadowReviewService().validate_report(
            human_review_report
        )
        if circuit["request"]["tenant_id"] != review["tenant_id"]:
            raise ValueError("Module 5 recovery tenant mismatch.")
        state = circuit["assessment"]["circuit_state"]
        shadow_passed = review["shadow_validation_passed"] is True
        if state == "closed" and shadow_passed:
            action = "controlled_shadow_certification_review"
            engineering_ready = True
        elif state == "open":
            action = "manual_quarantine_recovery_review"
            engineering_ready = False
        elif state == "review_required":
            action = "manual_stability_investigation"
            engineering_ready = False
        else:
            action = "await_manual_shadow_resolution"
            engineering_ready = False
        plan = {
            "recommended_action": action,
            "engineering_evidence_complete": engineering_ready,
            "production_certified": False,
            "recovery_executed": False,
            "manual_execution_required": True,
            "production_execution_authorized": False,
        }
        fingerprint = stable_fingerprint({
            "policy_version": MODULE5_RECOVERY_POLICY_VERSION,
            "tenant_id": review["tenant_id"],
            "source_circuit_breaker_report_fingerprint": circuit[
                "circuit_breaker_report_fingerprint"
            ],
            "source_human_review_report_fingerprint": review[
                "human_review_report_fingerprint"
            ],
            "plan": plan,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE5_RECOVERY_POLICY_VERSION,
            "tenant_id": review["tenant_id"],
            "source_circuit_breaker_report_fingerprint": circuit[
                "circuit_breaker_report_fingerprint"
            ],
            "source_human_review_report_fingerprint": review[
                "human_review_report_fingerprint"
            ],
            "recovery_report_fingerprint": fingerprint,
            "recovery_plan": plan,
            "safety": {**_safety(), "recovery_executed": False},
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 5.5 report status.")
        if payload.get("policy_version") != MODULE5_RECOVERY_POLICY_VERSION:
            raise ValueError("Unsupported Module 5.5 policy version.")
        plan = payload.get("recovery_plan")
        if not isinstance(plan, Mapping):
            raise ValueError("Module 5.5 recovery plan is required.")
        if plan.get("recommended_action") not in {
            "controlled_shadow_certification_review",
            "manual_quarantine_recovery_review",
            "manual_stability_investigation",
            "await_manual_shadow_resolution",
        }:
            raise ValueError("Module 5.5 recovery action is invalid.")
        for field in (
            "production_certified",
            "recovery_executed",
            "production_execution_authorized",
        ):
            if plan.get(field) is not False:
                raise ValueError(f"Module 5.5 cannot set {field}.")
        if plan.get("manual_execution_required") is not True:
            raise ValueError("Module 5.5 manual execution is required.")
        source_circuit = required_sha256_digest(
            payload.get("source_circuit_breaker_report_fingerprint"),
            label="source_circuit_breaker_report_fingerprint",
        )
        source_review = required_sha256_digest(
            payload.get("source_human_review_report_fingerprint"),
            label="source_human_review_report_fingerprint",
        )
        expected = stable_fingerprint({
            "policy_version": MODULE5_RECOVERY_POLICY_VERSION,
            "tenant_id": required_slug(
                payload.get("tenant_id"), label="tenant_id"
            ),
            "source_circuit_breaker_report_fingerprint": source_circuit,
            "source_human_review_report_fingerprint": source_review,
            "plan": dict(plan),
        })
        if payload.get("recovery_report_fingerprint") != expected:
            raise ValueError("Module 5.5 recovery fingerprint mismatch.")
        if payload.get("safety", {}).get("recovery_executed") is not False:
            raise ValueError("Module 5.5 recovery cannot be executed.")
        _validate_safety(payload, "Module 5.5")
        return payload
