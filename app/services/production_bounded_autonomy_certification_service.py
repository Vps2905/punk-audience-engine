from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.production_bounded_autonomy_certification_contracts import (
    BOUNDED_AUTONOMY_CERTIFICATION_VERSION,
    ShadowCertificationCase,
    ShadowCertificationPolicy,
    bounded_latency_ms,
)
from app.models.production_bounded_autonomy_contracts import (
    required_metadata_token,
)
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    stable_fingerprint,
)
from app.services.production_bounded_autonomy_shadow_service import (
    ProductionBoundedAutonomyShadowComparisonService,
)


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction,
        6,
    )


class ProductionBoundedAutonomyCertificationService:
    """Aggregate repeated dual-run observations into non-authorizing evidence."""

    def __init__(
        self,
        *,
        comparison_service: (
            ProductionBoundedAutonomyShadowComparisonService | None
        ) = None,
        policy: ShadowCertificationPolicy | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        effective_environment = (
            environment if environment is not None else os.environ
        )
        self.comparison_service = (
            comparison_service
            or ProductionBoundedAutonomyShadowComparisonService(
                environment=effective_environment
            )
        )
        self.policy = policy or self._policy_from_environment(
            effective_environment
        )

    def run(
        self,
        *,
        tenant_id: str,
        cases: Sequence[ShadowCertificationCase],
    ) -> dict[str, Any]:
        tenant = required_slug(tenant_id, label="tenant_id")
        bounded_cases = tuple(cases)
        if len(bounded_cases) > self.policy.max_cases:
            raise ValueError("Certification case count exceeds max_cases.")
        if any(case.goal.tenant_id != tenant for case in bounded_cases):
            raise ValueError("Certification cases must share one tenant boundary.")

        samples: list[dict[str, Any]] = []
        comparison_fingerprints: set[str] = set()
        duplicate_case_count = 0
        contract_failure_count = 0

        for case_index, case in enumerate(bounded_cases):
            started = time.perf_counter_ns()
            try:
                comparison = self.comparison_service.run(
                    goal=case.goal,
                    legacy_result=case.legacy_result,
                )
                elapsed_ms = bounded_latency_ms(
                    (time.perf_counter_ns() - started) / 1_000_000.0,
                    label="shadow_latency_ms",
                )
                comparison = self.comparison_service.validate_report(comparison)
                fingerprint = str(
                    comparison.get("shadow_comparison_report_fingerprint") or ""
                )
                if fingerprint in comparison_fingerprints:
                    duplicate_case_count += 1
                    continue
                comparison_fingerprints.add(fingerprint)
                sample = self._sample_from_comparison(
                    case_index=case_index,
                    objective_sha256=case.goal.objective_sha256,
                    comparison=comparison,
                    shadow_latency_ms=elapsed_ms,
                    legacy_latency_ms=case.legacy_latency_ms,
                )
            except Exception:  # noqa: BLE001 - persisted only as bounded failure
                elapsed_ms = bounded_latency_ms(
                    (time.perf_counter_ns() - started) / 1_000_000.0,
                    label="shadow_latency_ms",
                )
                contract_failure_count += 1
                sample = self._failure_sample(
                    case_index=case_index,
                    objective_sha256=case.goal.objective_sha256,
                    shadow_latency_ms=elapsed_ms,
                    legacy_latency_ms=case.legacy_latency_ms,
                )
            samples.append(sample)

        summary = self._summarize(
            samples,
            submitted_case_count=len(bounded_cases),
        )
        if summary["duplicate_case_count"] != duplicate_case_count:
            raise ValueError("Internal duplicate accounting mismatch.")
        if summary["contract_failure_count"] != contract_failure_count:
            raise ValueError("Internal contract failure accounting mismatch.")
        gates = self._evaluate_gates(summary)
        eligible = bool(gates and all(value["passed"] for value in gates.values()))
        report = {
            "status": (
                "engineering_preview_ready"
                if eligible
                else "engineering_preview_blocked"
            ),
            "policy_version": BOUNDED_AUTONOMY_CERTIFICATION_VERSION,
            "tenant_id": tenant,
            "policy": self.policy.to_record(),
            "samples": samples,
            "summary": summary,
            "certification_gates": gates,
            "review": {
                "eligible_for_staging_review": eligible,
                "live_cutover_authorized": False,
                "automatic_cutover_performed": False,
                "production_routing_changed": False,
                "fresh_data_certified": False,
                "requires_human_approval": True,
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
        report["bounded_autonomy_certification_fingerprint"] = stable_fingerprint(
            report
        )
        return self.validate_report(report)

    def _sample_from_comparison(
        self,
        *,
        case_index: int,
        objective_sha256: str,
        comparison: Mapping[str, Any],
        shadow_latency_ms: float,
        legacy_latency_ms: float | None,
    ) -> dict[str, Any]:
        legacy = dict(comparison.get("legacy_observation") or {})
        autonomous = dict(comparison.get("autonomous_observation") or {})
        comparison_section = dict(comparison.get("comparison") or {})
        checks = dict(comparison_section.get("checks") or {})
        sample = {
            "case_index": int(case_index),
            "objective_sha256": objective_sha256,
            "comparison_report_fingerprint": comparison[
                "shadow_comparison_report_fingerprint"
            ],
            "comparison_status": comparison.get("status"),
            "legacy_route": legacy.get("decision_route"),
            "autonomous_route": autonomous.get("route"),
            "legacy_stage": legacy.get("decision_stage"),
            "autonomous_stage": autonomous.get("stage"),
            "terminal": bool(autonomous.get("terminal", True)),
            "approval_required": bool(
                autonomous.get("approval_required", True)
            ),
            "route_match": bool(checks.get("route", {}).get("match", False)),
            "stage_match": bool(checks.get("stage", {}).get("match", False)),
            "freshness_match": bool(
                checks.get("freshness", {}).get("match", False)
            ),
            "divergence_count": int(
                comparison_section.get("divergence_count") or 0
            ),
            "critical_divergence_count": int(
                comparison_section.get("critical_divergence_count") or 0
            ),
            "shadow_latency_ms": shadow_latency_ms,
            "legacy_latency_reported": legacy_latency_ms is not None,
            "legacy_latency_ms": float(legacy_latency_ms or 0.0),
            "contract_failure": False,
        }
        sample["sample_fingerprint"] = stable_fingerprint(sample)
        return sample

    def _failure_sample(
        self,
        *,
        case_index: int,
        objective_sha256: str,
        shadow_latency_ms: float,
        legacy_latency_ms: float | None,
    ) -> dict[str, Any]:
        sample = {
            "case_index": int(case_index),
            "objective_sha256": objective_sha256,
            "comparison_report_fingerprint": None,
            "comparison_status": "contract_failure",
            "legacy_route": "not_evaluated",
            "autonomous_route": "blocked",
            "legacy_stage": "not_evaluated",
            "autonomous_stage": "certification",
            "terminal": True,
            "approval_required": True,
            "route_match": False,
            "stage_match": False,
            "freshness_match": False,
            "divergence_count": 1,
            "critical_divergence_count": 1,
            "shadow_latency_ms": shadow_latency_ms,
            "legacy_latency_reported": legacy_latency_ms is not None,
            "legacy_latency_ms": float(legacy_latency_ms or 0.0),
            "contract_failure": True,
        }
        sample["sample_fingerprint"] = stable_fingerprint(sample)
        return sample

    def _summarize(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        submitted_case_count: int,
    ) -> dict[str, Any]:
        total = len(samples)
        if int(submitted_case_count) < total:
            raise ValueError("submitted_case_count cannot be below evaluated runs.")
        duplicate_case_count = int(submitted_case_count) - total
        contract_failure_count = sum(
            sample.get("contract_failure") is True for sample in samples
        )
        divisor = max(total, 1)
        unique_goals = len(
            {str(sample.get("objective_sha256")) for sample in samples}
        )
        route_matches = sum(bool(sample.get("route_match")) for sample in samples)
        stage_matches = sum(bool(sample.get("stage_match")) for sample in samples)
        freshness_matches = sum(
            bool(sample.get("freshness_match")) for sample in samples
        )
        divergent = sum(
            int(sample.get("divergence_count") or 0) > 0 for sample in samples
        )
        critical = sum(
            int(sample.get("critical_divergence_count") or 0) > 0
            for sample in samples
        )
        terminal_safety = sum(
            bool(sample.get("terminal"))
            and str(sample.get("autonomous_route")) in {"blocked", "failed"}
            for sample in samples
        )
        nonterminal_review = sum(
            not bool(sample.get("terminal"))
            and bool(sample.get("approval_required"))
            for sample in samples
        )
        latencies = [
            float(sample.get("shadow_latency_ms") or 0.0)
            for sample in samples
        ]
        legacy_latencies = [
            float(sample.get("legacy_latency_ms") or 0.0)
            for sample in samples
            if sample.get("legacy_latency_reported") is True
        ]
        return {
            "submitted_case_count": int(submitted_case_count),
            "evaluated_run_count": total,
            "duplicate_case_count": int(duplicate_case_count),
            "contract_failure_count": int(contract_failure_count),
            "unique_goal_hash_count": unique_goals,
            "terminal_safety_run_count": terminal_safety,
            "nonterminal_review_run_count": nonterminal_review,
            "route_agreement_rate": _round_rate(route_matches / divisor),
            "stage_agreement_rate": _round_rate(stage_matches / divisor),
            "freshness_agreement_rate": _round_rate(freshness_matches / divisor),
            "overall_divergence_rate": _round_rate(divergent / divisor),
            "critical_divergence_rate": _round_rate(critical / divisor),
            "shadow_latency_ms": {
                "min": round(min(latencies), 6) if latencies else 0.0,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
                "max": round(max(latencies), 6) if latencies else 0.0,
            },
            "legacy_latency_ms": {
                "reported_run_count": len(legacy_latencies),
                "p50": _percentile(legacy_latencies, 0.50),
                "p95": _percentile(legacy_latencies, 0.95),
            },
        }

    def _evaluate_gates(
        self,
        summary: Mapping[str, Any],
        *,
        policy: ShadowCertificationPolicy | None = None,
    ) -> dict[str, dict[str, Any]]:
        policy = policy or self.policy

        def minimum(observed: Any, required: Any) -> dict[str, Any]:
            return {
                "passed": observed >= required,
                "observed": observed,
                "required_minimum": required,
            }

        def maximum(observed: Any, allowed: Any) -> dict[str, Any]:
            return {
                "passed": observed <= allowed,
                "observed": observed,
                "allowed_maximum": allowed,
            }

        latency = dict(summary.get("shadow_latency_ms") or {})
        return {
            "minimum_runs": minimum(
                summary.get("evaluated_run_count", 0), policy.min_total_runs
            ),
            "goal_diversity": minimum(
                summary.get("unique_goal_hash_count", 0),
                policy.min_unique_goal_hashes,
            ),
            "terminal_safety_coverage": minimum(
                summary.get("terminal_safety_run_count", 0),
                policy.min_terminal_safety_runs,
            ),
            "nonterminal_review_coverage": minimum(
                summary.get("nonterminal_review_run_count", 0),
                policy.min_nonterminal_review_runs,
            ),
            "route_agreement": minimum(
                summary.get("route_agreement_rate", 0.0),
                policy.min_route_agreement_rate,
            ),
            "stage_agreement": minimum(
                summary.get("stage_agreement_rate", 0.0),
                policy.min_stage_agreement_rate,
            ),
            "freshness_agreement": minimum(
                summary.get("freshness_agreement_rate", 0.0),
                policy.min_freshness_agreement_rate,
            ),
            "overall_divergence": maximum(
                summary.get("overall_divergence_rate", 1.0),
                policy.max_overall_divergence_rate,
            ),
            "critical_divergence": maximum(
                summary.get("critical_divergence_rate", 1.0),
                policy.max_critical_divergence_rate,
            ),
            "shadow_p95_latency": maximum(
                latency.get("p95", 0.0),
                policy.max_shadow_p95_latency_ms,
            ),
            "contract_integrity": maximum(
                summary.get("contract_failure_count", 0),
                0,
            ),
            "duplicate_resistance": maximum(
                summary.get("duplicate_case_count", 0),
                0,
            ),
        }

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(report)))
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") not in {
            "engineering_preview_ready",
            "engineering_preview_blocked",
        }:
            raise ValueError("Unsupported autonomy certification status.")
        if payload.get("policy_version") != BOUNDED_AUTONOMY_CERTIFICATION_VERSION:
            raise ValueError("Unsupported autonomy certification policy version.")
        required_slug(payload.get("tenant_id"), label="tenant_id")
        policy_record = payload.get("policy")
        samples = payload.get("samples")
        summary = payload.get("summary")
        gates = payload.get("certification_gates")
        review = payload.get("review")
        safety = payload.get("safety")
        if not isinstance(policy_record, Mapping):
            raise TypeError("Certification policy evidence is required.")
        if not isinstance(samples, list):
            raise TypeError("Certification samples are required.")
        if not all(isinstance(value, Mapping) for value in samples):
            raise TypeError("Certification samples must be mappings.")
        if not all(
            isinstance(value, Mapping)
            for value in (summary, gates, review, safety)
        ):
            raise TypeError("Certification evidence sections are required.")
        policy = ShadowCertificationPolicy(**dict(policy_record))
        if len(samples) > policy.max_cases:
            raise ValueError("Certification evidence exceeds max_cases.")
        sample_fingerprints: set[str] = set()
        comparison_fingerprints: set[str] = set()
        case_indexes: set[int] = set()
        for sample in samples:
            self._validate_sample(sample, policy=policy)
            case_index = int(sample.get("case_index"))
            if case_index in case_indexes:
                raise ValueError("Certification case indexes must be unique.")
            case_indexes.add(case_index)
            sample_payload = dict(sample)
            supplied_sample_fingerprint = sample_payload.pop(
                "sample_fingerprint", None
            )
            if stable_fingerprint(sample_payload) != supplied_sample_fingerprint:
                raise ValueError("Certification sample fingerprint mismatch.")
            if supplied_sample_fingerprint in sample_fingerprints:
                raise ValueError("Certification sample fingerprints must be unique.")
            sample_fingerprints.add(str(supplied_sample_fingerprint))
            comparison_fingerprint = sample.get("comparison_report_fingerprint")
            if comparison_fingerprint:
                if comparison_fingerprint in comparison_fingerprints:
                    raise ValueError(
                        "Comparison report fingerprints must be unique."
                    )
                comparison_fingerprints.add(str(comparison_fingerprint))
            bounded_latency_ms(
                sample.get("shadow_latency_ms"),
                label="shadow_latency_ms",
            )
            bounded_latency_ms(
                sample.get("legacy_latency_ms"),
                label="legacy_latency_ms",
            )
        recomputed_summary = self._summarize(
            samples,
            submitted_case_count=int(summary.get("submitted_case_count") or 0),
        )
        if int(summary.get("submitted_case_count") or 0) > policy.max_cases:
            raise ValueError("Submitted certification evidence exceeds max_cases.")
        if recomputed_summary != dict(summary):
            raise ValueError("Certification summary mismatch.")
        recomputed_gates = self._evaluate_gates(
            recomputed_summary,
            policy=policy,
        )
        if recomputed_gates != dict(gates):
            raise ValueError("Certification gate evidence mismatch.")
        all_gates_passed = bool(gates) and all(
            value.get("passed") is True for value in gates.values()
        )
        expected_status = (
            "engineering_preview_ready"
            if all_gates_passed
            else "engineering_preview_blocked"
        )
        if payload.get("status") != expected_status:
            raise ValueError("Certification status does not match its gates.")
        if review.get("eligible_for_staging_review") is not all_gates_passed:
            raise ValueError("Staging review eligibility mismatch.")
        for key in (
            "live_cutover_authorized",
            "automatic_cutover_performed",
            "production_routing_changed",
            "fresh_data_certified",
        ):
            if review.get(key) is not False:
                raise ValueError(f"Unsafe certification review field: {key}.")
        if review.get("requires_human_approval") is not True:
            raise ValueError("Certification requires human approval.")
        if safety.get("shadow_only") is not True:
            raise ValueError("Certification evidence must remain shadow-only.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Certification evidence requires manual approval.")
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
                raise ValueError(f"Unsafe certification safety field: {key}.")
        if self._contains_forbidden_payload_key(payload):
            raise ValueError("Prompt, objective, or payload content entered evidence.")
        supplied = payload.pop(
            "bounded_autonomy_certification_fingerprint", None
        )
        if stable_fingerprint(payload) != supplied:
            raise ValueError("Autonomy certification fingerprint mismatch.")
        payload["bounded_autonomy_certification_fingerprint"] = supplied
        return payload

    def _validate_sample(
        self,
        sample: Mapping[str, Any],
        *,
        policy: ShadowCertificationPolicy,
    ) -> None:
        raw_case_index = sample.get("case_index")
        if not isinstance(raw_case_index, int) or isinstance(raw_case_index, bool):
            raise TypeError("case_index must be an integer.")
        case_index = raw_case_index
        if not 0 <= case_index < policy.max_cases:
            raise ValueError("case_index is outside the certification budget.")
        objective_hash = str(sample.get("objective_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", objective_hash) is None:
            raise ValueError("objective_sha256 must be a SHA-256 digest.")
        for field_name in (
            "legacy_route",
            "autonomous_route",
            "legacy_stage",
            "autonomous_stage",
            "comparison_status",
        ):
            required_metadata_token(sample.get(field_name), label=field_name)
        for field_name in (
            "terminal",
            "approval_required",
            "route_match",
            "stage_match",
            "freshness_match",
            "legacy_latency_reported",
            "contract_failure",
        ):
            if not isinstance(sample.get(field_name), bool):
                raise TypeError(f"{field_name} must be boolean.")
        for field_name in (
            "divergence_count",
            "critical_divergence_count",
        ):
            value = sample.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer.")
            if not 0 <= value <= 64:
                raise ValueError(f"{field_name} must be between 0 and 64.")
        if sample.get("critical_divergence_count") > sample.get(
            "divergence_count"
        ):
            raise ValueError("Critical divergences cannot exceed divergences.")
        comparison_fingerprint = sample.get("comparison_report_fingerprint")
        contract_failure = sample.get("contract_failure") is True
        if contract_failure:
            if comparison_fingerprint is not None:
                raise ValueError(
                    "Contract failures cannot claim a comparison fingerprint."
                )
            if (
                sample.get("route_match") is not False
                or sample.get("stage_match") is not False
                or sample.get("freshness_match") is not False
                or sample.get("critical_divergence_count", 0) < 1
            ):
                raise ValueError("Contract failure sample semantics are invalid.")
        else:
            if re.fullmatch(
                r"[0-9a-f]{64}", str(comparison_fingerprint or "")
            ) is None:
                raise ValueError(
                    "Successful samples require a comparison SHA-256 fingerprint."
                )
            if sample.get("comparison_status") not in {
                "engineering_preview_ready",
                "engineering_preview_blocked",
            }:
                raise ValueError("Unsupported comparison sample status.")
        if sample.get("legacy_latency_reported") is False and float(
            sample.get("legacy_latency_ms") or 0.0
        ) != 0.0:
            raise ValueError("Unreported legacy latency must remain zero.")

    def _contains_forbidden_payload_key(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key or "").strip().lower()
                if normalized in {
                    "prompt",
                    "objective",
                    "tool_arguments",
                    "tool_results",
                    "source_rows",
                    "source_payload",
                }:
                    return True
                if self._contains_forbidden_payload_key(item):
                    return True
        elif isinstance(value, list):
            return any(self._contains_forbidden_payload_key(item) for item in value)
        return False

    def _policy_from_environment(
        self,
        environment: Mapping[str, str],
    ) -> ShadowCertificationPolicy:
        def integer(key: str, default: int) -> int:
            raw = str(environment.get(key) or default).strip()
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer.") from exc

        def decimal(key: str, default: float) -> float:
            raw = str(environment.get(key) or default).strip()
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be numeric.") from exc

        return ShadowCertificationPolicy(
            min_total_runs=integer(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_RUNS", 25
            ),
            min_unique_goal_hashes=integer(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_UNIQUE_GOALS", 10
            ),
            min_terminal_safety_runs=integer(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_TERMINAL_SAFETY_RUNS", 5
            ),
            min_nonterminal_review_runs=integer(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_REVIEW_RUNS", 5
            ),
            min_route_agreement_rate=decimal(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_ROUTE_AGREEMENT", 1.0
            ),
            min_stage_agreement_rate=decimal(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_STAGE_AGREEMENT", 1.0
            ),
            min_freshness_agreement_rate=decimal(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MIN_FRESHNESS_AGREEMENT", 1.0
            ),
            max_overall_divergence_rate=decimal(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_DIVERGENCE_RATE", 0.0
            ),
            max_critical_divergence_rate=decimal(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_CRITICAL_DIVERGENCE_RATE",
                0.0,
            ),
            max_shadow_p95_latency_ms=decimal(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_P95_LATENCY_MS", 250.0
            ),
            max_cases=integer(
                "MODULE5_BOUNDED_AUTONOMY_CERT_MAX_CASES", 10_000
            ),
        )
