from __future__ import annotations

import json
import math
import os
import threading
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)
from app.models.production_module5_scale_recovery_contracts import (
    MODULE5_SCALE_RECOVERY_CERTIFICATION_VERSION,
    REQUIRED_SCALE_RECOVERY_SCENARIOS,
    ScaleRecoveryCertificationRequest,
    ScaleRecoveryExerciseCase,
    ScaleRecoveryInvocationResult,
    ScaleRecoveryPolicy,
)
from app.services.production_agent_security_certification_service import (
    ProductionAgentSecurityCertificationService,
)
from app.services.production_bounded_autonomy_functional_shadow_service import (
    ProductionBoundedAutonomyFunctionalShadowService,
)


class ScaleRecoveryWorkloadRunner(Protocol):
    """Application adapter used by the harness to execute real safe work."""

    def execute(
        self,
        *,
        case: ScaleRecoveryExerciseCase,
        invocation_index: int,
    ) -> ScaleRecoveryInvocationResult: ...


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 6)


class ProductionModule5ScaleRecoveryCertificationService:
    """Measure concurrent safe work and certify bounded recovery semantics."""

    def __init__(
        self,
        *,
        policy: ScaleRecoveryPolicy | None = None,
        functional_shadow_validator: (
            Callable[[Mapping[str, Any]], dict[str, Any]] | None
        ) = None,
        agent_security_validator: (
            Callable[[Mapping[str, Any]], dict[str, Any]] | None
        ) = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self.policy = policy or self._policy_from_environment()
        self._functional_shadow_validator = (
            functional_shadow_validator
            or ProductionBoundedAutonomyFunctionalShadowService(
                environment={}
            ).validate_report
        )
        self._agent_security_validator = (
            agent_security_validator
            or ProductionAgentSecurityCertificationService().validate_report
        )

    def run(
        self,
        *,
        request: ScaleRecoveryCertificationRequest,
        functional_shadow_report: Mapping[str, Any],
        agent_security_certification_report: Mapping[str, Any],
        cases: Sequence[ScaleRecoveryExerciseCase],
        runner: ScaleRecoveryWorkloadRunner,
    ) -> dict[str, Any]:
        functional = self._functional_shadow_validator(functional_shadow_report)
        authorization = self._agent_security_validator(
            agent_security_certification_report
        )
        self._validate_lineage(
            request=request,
            functional=functional,
            authorization=authorization,
        )
        normalized_cases = self._validate_cases(cases)
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()

        def invoke(
            case_order: int,
            case: ScaleRecoveryExerciseCase,
            invocation_index: int,
        ) -> dict[str, Any]:
            nonlocal active, maximum_active
            started = time.perf_counter_ns()
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                try:
                    result = runner.execute(
                        case=case,
                        invocation_index=invocation_index,
                    )
                    if not isinstance(result, ScaleRecoveryInvocationResult):
                        raise TypeError("Runner returned an unsupported contract.")
                except Exception:
                    result = ScaleRecoveryInvocationResult(
                        status="failed",
                        result_fingerprint=None,
                        idempotency_key=(
                            f"{case.case_id}-invocation-{invocation_index}"
                        ),
                        error_code="workload_runner_exception",
                        observed_work_units=0,
                    )
            finally:
                with active_lock:
                    active -= 1
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            declared_work_units = case.work_units_per_invocation
            measured_work_units = (
                result.observed_work_units
                if result.observed_work_units is not None
                else declared_work_units
            )
            observation = {
                "case_order": case_order,
                "case_id": case.case_id,
                "scenario": case.scenario,
                "invocation_index": invocation_index,
                "declared_work_units": declared_work_units,
                "work_units": measured_work_units,
                "work_units_observed": (
                    result.observed_work_units is not None
                ),
                "latency_ms": round(latency_ms, 6),
                **result.to_record(),
            }
            return observation

        started_tracing = not tracemalloc.is_tracing()
        if started_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter_ns()
        futures = []
        with ThreadPoolExecutor(
            max_workers=max(case.requested_concurrency for case in normalized_cases)
        ) as executor:
            for case_order, case in enumerate(normalized_cases):
                for invocation_index in range(case.invocation_count):
                    futures.append(
                        executor.submit(
                            invoke,
                            case_order,
                            case,
                            invocation_index,
                        )
                    )
            observations = [future.result() for future in as_completed(futures)]
        duration_seconds = max(
            (time.perf_counter_ns() - started) / 1_000_000_000.0,
            0.000001,
        )
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        if started_tracing:
            tracemalloc.stop()
        observations.sort(
            key=lambda value: (value["case_order"], value["invocation_index"])
        )
        for value in observations:
            value.pop("case_order")
            value["observation_fingerprint"] = stable_fingerprint(value)

        summary = self._summary(
            observations=observations,
            duration_seconds=duration_seconds,
            peak_python_bytes=peak_python_bytes,
            maximum_active=maximum_active,
        )
        scenario_results = self._scenario_results(observations)
        gates = self._gates(summary=summary, scenario_results=scenario_results)
        eligible = all(value["passed"] is True for value in gates.values())
        report = {
            "status": (
                "engineering_preview_ready"
                if eligible
                else "engineering_preview_blocked"
            ),
            "policy_version": MODULE5_SCALE_RECOVERY_CERTIFICATION_VERSION,
            "request": request.to_record(),
            "lineage": {
                "source_functional_shadow_report_fingerprint": functional[
                    "functional_shadow_report_fingerprint"
                ],
                "source_agent_security_certification_fingerprint": authorization[
                    "agent_security_certification_fingerprint"
                ],
            },
            "policy": self.policy.to_record(),
            "cases": [value.to_record() for value in normalized_cases],
            "observations": observations,
            "summary": summary,
            "scenario_results": scenario_results,
            "certification_gates": gates,
            "review": {
                "eligible_for_staging_review": eligible,
                "live_cutover_authorized": False,
                "fresh_data_certified": False,
                "external_distributed_load_certified": False,
                "database_failover_certified": False,
                "backup_restore_certified": False,
                "disaster_recovery_certified": False,
                "automatic_cutover_performed": False,
            },
            "safety": {
                "raw_identifiers_returned": False,
                "prompt_content_stored": False,
                "tool_arguments_stored": False,
                "duplicate_side_effect_performed": False,
                "production_effect_performed": False,
                "automatic_approval_performed": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "manual_approval_required": True,
            },
        }
        report["scale_recovery_certification_fingerprint"] = stable_fingerprint(
            report
        )
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(report)))
        assert_no_raw_identifier_fields(payload)
        expected_top_level = {
            "status",
            "policy_version",
            "request",
            "lineage",
            "policy",
            "cases",
            "observations",
            "summary",
            "scenario_results",
            "certification_gates",
            "review",
            "safety",
            "scale_recovery_certification_fingerprint",
        }
        if set(payload) != expected_top_level:
            raise ValueError("Scale/recovery report contains unsupported fields.")
        if payload.get("policy_version") != (
            MODULE5_SCALE_RECOVERY_CERTIFICATION_VERSION
        ):
            raise ValueError("Unsupported scale/recovery certification policy.")
        request = ScaleRecoveryCertificationRequest(
            **dict(payload.get("request") or {})
        )
        policy = ScaleRecoveryPolicy(**dict(payload.get("policy") or {}))
        cases = [
            ScaleRecoveryExerciseCase(**dict(value))
            for value in payload.get("cases") or []
        ]
        self._validate_cases(cases, policy=policy)
        lineage = payload.get("lineage")
        observations = payload.get("observations")
        summary = payload.get("summary")
        scenario_results = payload.get("scenario_results")
        gates = payload.get("certification_gates")
        review = payload.get("review")
        safety = payload.get("safety")
        if not all(
            isinstance(value, Mapping)
            for value in (lineage, summary, scenario_results, gates, review, safety)
        ) or not isinstance(observations, list):
            raise TypeError("Scale/recovery certification sections are required.")
        if lineage.get("source_functional_shadow_report_fingerprint") != (
            request.source_functional_shadow_report_fingerprint
        ) or lineage.get("source_agent_security_certification_fingerprint") != (
            request.source_agent_security_certification_fingerprint
        ):
            raise ValueError("Scale/recovery certification lineage mismatch.")
        if set(lineage) != {
            "source_functional_shadow_report_fingerprint",
            "source_agent_security_certification_fingerprint",
        }:
            raise ValueError("Scale/recovery lineage contains unsupported fields.")
        for value in lineage.values():
            required_sha256_digest(value, label="scale_recovery_lineage")

        case_map = {value.case_id: value for value in cases}
        normalized_observations = []
        seen = set()
        for value in observations:
            if not isinstance(value, Mapping):
                raise TypeError("Scale/recovery observation is invalid.")
            expected_observation_fields = {
                "case_id",
                "scenario",
                "invocation_index",
                "declared_work_units",
                "work_units",
                "work_units_observed",
                "latency_ms",
                "observation_fingerprint",
                *ScaleRecoveryInvocationResult.__dataclass_fields__,
            }
            if set(value) != expected_observation_fields:
                raise ValueError(
                    "Scale/recovery observation contains unsupported fields."
                )
            record = dict(value)
            supplied = record.pop("observation_fingerprint", None)
            if stable_fingerprint(record) != supplied:
                raise ValueError("Scale/recovery observation fingerprint mismatch.")
            case = case_map.get(str(record.get("case_id") or ""))
            if case is None or record.get("scenario") != case.scenario:
                raise ValueError("Scale/recovery observation case mismatch.")
            key = (case.case_id, int(record.get("invocation_index", -1)))
            if key in seen or not 0 <= key[1] < case.invocation_count:
                raise ValueError("Scale/recovery invocation evidence is invalid.")
            seen.add(key)
            if int(record.get("declared_work_units", -1)) != (
                case.work_units_per_invocation
            ):
                raise ValueError(
                    "Scale/recovery declared work-unit evidence is invalid."
                )
            latency = float(record.get("latency_ms", -1.0))
            if latency < 0.0 or latency > 3_600_000.0:
                raise ValueError("Scale/recovery latency evidence is invalid.")
            result_fields = {
                key: record.get(key)
                for key in ScaleRecoveryInvocationResult.__dataclass_fields__
            }
            normalized_result = ScaleRecoveryInvocationResult(**result_fields)
            observed = normalized_result.observed_work_units
            measurement_flag = record.get("work_units_observed")
            if not isinstance(measurement_flag, bool):
                raise TypeError("Work-unit measurement flag must be boolean.")
            expected_work_units = (
                observed
                if observed is not None
                else case.work_units_per_invocation
            )
            if int(record.get("work_units", -1)) != expected_work_units or (
                measurement_flag is not (observed is not None)
            ):
                raise ValueError("Scale/recovery work-unit evidence is invalid.")
            record["observation_fingerprint"] = required_sha256_digest(
                supplied,
                label="observation_fingerprint",
            )
            normalized_observations.append(record)
        expected_count = sum(value.invocation_count for value in cases)
        if len(seen) != expected_count:
            raise ValueError("Scale/recovery observation coverage is incomplete.")

        duration = float(summary.get("duration_seconds", 0.0))
        peak = int(summary.get("peak_python_bytes", -1))
        concurrency = int(summary.get("maximum_observed_concurrency", 0))
        expected_summary = self._summary(
            observations=normalized_observations,
            duration_seconds=duration,
            peak_python_bytes=peak,
            maximum_active=concurrency,
        )
        if dict(summary) != expected_summary:
            raise ValueError("Scale/recovery summary is inconsistent.")
        expected_scenarios = self._scenario_results(normalized_observations)
        if dict(scenario_results) != expected_scenarios:
            raise ValueError("Scale/recovery scenario evidence is inconsistent.")
        expected_gates = self._gates(
            summary=expected_summary,
            scenario_results=expected_scenarios,
            policy=policy,
        )
        if dict(gates) != expected_gates:
            raise ValueError("Scale/recovery certification gates are inconsistent.")
        eligible = all(value.get("passed") is True for value in gates.values())
        expected_status = (
            "engineering_preview_ready"
            if eligible
            else "engineering_preview_blocked"
        )
        if payload.get("status") != expected_status or (
            review.get("eligible_for_staging_review") is not eligible
        ):
            raise ValueError("Scale/recovery readiness is inconsistent.")
        for field in (
            "live_cutover_authorized",
            "fresh_data_certified",
            "external_distributed_load_certified",
            "database_failover_certified",
            "backup_restore_certified",
            "disaster_recovery_certified",
            "automatic_cutover_performed",
        ):
            if review.get(field) is not False:
                raise ValueError(f"Scale/recovery review cannot set {field}.")
        if set(review) != {
            "eligible_for_staging_review",
            "live_cutover_authorized",
            "fresh_data_certified",
            "external_distributed_load_certified",
            "database_failover_certified",
            "backup_restore_certified",
            "disaster_recovery_certified",
            "automatic_cutover_performed",
        }:
            raise ValueError("Scale/recovery review contains unsupported fields.")
        for field in (
            "raw_identifiers_returned",
            "prompt_content_stored",
            "tool_arguments_stored",
            "duplicate_side_effect_performed",
            "production_effect_performed",
            "automatic_approval_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe scale/recovery field: {field}.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Scale/recovery evidence requires manual approval.")
        if set(safety) != {
            "raw_identifiers_returned",
            "prompt_content_stored",
            "tool_arguments_stored",
            "duplicate_side_effect_performed",
            "production_effect_performed",
            "automatic_approval_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
            "manual_approval_required",
        }:
            raise ValueError("Scale/recovery safety contains unsupported fields.")
        supplied = payload.pop("scale_recovery_certification_fingerprint", None)
        if stable_fingerprint(payload) != supplied:
            raise ValueError("Scale/recovery certification fingerprint mismatch.")
        payload["scale_recovery_certification_fingerprint"] = (
            required_sha256_digest(
                supplied,
                label="scale_recovery_certification_fingerprint",
            )
        )
        return payload

    def _validate_lineage(
        self,
        *,
        request: ScaleRecoveryCertificationRequest,
        functional: Mapping[str, Any],
        authorization: Mapping[str, Any],
    ) -> None:
        if functional.get("request", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Functional shadow tenant lineage mismatch.")
        if authorization.get("request", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Agent security tenant lineage mismatch.")
        if functional.get("functional_shadow_report_fingerprint") != (
            request.source_functional_shadow_report_fingerprint
        ):
            raise ValueError("Functional shadow fingerprint lineage mismatch.")
        if authorization.get("agent_security_certification_fingerprint") != (
            request.source_agent_security_certification_fingerprint
        ):
            raise ValueError("Agent security fingerprint lineage mismatch.")
        if authorization.get("lineage", {}).get(
            "source_functional_shadow_report_fingerprint"
        ) != functional.get("functional_shadow_report_fingerprint"):
            raise ValueError("Agent security lineage does not bind functional shadow.")
        if functional.get("status") != "engineering_preview_ready" or (
            authorization.get("status") != "engineering_preview_ready"
        ):
            raise ValueError("Module 5.9 and 5.10 evidence must be ready.")
        if authorization.get("review", {}).get("live_cutover_authorized") is not False:
            raise ValueError("Agent security evidence cannot authorize cutover.")

    def _validate_cases(
        self,
        cases: Sequence[ScaleRecoveryExerciseCase],
        *,
        policy: ScaleRecoveryPolicy | None = None,
    ) -> tuple[ScaleRecoveryExerciseCase, ...]:
        active_policy = policy or self.policy
        values = tuple(cases)
        if not values:
            raise ValueError("Scale/recovery cases are required.")
        if len({value.case_id for value in values}) != len(values):
            raise ValueError("Scale/recovery case identifiers must be unique.")
        scenarios = {value.scenario for value in values}
        if set(REQUIRED_SCALE_RECOVERY_SCENARIOS) - scenarios:
            raise ValueError("All required scale/recovery scenarios are required.")
        invocations = sum(value.invocation_count for value in values)
        work_units = sum(
            value.invocation_count * value.work_units_per_invocation
            for value in values
        )
        if invocations > active_policy.maximum_total_invocations:
            raise ValueError("Scale/recovery invocation limit exceeded.")
        if work_units > active_policy.maximum_total_work_units:
            raise ValueError("Scale/recovery work-unit limit exceeded.")
        return values

    def _summary(
        self,
        *,
        observations: Sequence[Mapping[str, Any]],
        duration_seconds: float,
        peak_python_bytes: int,
        maximum_active: int,
    ) -> dict[str, Any]:
        duration = round(float(duration_seconds), 9)
        if duration <= 0.0:
            raise ValueError("Scale/recovery duration must be positive.")
        total_work = sum(int(value["work_units"]) for value in observations)
        declared_work = sum(
            int(value["declared_work_units"]) for value in observations
        )
        observed_work_invocations = sum(
            value.get("work_units_observed") is True
            for value in observations
        )
        workload_sources = sorted(
            {str(value["workload_source"]) for value in observations}
        )
        historical_pipeline_invocations = sum(
            value.get("workload_source") == "historical_postgres_pipeline"
            for value in observations
        )
        latencies = [float(value["latency_ms"]) for value in observations]
        vector_queries = sum(
            int(value["vector_query_count"]) for value in observations
        )
        database_waits = [
            float(value["database_connection_wait_ms"]) for value in observations
        ]
        stage_values: dict[str, list[float]] = defaultdict(list)
        for observation in observations:
            for stage, latency in dict(
                observation.get("stage_latencies_ms") or {}
            ).items():
                stage_values[str(stage)].append(float(latency))
        failed = sum(value["status"] == "failed" for value in observations)
        recovered = sum(value["status"] == "recovered" for value in observations)
        blocked = sum(value["status"] == "blocked" for value in observations)
        return {
            "invocation_count": len(observations),
            "total_work_units": total_work,
            "declared_total_work_units": declared_work,
            "observed_work_unit_invocation_count": observed_work_invocations,
            "work_unit_measurement_complete": (
                observed_work_invocations == len(observations)
            ),
            "workload_sources": workload_sources,
            "historical_pipeline_invocation_count": (
                historical_pipeline_invocations
            ),
            "duration_seconds": duration,
            "work_units_per_second": round(total_work / duration, 6),
            "invocations_per_second": round(len(observations) / duration, 6),
            "vector_query_count": vector_queries,
            "vector_queries_per_second": round(vector_queries / duration, 6),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_p99_ms": _percentile(latencies, 0.99),
            "database_connection_wait_p99_ms": _percentile(
                database_waits,
                0.99,
            ),
            "maximum_queue_depth_observed": max(
                (int(value["queue_depth_observed"]) for value in observations),
                default=0,
            ),
            "stage_latency_percentiles": {
                stage: {
                    "observation_count": len(values),
                    "p50_ms": _percentile(values, 0.50),
                    "p95_ms": _percentile(values, 0.95),
                    "p99_ms": _percentile(values, 0.99),
                }
                for stage, values in sorted(stage_values.items())
            },
            "maximum_observed_concurrency": int(maximum_active),
            "peak_python_bytes": int(peak_python_bytes),
            "completed_invocation_count": (
                len(observations) - failed - recovered - blocked
            ),
            "recovered_invocation_count": recovered,
            "blocked_invocation_count": blocked,
            "failed_invocation_count": failed,
            "total_attempt_count": sum(
                int(value["attempt_count"]) for value in observations
            ),
            "total_recovery_count": sum(
                int(value["recovery_count"]) for value in observations
            ),
        }

    def _scenario_results(
        self,
        observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for value in observations:
            grouped[str(value["scenario"])].append(value)

        def all_status(scenario: str, allowed: set[str]) -> bool:
            values = grouped.get(scenario, [])
            return bool(values) and all(value["status"] in allowed for value in values)

        duplicate_values = grouped.get("duplicate_replay", [])
        fingerprints: dict[str, set[str | None]] = defaultdict(set)
        for value in duplicate_values:
            fingerprints[str(value["idempotency_key"])].add(
                value.get("result_fingerprint")
            )
        duplicate_consistent = bool(duplicate_values) and all(
            len(values) == 1 for values in fingerprints.values()
        ) and len(duplicate_values) > len(fingerprints)
        results = {
            "baseline_throughput": all_status(
                "baseline_throughput", {"completed"}
            ),
            "concurrent_execution": all_status(
                "concurrent_execution", {"completed"}
            ),
            "duplicate_replay": (
                all_status("duplicate_replay", {"completed", "recovered"})
                and duplicate_consistent
                and all(
                    value["duplicate_side_effect_performed"] is False
                    for value in duplicate_values
                )
            ),
            "timeout_containment": (
                all_status("timeout_containment", {"blocked"})
                and all(
                    value["error_code"] == "deadline_exceeded"
                    for value in grouped.get("timeout_containment", [])
                )
            ),
            "worker_restart_recovery": (
                all_status("worker_restart_recovery", {"recovered"})
                and all(
                    int(value["recovery_count"]) >= 1
                    for value in grouped.get("worker_restart_recovery", [])
                )
            ),
            "transient_database_recovery": (
                all_status("transient_database_recovery", {"recovered"})
                and all(
                    int(value["attempt_count"]) >= 2
                    and int(value["recovery_count"]) >= 1
                    for value in grouped.get("transient_database_recovery", [])
                )
            ),
            "backpressure_containment": (
                all_status("backpressure_containment", {"completed", "blocked"})
                and all(
                    value["backpressure_observed"] is True
                    for value in grouped.get("backpressure_containment", [])
                )
            ),
            "circuit_breaker_containment": (
                all_status("circuit_breaker_containment", {"blocked"})
                and all(
                    value["circuit_breaker_opened"] is True
                    and value["error_code"] == "circuit_breaker_open"
                    for value in grouped.get("circuit_breaker_containment", [])
                )
            ),
        }
        return {
            scenario: {
                "passed": results[scenario],
                "observation_count": len(grouped.get(scenario, [])),
            }
            for scenario in REQUIRED_SCALE_RECOVERY_SCENARIOS
        }

    def _gates(
        self,
        *,
        summary: Mapping[str, Any],
        scenario_results: Mapping[str, Any],
        policy: ScaleRecoveryPolicy | None = None,
    ) -> dict[str, Any]:
        active_policy = policy or self.policy
        return {
            "observed_work_units": {
                "passed": (
                    not active_policy.require_observed_work_units
                    or summary["work_unit_measurement_complete"] is True
                ),
                "observed_invocations": summary[
                    "observed_work_unit_invocation_count"
                ],
                "required_invocations": (
                    summary["invocation_count"]
                    if active_policy.require_observed_work_units
                    else 0
                ),
            },
            "historical_pipeline": {
                "passed": (
                    not active_policy.require_historical_pipeline
                    or summary["historical_pipeline_invocation_count"] > 0
                ),
                "observed_invocations": summary[
                    "historical_pipeline_invocation_count"
                ],
                "minimum_invocations": (
                    1 if active_policy.require_historical_pipeline else 0
                ),
            },
            "minimum_workload": {
                "passed": summary["total_work_units"]
                >= active_policy.minimum_total_work_units,
                "observed": summary["total_work_units"],
                "minimum": active_policy.minimum_total_work_units,
            },
            "minimum_invocations": {
                "passed": summary["invocation_count"]
                >= active_policy.minimum_invocation_count,
                "observed": summary["invocation_count"],
                "minimum": active_policy.minimum_invocation_count,
            },
            "concurrency": {
                "passed": summary["maximum_observed_concurrency"]
                >= active_policy.minimum_observed_concurrency,
                "observed": summary["maximum_observed_concurrency"],
                "minimum": active_policy.minimum_observed_concurrency,
            },
            "throughput": {
                "passed": summary["work_units_per_second"]
                >= active_policy.minimum_work_units_per_second,
                "observed": summary["work_units_per_second"],
                "minimum": active_policy.minimum_work_units_per_second,
            },
            "vector_throughput": {
                "passed": summary["vector_queries_per_second"]
                >= active_policy.minimum_vector_queries_per_second,
                "observed": summary["vector_queries_per_second"],
                "minimum": active_policy.minimum_vector_queries_per_second,
            },
            "p95_latency": {
                "passed": summary["latency_p95_ms"]
                <= active_policy.maximum_p95_latency_ms,
                "observed": summary["latency_p95_ms"],
                "maximum": active_policy.maximum_p95_latency_ms,
            },
            "p99_latency": {
                "passed": summary["latency_p99_ms"]
                <= active_policy.maximum_p99_latency_ms,
                "observed": summary["latency_p99_ms"],
                "maximum": active_policy.maximum_p99_latency_ms,
            },
            "peak_python_memory": {
                "passed": summary["peak_python_bytes"]
                <= active_policy.maximum_peak_python_bytes,
                "observed": summary["peak_python_bytes"],
                "maximum": active_policy.maximum_peak_python_bytes,
            },
            "database_pool_wait": {
                "passed": summary["database_connection_wait_p99_ms"]
                <= active_policy.maximum_database_connection_wait_p99_ms,
                "observed": summary["database_connection_wait_p99_ms"],
                "maximum": (
                    active_policy.maximum_database_connection_wait_p99_ms
                ),
            },
            "queue_depth": {
                "passed": summary["maximum_queue_depth_observed"]
                <= active_policy.maximum_queue_depth,
                "observed": summary["maximum_queue_depth_observed"],
                "maximum": active_policy.maximum_queue_depth,
            },
            "zero_unrecovered_failures": {
                "passed": summary["failed_invocation_count"] == 0,
                "observed": summary["failed_invocation_count"],
                "maximum": 0,
            },
            "required_recovery_scenarios": {
                "passed": all(
                    value.get("passed") is True
                    for value in scenario_results.values()
                ),
                "passed_count": sum(
                    value.get("passed") is True
                    for value in scenario_results.values()
                ),
                "required_count": len(REQUIRED_SCALE_RECOVERY_SCENARIOS),
            },
        }

    def _policy_from_environment(self) -> ScaleRecoveryPolicy:
        defaults = ScaleRecoveryPolicy()

        def value(key: str, fallback: int | float) -> str | int | float:
            configured = str(self._environment.get(key) or "").strip()
            return configured if configured else fallback

        return ScaleRecoveryPolicy(
            minimum_total_work_units=value(
                "MODULE5_SCALE_RECOVERY_MIN_TOTAL_WORK_UNITS",
                defaults.minimum_total_work_units,
            ),
            minimum_invocation_count=value(
                "MODULE5_SCALE_RECOVERY_MIN_INVOCATIONS",
                defaults.minimum_invocation_count,
            ),
            minimum_observed_concurrency=value(
                "MODULE5_SCALE_RECOVERY_MIN_OBSERVED_CONCURRENCY",
                defaults.minimum_observed_concurrency,
            ),
            minimum_work_units_per_second=value(
                "MODULE5_SCALE_RECOVERY_MIN_WORK_UNITS_PER_SECOND",
                defaults.minimum_work_units_per_second,
            ),
            minimum_vector_queries_per_second=value(
                "MODULE5_SCALE_RECOVERY_MIN_VECTOR_QUERIES_PER_SECOND",
                defaults.minimum_vector_queries_per_second,
            ),
            maximum_p95_latency_ms=value(
                "MODULE5_SCALE_RECOVERY_MAX_P95_LATENCY_MS",
                defaults.maximum_p95_latency_ms,
            ),
            maximum_p99_latency_ms=value(
                "MODULE5_SCALE_RECOVERY_MAX_P99_LATENCY_MS",
                defaults.maximum_p99_latency_ms,
            ),
            maximum_database_connection_wait_p99_ms=value(
                "MODULE5_SCALE_RECOVERY_MAX_DATABASE_WAIT_P99_MS",
                defaults.maximum_database_connection_wait_p99_ms,
            ),
            maximum_queue_depth=value(
                "MODULE5_SCALE_RECOVERY_MAX_QUEUE_DEPTH",
                defaults.maximum_queue_depth,
            ),
            maximum_peak_python_bytes=value(
                "MODULE5_SCALE_RECOVERY_MAX_PEAK_PYTHON_BYTES",
                defaults.maximum_peak_python_bytes,
            ),
            require_observed_work_units=(
                str(
                    self._environment.get(
                        "MODULE5_SCALE_RECOVERY_REQUIRE_OBSERVED_WORK_UNITS",
                        "true",
                    )
                )
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            require_historical_pipeline=(
                str(
                    self._environment.get(
                        "MODULE5_SCALE_RECOVERY_REQUIRE_HISTORICAL_PIPELINE",
                        "true",
                    )
                )
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            maximum_total_invocations=defaults.maximum_total_invocations,
            maximum_total_work_units=defaults.maximum_total_work_units,
        )
