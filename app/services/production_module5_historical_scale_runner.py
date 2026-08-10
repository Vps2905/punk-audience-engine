from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from app.agents.audience_supervisor_agent import build_audience_execution_agent
from app.core.certification_evaluation_context import (
    certification_evaluation_context,
)
from app.models.production_module3_cohort_contracts import stable_fingerprint
from app.models.production_module5_historical_scale_contracts import (
    HistoricalScaleWorkloadConfiguration,
)
from app.models.production_module5_scale_recovery_contracts import (
    ScaleRecoveryExerciseCase,
    ScaleRecoveryInvocationResult,
)


class HistoricalPipelineAgent(Protocol):
    def run(self, **kwargs: Any) -> dict[str, Any]: ...


class HistoricalScaleFaultExerciseController(Protocol):
    """Staging-owned driver for real timeout/restart/DB/queue/breaker faults."""

    def execute(
        self,
        *,
        case: ScaleRecoveryExerciseCase,
        invocation_index: int,
        execute_historical: Callable[[], ScaleRecoveryInvocationResult],
    ) -> ScaleRecoveryInvocationResult: ...


_REAL_PIPELINE_SCENARIOS = {
    "baseline_throughput",
    "concurrent_execution",
    "duplicate_replay",
}

_RELEASE_EFFECT_FLAGS = (
    "MODULE3_PRODUCTION_ROUTING_ENABLED",
    "MODULE4_PRODUCTION_ROUTING_ENABLED",
    "MODULE5_PRODUCTION_ROUTING_ENABLED",
    "MODULE5_AGENT_PRODUCTION_EFFECT_AUTHORIZATION_ENABLED",
    "MODULE5_SCALE_RECOVERY_PRODUCTION_CUTOVER_ENABLED",
    "SECURITY_PRODUCTION_RELEASE_ENABLED",
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionHistoricalScaleWorkloadRunner:
    """Run the real historical audience path with every release effect disabled."""

    def __init__(
        self,
        *,
        configuration: HistoricalScaleWorkloadConfiguration,
        agent_factory: Callable[[], HistoricalPipelineAgent] | None = None,
        fault_controller: HistoricalScaleFaultExerciseController | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.configuration = configuration
        self._agent_factory = agent_factory or build_audience_execution_agent
        self._fault_controller = fault_controller
        self._environment = dict(
            os.environ if environment is None else environment
        )

    def execute(
        self,
        *,
        case: ScaleRecoveryExerciseCase,
        invocation_index: int,
    ) -> ScaleRecoveryInvocationResult:
        unsafe_flags = [
            name
            for name in _RELEASE_EFFECT_FLAGS
            if _truthy(self._environment.get(name))
        ]
        key = self._idempotency_key(case, invocation_index)
        if unsafe_flags:
            return ScaleRecoveryInvocationResult(
                status="blocked",
                result_fingerprint=None,
                idempotency_key=key,
                error_code="release_effect_flag_enabled",
                observed_work_units=0,
                workload_source="historical_postgres_pipeline",
            )

        def execute_historical() -> ScaleRecoveryInvocationResult:
            return self._execute_historical(
                case=case,
                invocation_index=invocation_index,
            )

        if case.scenario in _REAL_PIPELINE_SCENARIOS:
            return execute_historical()
        if self._fault_controller is None:
            return ScaleRecoveryInvocationResult(
                status="blocked",
                result_fingerprint=None,
                idempotency_key=key,
                error_code="fault_exercise_not_configured",
                observed_work_units=0,
                workload_source="infrastructure_fault_driver",
            )
        try:
            result = self._fault_controller.execute(
                case=case,
                invocation_index=invocation_index,
                execute_historical=execute_historical,
            )
            if not isinstance(result, ScaleRecoveryInvocationResult):
                raise TypeError("Fault controller returned an invalid contract.")
            if result.workload_source != "infrastructure_fault_driver":
                raise ValueError(
                    "Fault-controller evidence must identify its workload source."
                )
            return result
        except Exception:  # noqa: BLE001 - evidence stores only a fixed code.
            return ScaleRecoveryInvocationResult(
                status="failed",
                result_fingerprint=None,
                idempotency_key=key,
                error_code="fault_exercise_driver_failed",
                observed_work_units=0,
                workload_source="infrastructure_fault_driver",
            )

    def _execute_historical(
        self,
        *,
        case: ScaleRecoveryExerciseCase,
        invocation_index: int,
    ) -> ScaleRecoveryInvocationResult:
        key = self._idempotency_key(case, invocation_index)
        started = time.perf_counter_ns()
        try:
            with certification_evaluation_context():
                result = self._agent_factory().run(
                    prompt=self.configuration.objective,
                    output_root="data/non_persistent_scale_evaluation",
                    source="postgres",
                    safe_cohort_path=None,
                    postgres_limit=self.configuration.postgres_limit,
                    k_min=self.configuration.k_min,
                    epsilon=self.configuration.epsilon,
                    synthetic_rows=self.configuration.synthetic_rows,
                    max_export_cohorts=(
                        self.configuration.max_export_cohorts
                    ),
                    min_export_quality=self.configuration.min_export_quality,
                    approval_required=True,
                    persist_artifacts=False,
                    certification_evaluation=True,
                    tenant_id=self.configuration.tenant_id,
                    request_id=(
                        f"{self.configuration.workload_id}-{case.case_id}-"
                        f"{invocation_index}"
                    ),
                )
            projection = self._validate_and_project(result)
        except Exception:  # noqa: BLE001 - never persist backend error text.
            return ScaleRecoveryInvocationResult(
                status="failed",
                result_fingerprint=None,
                idempotency_key=key,
                error_code="historical_pipeline_execution_failed",
                observed_work_units=0,
                workload_source="historical_postgres_pipeline",
            )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        ranked_matches = int(projection["ranked_match_count"])
        return ScaleRecoveryInvocationResult(
            status="completed",
            result_fingerprint=stable_fingerprint(projection),
            idempotency_key=key,
            stage_latencies_ms={"historical_pipeline": latency_ms},
            vector_query_count=1 if ranked_matches > 0 else 0,
            observed_work_units=int(projection["source_rows"]),
            workload_source="historical_postgres_pipeline",
        )

    def _idempotency_key(
        self,
        case: ScaleRecoveryExerciseCase,
        invocation_index: int,
    ) -> str:
        if case.scenario == "duplicate_replay":
            return f"{self.configuration.workload_id}-duplicate-replay"
        return (
            f"{self.configuration.workload_id}-{case.case_id}-"
            f"{invocation_index}"
        )

    def _validate_and_project(self, result: Any) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError("Historical pipeline returned an invalid payload.")
        source_rows = int(result.get("source_rows") or 0)
        privacy = dict(result.get("privacy_guarantees") or {})
        v2 = dict(result.get("v2_autonomous") or {})
        embedding_manifest = dict(v2.get("embedding_manifest") or {})
        safe_export = dict(result.get("safe_export") or {})
        unsafe_privacy = any(
            privacy.get(field) is not False
            for field in (
                "raw_maids_exported",
                "hashed_identifiers_exported",
                "raw_observations_exported",
                "raw_lat_lng_exported",
                "raw_email_exported",
                "raw_phone_exported",
                "individual_user_data_exported",
            )
        )
        if (
            result.get("status") != "completed"
            or result.get("source_mode") != "postgres_safe_derived"
            or source_rows < 1
            or result.get("certification_evaluation") is not True
            or result.get("downstream_export_enabled") is not False
            or privacy.get("approval_required") is not True
            or unsafe_privacy
            or safe_export.get("downstream_export_enabled") is not False
        ):
            raise ValueError("Historical pipeline violated certification safety.")
        return {
            "source_mode": "postgres_safe_derived",
            "source_rows": source_rows,
            "freshness_status": str(
                result.get("freshness_status") or "unknown"
            ),
            "privacy_cohorts": int(result.get("privacy_cohorts") or 0),
            "selected_cohorts": int(
                result.get("prompt_selected_cohorts") or 0
            ),
            "vector_count": int(embedding_manifest.get("vector_count") or 0),
            "vector_dimension": int(
                embedding_manifest.get("vector_dimension") or 0
            ),
            "ranked_match_count": int(v2.get("ranked_match_count") or 0),
            "approval_status": str(result.get("approval_status") or "blocked"),
            "downstream_export_enabled": False,
            "manual_approval_required": True,
            "raw_identifiers_returned": False,
            "certification_evaluation": True,
        }
