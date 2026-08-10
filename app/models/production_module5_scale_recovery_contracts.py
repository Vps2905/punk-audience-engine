from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.models.production_bounded_autonomy_contracts import (
    required_metadata_token,
)
from app.models.production_module3_cohort_contracts import required_slug
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


MODULE5_SCALE_RECOVERY_CERTIFICATION_VERSION = (
    "module5_scale_latency_recovery_certification_v1"
)

REQUIRED_SCALE_RECOVERY_SCENARIOS = (
    "baseline_throughput",
    "concurrent_execution",
    "duplicate_replay",
    "timeout_containment",
    "worker_restart_recovery",
    "transient_database_recovery",
    "backpressure_containment",
    "circuit_breaker_containment",
)


def _required_identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
    )
    if not text or len(text) > 128 or any(char not in allowed for char in text):
        raise ValueError(f"{label} must be a safe opaque identifier.")
    return text


@dataclass(frozen=True)
class ScaleRecoveryPolicy:
    """Acceptance thresholds for measured, safe staging executions."""

    minimum_total_work_units: int = 10_000
    minimum_invocation_count: int = 16
    minimum_observed_concurrency: int = 4
    minimum_work_units_per_second: float = 50.0
    minimum_vector_queries_per_second: float = 0.0
    maximum_p95_latency_ms: float = 2_000.0
    maximum_p99_latency_ms: float = 5_000.0
    maximum_database_connection_wait_p99_ms: float = 500.0
    maximum_queue_depth: int = 10_000
    maximum_peak_python_bytes: int = 1_073_741_824
    maximum_total_invocations: int = 10_000
    maximum_total_work_units: int = 100_000_000
    require_observed_work_units: bool = False
    require_historical_pipeline: bool = False

    def __post_init__(self) -> None:
        for label, maximum in (
            ("minimum_total_work_units", 100_000_000),
            ("minimum_invocation_count", 10_000),
            ("minimum_observed_concurrency", 256),
            ("maximum_total_invocations", 100_000),
            ("maximum_total_work_units", 1_000_000_000),
            ("maximum_peak_python_bytes", 64_000_000_000),
            ("maximum_queue_depth", 10_000_000),
        ):
            value = int(getattr(self, label))
            if value < 1 or value > maximum:
                raise ValueError(f"{label} must be between 1 and {maximum}.")
            object.__setattr__(self, label, value)
        if self.minimum_invocation_count > self.maximum_total_invocations:
            raise ValueError("Invocation policy bounds are inconsistent.")
        if self.minimum_total_work_units > self.maximum_total_work_units:
            raise ValueError("Work-unit policy bounds are inconsistent.")
        for label, allow_zero in (
            ("minimum_work_units_per_second", True),
            ("minimum_vector_queries_per_second", True),
            ("maximum_p95_latency_ms", False),
            ("maximum_p99_latency_ms", False),
            ("maximum_database_connection_wait_p99_ms", False),
        ):
            value = float(getattr(self, label))
            minimum = 0.0 if allow_zero else 0.000001
            if value < minimum or value > 3_600_000.0:
                raise ValueError(f"{label} is outside the supported range.")
            object.__setattr__(self, label, value)
        if self.maximum_p99_latency_ms < self.maximum_p95_latency_ms:
            raise ValueError("p99 latency threshold cannot be below p95.")
        if not isinstance(self.require_observed_work_units, bool):
            raise TypeError("require_observed_work_units must be a boolean.")
        if not isinstance(self.require_historical_pipeline, bool):
            raise TypeError("require_historical_pipeline must be a boolean.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleRecoveryCertificationRequest:
    tenant_id: str
    certification_id: str
    evaluation_epoch_seconds: int
    source_functional_shadow_report_fingerprint: str
    source_agent_security_certification_fingerprint: str
    environment: Literal["staging", "preproduction"] = "staging"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "certification_id",
            _required_identifier(self.certification_id, label="certification_id"),
        )
        evaluated = int(self.evaluation_epoch_seconds)
        if evaluated < 1:
            raise ValueError("evaluation_epoch_seconds must be positive.")
        object.__setattr__(self, "evaluation_epoch_seconds", evaluated)
        for field in (
            "source_functional_shadow_report_fingerprint",
            "source_agent_security_certification_fingerprint",
        ):
            object.__setattr__(
                self,
                field,
                required_sha256_digest(getattr(self, field), label=field),
            )
        if self.environment not in {"staging", "preproduction"}:
            raise ValueError("Scale certification requires staging evidence.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleRecoveryExerciseCase:
    case_id: str
    scenario: str
    invocation_count: int
    work_units_per_invocation: int
    requested_concurrency: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _required_identifier(self.case_id, label="case_id"),
        )
        scenario = required_metadata_token(self.scenario, label="scenario")
        if scenario not in REQUIRED_SCALE_RECOVERY_SCENARIOS:
            raise ValueError("Unsupported scale/recovery scenario.")
        object.__setattr__(self, "scenario", scenario)
        for label, maximum in (
            ("invocation_count", 10_000),
            ("work_units_per_invocation", 100_000_000),
            ("requested_concurrency", 256),
        ):
            value = int(getattr(self, label))
            if value < 1 or value > maximum:
                raise ValueError(f"{label} must be between 1 and {maximum}.")
            object.__setattr__(self, label, value)
        if self.requested_concurrency > self.invocation_count:
            raise ValueError("requested_concurrency cannot exceed invocations.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleRecoveryInvocationResult:
    status: Literal["completed", "recovered", "blocked", "failed"]
    result_fingerprint: str | None
    idempotency_key: str
    attempt_count: int = 1
    recovery_count: int = 0
    error_code: str | None = None
    backpressure_observed: bool = False
    circuit_breaker_opened: bool = False
    stage_latencies_ms: Mapping[str, float] = field(default_factory=dict)
    vector_query_count: int = 0
    database_connection_wait_ms: float = 0.0
    queue_depth_observed: int = 0
    raw_identifiers_returned: bool = False
    duplicate_side_effect_performed: bool = False
    production_effect_performed: bool = False
    activation_or_export_performed: bool = False
    manual_approval_required: bool = True
    observed_work_units: int | None = None
    workload_source: Literal[
        "generic_safe_adapter",
        "historical_postgres_pipeline",
        "infrastructure_fault_driver",
    ] = "generic_safe_adapter"

    def __post_init__(self) -> None:
        if self.status not in {"completed", "recovered", "blocked", "failed"}:
            raise ValueError("Unsupported scale invocation status.")
        if self.result_fingerprint is not None:
            object.__setattr__(
                self,
                "result_fingerprint",
                required_sha256_digest(
                    self.result_fingerprint,
                    label="result_fingerprint",
                ),
            )
        object.__setattr__(
            self,
            "idempotency_key",
            _required_identifier(self.idempotency_key, label="idempotency_key"),
        )
        attempts = int(self.attempt_count)
        recoveries = int(self.recovery_count)
        if attempts < 1 or attempts > 100:
            raise ValueError("attempt_count must be between 1 and 100.")
        if recoveries < 0 or recoveries >= attempts:
            raise ValueError("recovery_count must be below attempt_count.")
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "recovery_count", recoveries)
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                required_metadata_token(self.error_code, label="error_code"),
            )
        if self.status in {"completed", "recovered"} and (
            self.result_fingerprint is None
        ):
            raise ValueError("Successful invocations require a result fingerprint.")
        if self.status in {"blocked", "failed"} and self.error_code is None:
            raise ValueError("Blocked or failed invocations require an error code.")
        for field in (
            "raw_identifiers_returned",
            "duplicate_side_effect_performed",
            "production_effect_performed",
            "activation_or_export_performed",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"Unsafe scale invocation field: {field}.")
        if self.manual_approval_required is not True:
            raise ValueError("Scale execution cannot remove manual approval.")
        if self.observed_work_units is not None:
            observed_work_units = int(self.observed_work_units)
            if observed_work_units < 0 or observed_work_units > 100_000_000:
                raise ValueError(
                    "observed_work_units must be between 0 and 100000000."
                )
            object.__setattr__(
                self,
                "observed_work_units",
                observed_work_units,
            )
        if self.workload_source not in {
            "generic_safe_adapter",
            "historical_postgres_pipeline",
            "infrastructure_fault_driver",
        }:
            raise ValueError("Unsupported workload_source.")
        if len(self.stage_latencies_ms) > 32:
            raise ValueError("Stage latency evidence is bounded to 32 stages.")
        stages: dict[str, float] = {}
        for key, value in self.stage_latencies_ms.items():
            token = required_metadata_token(key, label="stage_id")
            latency = float(value)
            if latency < 0.0 or latency > 3_600_000.0:
                raise ValueError("Stage latency evidence must be bounded.")
            stages[token] = round(latency, 6)
        object.__setattr__(self, "stage_latencies_ms", stages)
        for label, maximum in (
            ("vector_query_count", 100_000_000),
            ("queue_depth_observed", 10_000_000),
        ):
            value = int(getattr(self, label))
            if value < 0 or value > maximum:
                raise ValueError(f"{label} must be between 0 and {maximum}.")
            object.__setattr__(self, label, value)
        wait = float(self.database_connection_wait_ms)
        if wait < 0.0 or wait > 3_600_000.0:
            raise ValueError("database_connection_wait_ms must be bounded.")
        object.__setattr__(
            self,
            "database_connection_wait_ms",
            round(wait, 6),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
