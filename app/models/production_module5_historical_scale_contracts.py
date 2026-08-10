from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import required_slug
from app.models.production_module5_scale_recovery_contracts import (
    ScaleRecoveryExerciseCase,
)

MODULE5_HISTORICAL_SCALE_ADAPTER_VERSION = (
    "module5_historical_pipeline_scale_adapter_v1"
)


@dataclass(frozen=True)
class HistoricalScaleWorkloadConfiguration:
    """Runtime-only configuration; the objective is never copied to evidence."""

    tenant_id: str
    workload_id: str
    objective: str
    expected_source_rows: int
    postgres_limit: int = 10_000
    k_min: int = 1_000
    epsilon: float = 1.0
    synthetic_rows: int = 1_000
    max_export_cohorts: int = 25
    min_export_quality: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "workload_id",
            required_slug(self.workload_id, label="workload_id"),
        )
        objective = str(self.objective or "").strip()
        if len(objective) < 3 or len(objective) > 4_000:
            raise ValueError("objective must contain 3 to 4000 characters.")
        object.__setattr__(self, "objective", objective)
        for label, maximum in (
            ("expected_source_rows", 100_000_000),
            ("postgres_limit", 1_000_000),
            ("k_min", 10_000_000),
            ("synthetic_rows", 1_000_000),
            ("max_export_cohorts", 10_000),
        ):
            value = int(getattr(self, label))
            if value < 1 or value > maximum:
                raise ValueError(f"{label} must be between 1 and {maximum}.")
            object.__setattr__(self, label, value)
        epsilon = float(self.epsilon)
        quality = float(self.min_export_quality)
        if epsilon <= 0.0 or epsilon > 100.0:
            raise ValueError("epsilon must be greater than 0 and at most 100.")
        if quality < 0.0 or quality > 1.0:
            raise ValueError("min_export_quality must be between 0 and 1.")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "min_export_quality", quality)

    def evidence_descriptor(self) -> dict[str, Any]:
        return {
            "adapter_version": MODULE5_HISTORICAL_SCALE_ADAPTER_VERSION,
            "tenant_id": self.tenant_id,
            "workload_id": self.workload_id,
            "objective_fingerprint": hashlib.sha256(
                self.objective.encode("utf-8")
            ).hexdigest(),
            "expected_source_rows": self.expected_source_rows,
            "postgres_limit": self.postgres_limit,
            "k_min": self.k_min,
            "epsilon": self.epsilon,
            "synthetic_rows": self.synthetic_rows,
            "max_export_cohorts": self.max_export_cohorts,
            "min_export_quality": self.min_export_quality,
            "source": "postgres_safe_derived",
            "artifact_persistence": False,
            "privacy_budget_spend_recorded": False,
            "manual_approval_required": True,
            "downstream_export_enabled": False,
        }


HistoricalScaleProfile = Literal["smoke", "10k", "100k"]


def build_historical_scale_cases(
    *,
    profile: HistoricalScaleProfile,
    expected_source_rows: int,
    concurrency: int = 4,
) -> tuple[ScaleRecoveryExerciseCase, ...]:
    """Plan real calls; certification later uses observed rows, not this estimate."""
    if profile not in {"smoke", "10k", "100k"}:
        raise ValueError("profile must be smoke, 10k, or 100k.")
    expected = int(expected_source_rows)
    if expected < 1 or expected > 100_000_000:
        raise ValueError("expected_source_rows is outside supported bounds.")
    workers = int(concurrency)
    if workers < 1 or workers > 64:
        raise ValueError("concurrency must be between 1 and 64.")
    target = {"smoke": expected * 11, "10k": 10_000, "100k": 100_000}[
        profile
    ]
    real_invocations = max(11, math.ceil(target / expected))
    concurrent_invocations = max(workers, real_invocations - 3)
    real_invocations = concurrent_invocations + 3
    cases = [
        ScaleRecoveryExerciseCase(
            case_id=f"{profile}-historical-baseline",
            scenario="baseline_throughput",
            invocation_count=1,
            work_units_per_invocation=expected,
            requested_concurrency=1,
        ),
        ScaleRecoveryExerciseCase(
            case_id=f"{profile}-historical-concurrent",
            scenario="concurrent_execution",
            invocation_count=concurrent_invocations,
            work_units_per_invocation=expected,
            requested_concurrency=min(workers, concurrent_invocations),
        ),
        ScaleRecoveryExerciseCase(
            case_id=f"{profile}-historical-duplicate",
            scenario="duplicate_replay",
            invocation_count=2,
            work_units_per_invocation=expected,
            requested_concurrency=min(2, workers),
        ),
    ]
    for scenario in (
        "timeout_containment",
        "worker_restart_recovery",
        "transient_database_recovery",
        "backpressure_containment",
        "circuit_breaker_containment",
    ):
        cases.append(
            ScaleRecoveryExerciseCase(
                case_id=f"{profile}-{scenario}",
                scenario=scenario,
                invocation_count=1,
                work_units_per_invocation=expected,
                requested_concurrency=1,
            )
        )
    return tuple(cases)
