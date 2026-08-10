from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from app.models.production_bounded_autonomy_contracts import AutonomyGoal
from app.models.production_module3_cohort_contracts import finite_unit_interval

BOUNDED_AUTONOMY_CERTIFICATION_VERSION = (
    "module5_bounded_autonomy_certification_v1"
)


def bounded_latency_ms(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 3_600_000.0:
        raise ValueError(f"{label} must be between 0 and 3600000 milliseconds.")
    return round(parsed, 6)


@dataclass(frozen=True)
class ShadowCertificationPolicy:
    min_total_runs: int = 25
    min_unique_goal_hashes: int = 10
    min_terminal_safety_runs: int = 5
    min_nonterminal_review_runs: int = 5
    min_route_agreement_rate: float = 1.0
    min_stage_agreement_rate: float = 1.0
    min_freshness_agreement_rate: float = 1.0
    max_overall_divergence_rate: float = 0.0
    max_critical_divergence_rate: float = 0.0
    max_shadow_p95_latency_ms: float = 250.0
    max_cases: int = 10_000

    def __post_init__(self) -> None:
        for field_name in (
            "min_total_runs",
            "min_unique_goal_hashes",
            "min_terminal_safety_runs",
            "min_nonterminal_review_runs",
            "max_cases",
        ):
            value = int(getattr(self, field_name))
            if not 0 <= value <= 10_000:
                raise ValueError(f"{field_name} must be between 0 and 10000.")
            object.__setattr__(self, field_name, value)
        if self.min_total_runs < 1:
            raise ValueError("min_total_runs must be positive.")
        if self.max_cases < self.min_total_runs:
            raise ValueError("max_cases must cover min_total_runs.")
        if self.min_unique_goal_hashes > self.min_total_runs:
            raise ValueError(
                "min_unique_goal_hashes cannot exceed min_total_runs."
            )
        for field_name in (
            "min_route_agreement_rate",
            "min_stage_agreement_rate",
            "min_freshness_agreement_rate",
            "max_overall_divergence_rate",
            "max_critical_divergence_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                finite_unit_interval(
                    getattr(self, field_name),
                    label=field_name,
                ),
            )
        object.__setattr__(
            self,
            "max_shadow_p95_latency_ms",
            bounded_latency_ms(
                self.max_shadow_p95_latency_ms,
                label="max_shadow_p95_latency_ms",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowCertificationCase:
    """Ephemeral case input; prompt and legacy payload are never evidence."""

    goal: AutonomyGoal
    legacy_result: Mapping[str, Any]
    legacy_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.goal.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "shadow",
        }:
            raise ValueError("Certification cases cannot use production mode.")
        if not isinstance(self.legacy_result, Mapping):
            raise TypeError("legacy_result must be a mapping.")
        if self.legacy_latency_ms is not None:
            object.__setattr__(
                self,
                "legacy_latency_ms",
                bounded_latency_ms(
                    self.legacy_latency_ms,
                    label="legacy_latency_ms",
                ),
            )
