from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ProviderScaleAcceptancePolicy:
    target_events_per_day: int = 1_000_000_000
    minimum_average_events_per_second: float = 11_574.0
    minimum_peak_events_per_second: float = 34_722.0
    maximum_queue_age_p99_seconds: float = 900.0
    maximum_end_to_end_p99_seconds: float = 7_200.0
    maximum_rights_propagation_seconds: float = 86_400.0
    maximum_cost_per_billion_usd: float = 1_000.0

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if float(value) <= 0:
                raise ValueError(f"{field_name} must be > 0")


@dataclass(frozen=True)
class ProviderScaleAcceptanceEvidence:
    evidence_id: str
    environment: str
    total_events: int
    duration_seconds: float
    average_events_per_second: float
    peak_events_per_second: float
    queue_age_p99_seconds: float
    end_to_end_p99_seconds: float
    rights_propagation_seconds: float
    cost_usd: float
    duplicate_side_effect_count: int
    raw_identifier_output_count: int
    privacy_partition_violation_count: int
    unrecovered_failure_count: int
    canonical_checksum_failure_count: int
    stale_data_activated_count: int
    source_replay_verified: bool
    worker_restart_verified: bool
    backup_restore_verified: bool

    def __post_init__(self) -> None:
        if not str(self.evidence_id or "").strip():
            raise ValueError("evidence_id is required")
        if str(self.environment or "").strip().lower() not in {
            "staging",
            "preproduction",
        }:
            raise ValueError(
                "Scale evidence must come from staging or preproduction"
            )
        object.__setattr__(
            self,
            "environment",
            str(self.environment).strip().lower(),
        )
        nonnegative = (
            "total_events",
            "duration_seconds",
            "average_events_per_second",
            "peak_events_per_second",
            "queue_age_p99_seconds",
            "end_to_end_p99_seconds",
            "rights_propagation_seconds",
            "cost_usd",
            "duplicate_side_effect_count",
            "raw_identifier_output_count",
            "privacy_partition_violation_count",
            "unrecovered_failure_count",
            "canonical_checksum_failure_count",
            "stale_data_activated_count",
        )
        for field_name in nonnegative:
            if float(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "ProviderScaleAcceptanceEvidence":
        if not isinstance(payload, dict):
            raise ValueError("Scale evidence must be an object")
        return cls(**payload)

    def to_safe_dict(self) -> Dict[str, Any]:
        return asdict(self)

