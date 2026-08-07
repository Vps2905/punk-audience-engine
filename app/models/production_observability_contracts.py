from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import (
    finite_unit_interval,
    required_slug,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


OBSERVABILITY_SNAPSHOT_POLICY_VERSION = "production_observability_snapshot_v1"
OBSERVABILITY_INCIDENT_POLICY_VERSION = "production_incident_review_plan_v1"


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


@dataclass(frozen=True)
class ProductionObservabilitySnapshotRequest:
    tenant_id: str
    snapshot_id: str
    execution_mode: Literal[
        "historical_preview",
        "offline_evaluation",
        "production",
    ]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "snapshot_id",
            required_slug(self.snapshot_id, label="snapshot_id"),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported observability execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateTelemetryCoverage:
    component_name: str
    signal_type: Literal["metrics", "logs", "traces"]
    emitted_event_count: int
    accepted_event_count: int
    rejected_sensitive_attribute_count: int
    correlation_coverage_rate: float
    export_success_rate: float
    minimum_correlation_coverage_rate: float
    minimum_export_success_rate: float
    source_evidence_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_name",
            required_slug(self.component_name, label="component_name"),
        )
        if self.signal_type not in {"metrics", "logs", "traces"}:
            raise ValueError("Unsupported telemetry signal_type.")
        for field in (
            "emitted_event_count",
            "accepted_event_count",
            "rejected_sensitive_attribute_count",
        ):
            value = int(getattr(self, field))
            if not 0 <= value <= 10_000_000_000:
                raise ValueError(f"{field} must be between 0 and 10 billion.")
            object.__setattr__(self, field, value)
        if self.accepted_event_count > self.emitted_event_count:
            raise ValueError("accepted_event_count cannot exceed emitted_event_count.")
        for field in (
            "correlation_coverage_rate",
            "export_success_rate",
            "minimum_correlation_coverage_rate",
            "minimum_export_success_rate",
        ):
            object.__setattr__(
                self,
                field,
                finite_unit_interval(getattr(self, field), label=field),
            )
        object.__setattr__(
            self,
            "source_evidence_fingerprint",
            required_sha256_digest(
                self.source_evidence_fingerprint,
                label="source_evidence_fingerprint",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceLevelObservation:
    service_name: str
    sli_name: str
    objective_direction: Literal["at_least", "at_most"]
    observed_value: float
    target_value: float
    warning_boundary: float
    critical_boundary: float
    sample_count: int
    minimum_sample_count: int
    window_seconds: int
    source_evidence_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service_name",
            required_slug(self.service_name, label="service_name"),
        )
        object.__setattr__(
            self,
            "sli_name",
            required_slug(self.sli_name, label="sli_name"),
        )
        if self.objective_direction not in {"at_least", "at_most"}:
            raise ValueError("Unsupported SLO objective_direction.")
        for field in (
            "observed_value",
            "target_value",
            "warning_boundary",
            "critical_boundary",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), label=field),
            )
        if self.objective_direction == "at_least" and not (
            self.critical_boundary
            <= self.warning_boundary
            <= self.target_value
        ):
            raise ValueError(
                "at_least objectives require critical <= warning <= target."
            )
        if self.objective_direction == "at_most" and not (
            self.target_value
            <= self.warning_boundary
            <= self.critical_boundary
        ):
            raise ValueError(
                "at_most objectives require target <= warning <= critical."
            )
        for field in ("sample_count", "minimum_sample_count"):
            value = int(getattr(self, field))
            if not 1 <= value <= 10_000_000_000:
                raise ValueError(f"{field} must be between 1 and 10 billion.")
            object.__setattr__(self, field, value)
        if not 60 <= int(self.window_seconds) <= 31_536_000:
            raise ValueError("window_seconds must be between 60 and 31536000.")
        object.__setattr__(self, "window_seconds", int(self.window_seconds))
        object.__setattr__(
            self,
            "source_evidence_fingerprint",
            required_sha256_digest(
                self.source_evidence_fingerprint,
                label="source_evidence_fingerprint",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
