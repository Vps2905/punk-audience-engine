from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import (
    required_slug,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


QUALITY_SNAPSHOT_POLICY_VERSION = "aggregate_quality_snapshot_policy_v1"
QUALITY_DRIFT_POLICY_VERSION = "aggregate_quality_drift_policy_v1"
QUALITY_ALERT_POLICY_VERSION = "aggregate_quality_alert_plan_policy_v1"


QualityDomain = Literal[
    "ingestion",
    "privacy",
    "feature",
    "embedding",
    "retrieval",
    "cohort",
    "agent",
]


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


@dataclass(frozen=True)
class ProductionQualitySnapshotRequest:
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
            raise ValueError("Unsupported quality monitoring execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateQualityMetric:
    metric_name: str
    domain: QualityDomain
    observed_value: float
    target_min: float
    target_max: float
    critical_min: float
    critical_max: float
    drift_warning_delta: float
    drift_critical_delta: float
    sample_count: int
    source_evidence_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_name",
            required_slug(self.metric_name, label="metric_name"),
        )
        if self.domain not in {
            "ingestion",
            "privacy",
            "feature",
            "embedding",
            "retrieval",
            "cohort",
            "agent",
        }:
            raise ValueError("Unsupported aggregate quality domain.")
        for field in (
            "observed_value",
            "target_min",
            "target_max",
            "critical_min",
            "critical_max",
            "drift_warning_delta",
            "drift_critical_delta",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), label=field),
            )
        if not (
            self.critical_min
            <= self.target_min
            <= self.target_max
            <= self.critical_max
        ):
            raise ValueError(
                "Quality thresholds must satisfy critical_min <= target_min "
                "<= target_max <= critical_max."
            )
        if not 0.0 < self.drift_warning_delta <= self.drift_critical_delta:
            raise ValueError("Quality drift thresholds are invalid.")
        if not 1 <= int(self.sample_count) <= 10_000_000_000:
            raise ValueError("sample_count must be between 1 and 10 billion.")
        object.__setattr__(self, "sample_count", int(self.sample_count))
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
class ProductionQualityDriftRequest:
    tenant_id: str
    baseline_snapshot_fingerprint: str
    current_snapshot_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        for field in (
            "baseline_snapshot_fingerprint",
            "current_snapshot_fingerprint",
        ):
            object.__setattr__(
                self,
                field,
                required_sha256_digest(getattr(self, field), label=field),
            )
        if self.baseline_snapshot_fingerprint == self.current_snapshot_fingerprint:
            raise ValueError("Quality drift snapshots must be distinct.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
