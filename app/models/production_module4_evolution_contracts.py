from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    finite_unit_interval,
    required_slug,
    required_text,
)
from app.models.production_module3_overlap_contracts import required_sha256_digest


MODULE4_EVOLUTION_POLICY_VERSION = "module4_evolution_snapshot_policy_v1"


@dataclass(frozen=True)
class Module4EvolutionSnapshotRequest:
    tenant_id: str
    source_run_id: str
    execution_mode: Literal[
        "historical_preview", "offline_evaluation", "production"
    ]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant_id", required_slug(self.tenant_id, label="tenant_id")
        )
        object.__setattr__(
            self,
            "source_run_id",
            required_text(self.source_run_id, label="source_run_id"),
        )
        if self.execution_mode not in {
            "historical_preview", "offline_evaluation", "production"
        }:
            raise ValueError("Unsupported Module 4 execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernedEvolutionCohortSnapshot:
    tenant_id: str
    snapshot_fingerprint: str
    cohort_snapshot_fingerprint: str
    source_run_id: str
    export_cohort_id: str
    quality_score: float
    freshness_status: str
    approval_status: str
    data_safety_status: str
    risk_decision: str
    monitoring_status: str
    reason_codes: Sequence[str]
    lifecycle_mutated: bool = False
    automatic_approval_performed: bool = False
    routing_enabled: bool = False
    eligible_for_activation: bool = False
    eligible_for_export: bool = False

    def __post_init__(self) -> None:
        required_slug(self.tenant_id, label="tenant_id")
        required_sha256_digest(
            self.snapshot_fingerprint, label="snapshot_fingerprint"
        )
        required_sha256_digest(
            self.cohort_snapshot_fingerprint,
            label="cohort_snapshot_fingerprint",
        )
        required_text(self.source_run_id, label="source_run_id")
        required_text(self.export_cohort_id, label="export_cohort_id")
        finite_unit_interval(self.quality_score, label="quality_score")
        for label, value in (
            ("freshness_status", self.freshness_status),
            ("approval_status", self.approval_status),
            ("data_safety_status", self.data_safety_status),
            ("risk_decision", self.risk_decision),
            ("monitoring_status", self.monitoring_status),
        ):
            required_text(value, label=label)
        if not self.reason_codes:
            raise ValueError("Evolution snapshots require reason codes.")
        if (
            self.lifecycle_mutated
            or self.automatic_approval_performed
            or self.routing_enabled
            or self.eligible_for_activation
            or self.eligible_for_export
        ):
            raise ValueError("Module 4.1 evidence cannot enable release state.")
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        assert_no_raw_identifier_fields(payload)
        return payload


@dataclass(frozen=True)
class GovernedEvolutionSnapshotReport:
    status: str
    policy_version: str
    request: Module4EvolutionSnapshotRequest
    snapshot_fingerprint: str
    source_cohort_count: int
    monitoring_count: int
    review_required_count: int
    paused_count: int
    blocked_count: int
    historical_count: int
    cohort_snapshots: Sequence[GovernedEvolutionCohortSnapshot]
    safety: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status != "engineering_preview_ready":
            raise ValueError("Unsupported Module 4.1 report status.")
        required_text(self.policy_version, label="policy_version")
        required_sha256_digest(
            self.snapshot_fingerprint, label="snapshot_fingerprint"
        )
        if self.source_cohort_count != len(self.cohort_snapshots):
            raise ValueError("Module 4.1 cohort accounting is inconsistent.")
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "policy_version": self.policy_version,
            "request": self.request.to_record(),
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "source_cohort_count": int(self.source_cohort_count),
            "monitoring_count": int(self.monitoring_count),
            "review_required_count": int(self.review_required_count),
            "paused_count": int(self.paused_count),
            "blocked_count": int(self.blocked_count),
            "historical_count": int(self.historical_count),
            "cohort_snapshots": [value.to_record() for value in self.cohort_snapshots],
            "safety": dict(self.safety),
        }
        assert_no_raw_identifier_fields(payload)
        return payload
