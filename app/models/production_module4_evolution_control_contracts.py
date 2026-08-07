from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import (
    required_slug,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


MODULE4_DRIFT_POLICY_VERSION = "module4_aggregate_drift_policy_v1"
MODULE4_RECOMMENDATION_POLICY_VERSION = (
    "module4_evolution_recommendation_policy_v1"
)
MODULE4_APPROVAL_SHADOW_POLICY_VERSION = (
    "module4_manual_approval_shadow_policy_v1"
)
MODULE4_RECOVERY_POLICY_VERSION = "module4_recovery_planning_policy_v1"


@dataclass(frozen=True)
class Module4DriftAnalysisRequest:
    tenant_id: str
    baseline_snapshot_fingerprint: str
    current_snapshot_fingerprint: str
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
            "baseline_snapshot_fingerprint",
            required_sha256_digest(
                self.baseline_snapshot_fingerprint,
                label="baseline_snapshot_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "current_snapshot_fingerprint",
            required_sha256_digest(
                self.current_snapshot_fingerprint,
                label="current_snapshot_fingerprint",
            ),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 4 drift execution_mode.")
        if (
            self.baseline_snapshot_fingerprint
            == self.current_snapshot_fingerprint
        ):
            raise ValueError("Module 4 drift snapshots must be distinct.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
