from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    required_text,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


MODULE3_SHADOW_POLICY_VERSION = "module3_punk_ai_shadow_policy_v1"

ShadowAlignmentStatus = Literal[
    "aligned_historical_preview",
    "aligned_manual_shadow_review",
    "blocked_lifecycle",
    "unmatched_candidate",
    "ambiguous_candidate",
]


@dataclass(frozen=True)
class Module3ShadowObservationRequest:
    tenant_id: str
    proposal_id: str
    overlap_report_fingerprint: str
    lifecycle_evaluation_fingerprint: str
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
            "proposal_id",
            required_text(self.proposal_id, label="proposal_id"),
        )
        object.__setattr__(
            self,
            "overlap_report_fingerprint",
            required_sha256_digest(
                self.overlap_report_fingerprint,
                label="overlap_report_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "lifecycle_evaluation_fingerprint",
            required_sha256_digest(
                self.lifecycle_evaluation_fingerprint,
                label="lifecycle_evaluation_fingerprint",
            ),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 3.6 execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PunkAIShadowCandidateObservation:
    tenant_id: str
    observation_fingerprint: str
    candidate_observation_fingerprint: str
    proposal_id: str
    proposal_rank: int
    feature_id: str
    candidate_id: str | None
    candidate_version: int | None
    candidate_fingerprint: str | None
    recommended_lifecycle_status: str | None
    alignment_status: ShadowAlignmentStatus
    reason_codes: Sequence[str]
    proposal_modified: bool = False
    candidate_lifecycle_mutated: bool = False
    shadow_routing_enabled: bool = False
    eligible_for_activation: bool = False
    eligible_for_export: bool = False

    def __post_init__(self) -> None:
        required_slug(self.tenant_id, label="tenant_id")
        required_sha256_digest(
            self.observation_fingerprint,
            label="observation_fingerprint",
        )
        required_sha256_digest(
            self.candidate_observation_fingerprint,
            label="candidate_observation_fingerprint",
        )
        required_text(self.proposal_id, label="proposal_id")
        if int(self.proposal_rank) < 1:
            raise ValueError("proposal_rank must be >= 1.")
        required_text(self.feature_id, label="feature_id")
        identity_values = (
            self.candidate_id,
            self.candidate_version,
            self.candidate_fingerprint,
        )
        if any(value is not None for value in identity_values):
            if not all(value is not None for value in identity_values):
                raise ValueError("Mapped candidate identity is incomplete.")
            required_text(self.candidate_id, label="candidate_id")
            if int(self.candidate_version or 0) < 1:
                raise ValueError("candidate_version must be >= 1.")
            required_sha256_digest(
                self.candidate_fingerprint,
                label="candidate_fingerprint",
            )
        if self.alignment_status not in {
            "aligned_historical_preview",
            "aligned_manual_shadow_review",
            "blocked_lifecycle",
            "unmatched_candidate",
            "ambiguous_candidate",
        }:
            raise ValueError("Unsupported Module 3.6 alignment status.")
        if not self.reason_codes:
            raise ValueError("Shadow observations require reason codes.")
        if (
            self.proposal_modified
            or self.candidate_lifecycle_mutated
            or self.shadow_routing_enabled
            or self.eligible_for_activation
            or self.eligible_for_export
        ):
            raise ValueError("Module 3.6 cannot enable release state.")
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        assert_no_raw_identifier_fields(payload)
        return payload


@dataclass(frozen=True)
class PunkAIShadowObservationReport:
    status: str
    policy_version: str
    request: Module3ShadowObservationRequest
    observation_fingerprint: str
    proposal_status: str
    proposal_candidate_count: int
    observation_count: int
    aligned_candidate_count: int
    blocked_candidate_count: int
    unmatched_candidate_count: int
    ambiguous_candidate_count: int
    shadow_alignment_passed: bool
    observations: Sequence[PunkAIShadowCandidateObservation]
    safety: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.6 report status.")
        required_text(self.policy_version, label="policy_version")
        required_sha256_digest(
            self.observation_fingerprint,
            label="observation_fingerprint",
        )
        required_text(self.proposal_status, label="proposal_status")
        for label, value in (
            ("proposal_candidate_count", self.proposal_candidate_count),
            ("observation_count", self.observation_count),
            ("aligned_candidate_count", self.aligned_candidate_count),
            ("blocked_candidate_count", self.blocked_candidate_count),
            ("unmatched_candidate_count", self.unmatched_candidate_count),
            ("ambiguous_candidate_count", self.ambiguous_candidate_count),
        ):
            if int(value) < 0:
                raise ValueError(f"{label} must be >= 0.")
        if int(self.observation_count) != len(self.observations):
            raise ValueError("observation_count does not match observations.")
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "policy_version": self.policy_version,
            "request": self.request.to_record(),
            "observation_fingerprint": self.observation_fingerprint,
            "proposal_status": self.proposal_status,
            "proposal_candidate_count": int(self.proposal_candidate_count),
            "observation_count": int(self.observation_count),
            "aligned_candidate_count": int(self.aligned_candidate_count),
            "blocked_candidate_count": int(self.blocked_candidate_count),
            "unmatched_candidate_count": int(self.unmatched_candidate_count),
            "ambiguous_candidate_count": int(self.ambiguous_candidate_count),
            "shadow_alignment_passed": bool(self.shadow_alignment_passed),
            "observations": [value.to_record() for value in self.observations],
            "safety": dict(self.safety),
        }
        assert_no_raw_identifier_fields(payload)
        return payload
