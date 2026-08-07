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


MODULE3_LOOKALIKE_POLICY_VERSION = "module3_governed_lookalike_policy_v2"

ExecutionMode = Literal[
    "historical_preview",
    "offline_evaluation",
    "production",
]


@dataclass(frozen=True)
class Module3LookalikePolicy:
    """Aggregate-only Module 3.4 review policy."""

    max_targets_per_seed: int = 3
    min_seed_quality_score: float = 0.60
    min_target_quality_score: float = 0.60
    min_similarity_score: float = 0.75
    require_same_poi_type: bool = True
    require_same_daypart: bool = True
    require_distinct_location: bool = True
    exclude_overlap_review_candidates: bool = True
    policy_version: str = MODULE3_LOOKALIKE_POLICY_VERSION

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_targets_per_seed) <= 20:
            raise ValueError("max_targets_per_seed must be between 1 and 20.")

        finite_unit_interval(
            self.min_seed_quality_score,
            label="min_seed_quality_score",
        )
        finite_unit_interval(
            self.min_target_quality_score,
            label="min_target_quality_score",
        )
        finite_unit_interval(
            self.min_similarity_score,
            label="min_similarity_score",
        )
        required_text(self.policy_version, label="policy_version")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Module3LookalikeRequest:
    tenant_id: str
    overlap_report_fingerprint: str
    execution_mode: ExecutionMode
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "overlap_report_fingerprint",
            required_sha256_digest(
                self.overlap_report_fingerprint,
                label="overlap_report_fingerprint",
            ),
        )

        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 3.4 execution_mode.")

        object.__setattr__(
            self,
            "purpose",
            required_slug(self.purpose, label="purpose"),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernedLookalikePair:
    lookalike_id: str
    lookalike_version: int
    lookalike_fingerprint: str
    seed_candidate_id: str
    seed_candidate_version: int
    seed_candidate_fingerprint: str
    target_candidate_id: str
    target_candidate_version: int
    target_candidate_fingerprint: str
    similarity_score: float
    similarity_components: Mapping[str, float]
    reason_codes: Sequence[str]
    review_required: bool = True
    membership_generated: bool = False
    eligible_for_activation: bool = False
    eligible_for_export: bool = False

    def __post_init__(self) -> None:
        required_text(self.lookalike_id, label="lookalike_id")
        required_text(self.seed_candidate_id, label="seed_candidate_id")
        required_text(self.target_candidate_id, label="target_candidate_id")

        if int(self.lookalike_version) < 1:
            raise ValueError("lookalike_version must be >= 1.")
        if int(self.seed_candidate_version) < 1:
            raise ValueError("seed_candidate_version must be >= 1.")
        if int(self.target_candidate_version) < 1:
            raise ValueError("target_candidate_version must be >= 1.")

        required_sha256_digest(
            self.lookalike_fingerprint,
            label="lookalike_fingerprint",
        )
        required_sha256_digest(
            self.seed_candidate_fingerprint,
            label="seed_candidate_fingerprint",
        )
        required_sha256_digest(
            self.target_candidate_fingerprint,
            label="target_candidate_fingerprint",
        )

        if (
            self.seed_candidate_id == self.target_candidate_id
            and self.seed_candidate_version == self.target_candidate_version
        ):
            raise ValueError("A lookalike pair cannot target its seed candidate.")

        finite_unit_interval(
            self.similarity_score,
            label="similarity_score",
        )

        if not self.similarity_components:
            raise ValueError("similarity_components are required.")

        for component_name, value in self.similarity_components.items():
            required_text(component_name, label="similarity component name")
            finite_unit_interval(
                value,
                label=f"similarity component {component_name}",
            )

        if not self.reason_codes:
            raise ValueError("At least one lookalike reason code is required.")

        for reason_code in self.reason_codes:
            required_slug(reason_code, label="reason_code")

        if self.review_required is not True:
            raise ValueError("Module 3.4 lookalikes must require review.")

        if self.membership_generated:
            raise ValueError("Module 3.4 cannot generate audience membership.")

        if self.eligible_for_activation or self.eligible_for_export:
            raise ValueError(
                "Module 3.4 lookalikes cannot be activation/export eligible."
            )

        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "lookalike_id": self.lookalike_id,
            "lookalike_version": int(self.lookalike_version),
            "lookalike_fingerprint": self.lookalike_fingerprint,
            "seed_candidate_id": self.seed_candidate_id,
            "seed_candidate_version": int(self.seed_candidate_version),
            "seed_candidate_fingerprint": self.seed_candidate_fingerprint,
            "target_candidate_id": self.target_candidate_id,
            "target_candidate_version": int(self.target_candidate_version),
            "target_candidate_fingerprint": self.target_candidate_fingerprint,
            "similarity_score": float(self.similarity_score),
            "similarity_components": dict(self.similarity_components),
            "reason_codes": list(self.reason_codes),
            "review_required": self.review_required,
            "membership_generated": self.membership_generated,
            "eligible_for_activation": self.eligible_for_activation,
            "eligible_for_export": self.eligible_for_export,
        }
        assert_no_raw_identifier_fields(payload)
        return payload


@dataclass(frozen=True)
class GovernedLookalikeReport:
    status: str
    request: Module3LookalikeRequest
    policy: Module3LookalikePolicy
    report_fingerprint: str
    source_retained_candidate_count: int
    excluded_overlap_review_candidate_count: int
    excluded_policy_candidate_count: int
    eligible_seed_count: int
    generated_lookalike_candidate_count: int
    lookalike_candidates: Sequence[GovernedLookalikePair]
    safety: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.4 report status.")

        required_sha256_digest(
            self.report_fingerprint,
            label="report_fingerprint",
        )

        counts = {
            "source_retained_candidate_count": self.source_retained_candidate_count,
            "excluded_overlap_review_candidate_count": (
                self.excluded_overlap_review_candidate_count
            ),
            "excluded_policy_candidate_count": self.excluded_policy_candidate_count,
            "eligible_seed_count": self.eligible_seed_count,
            "generated_lookalike_candidate_count": (
                self.generated_lookalike_candidate_count
            ),
        }

        for label, value in counts.items():
            if int(value) < 0:
                raise ValueError(f"{label} must be >= 0.")

        if int(self.generated_lookalike_candidate_count) != len(
            self.lookalike_candidates
        ):
            raise ValueError(
                "generated_lookalike_candidate_count does not match output rows."
            )

        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "request": self.request.to_record(),
            "policy": self.policy.to_record(),
            "report_fingerprint": self.report_fingerprint,
            "source_retained_candidate_count": int(
                self.source_retained_candidate_count
            ),
            "excluded_overlap_review_candidate_count": int(
                self.excluded_overlap_review_candidate_count
            ),
            "excluded_policy_candidate_count": int(
                self.excluded_policy_candidate_count
            ),
            "eligible_seed_count": int(self.eligible_seed_count),
            "generated_lookalike_candidate_count": int(
                self.generated_lookalike_candidate_count
            ),
            "lookalike_candidates": [
                candidate.to_record()
                for candidate in self.lookalike_candidates
            ],
            "safety": dict(self.safety),
        }
        assert_no_raw_identifier_fields(payload)
        return payload
