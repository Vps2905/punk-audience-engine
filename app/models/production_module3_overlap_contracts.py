from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    required_text,
)

MODULE3_OVERLAP_POLICY_VERSION = "module3_overlap_dedup_policy_v1"
MODULE3_MAX_INPUT_CANDIDATES = 1000

OverlapGroupType = Literal[
    "potential_constraint_overlap",
    "potential_temporal_overlap",
]


def required_sha256_digest(value: Any, *, label: str) -> str:
    raw_digest = required_text(value, label=label)
    digest = raw_digest.lower()
    if raw_digest != digest or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


@dataclass(frozen=True)
class Module3OverlapDeduplicationPolicy:
    max_input_candidates: int = MODULE3_MAX_INPUT_CANDIDATES
    max_group_members: int = 250
    min_potential_overlap_group_size: int = 2
    policy_version: str = MODULE3_OVERLAP_POLICY_VERSION

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_input_candidates) <= 5000:
            raise ValueError("max_input_candidates must be between 1 and 5000.")
        if not 2 <= int(self.max_group_members) <= 1000:
            raise ValueError("max_group_members must be between 2 and 1000.")
        if not 2 <= int(self.min_potential_overlap_group_size) <= 100:
            raise ValueError(
                "min_potential_overlap_group_size must be between 2 and 100."
            )
        required_text(self.policy_version, label="policy_version")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Module3OverlapDeduplicationRequest:
    tenant_id: str
    batch_fingerprint: str
    execution_mode: Literal["historical_preview", "offline_evaluation", "production"]
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        fingerprint = required_sha256_digest(
            self.batch_fingerprint,
            label="batch_fingerprint",
        )
        object.__setattr__(self, "batch_fingerprint", fingerprint)
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 3 overlap execution_mode.")
        object.__setattr__(
            self,
            "purpose",
            required_slug(self.purpose, label="purpose"),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactDuplicateSuppression:
    retained_candidate_id: str
    retained_candidate_version: int
    candidate_fingerprint: str
    suppressed_occurrence_count: int
    reason: Literal["identical_candidate_fingerprint"] = (
        "identical_candidate_fingerprint"
    )

    def __post_init__(self) -> None:
        required_text(self.retained_candidate_id, label="retained_candidate_id")
        if int(self.retained_candidate_version) < 1:
            raise ValueError("retained_candidate_version must be >= 1.")
        required_sha256_digest(
            self.candidate_fingerprint,
            label="candidate_fingerprint",
        )
        if int(self.suppressed_occurrence_count) < 1:
            raise ValueError("suppressed_occurrence_count must be >= 1.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PotentialOverlapGroup:
    group_id: str
    group_version: int
    group_fingerprint: str
    group_type: OverlapGroupType
    normalized_signature: Mapping[str, str | None]
    candidate_refs: Sequence[Mapping[str, Any]]
    review_required: bool = True
    overlap_estimate_available: bool = False
    unique_reach_claimed: bool = False

    def __post_init__(self) -> None:
        required_text(self.group_id, label="group_id")
        if int(self.group_version) < 1:
            raise ValueError("group_version must be >= 1.")
        required_sha256_digest(
            self.group_fingerprint,
            label="group_fingerprint",
        )
        if self.group_type not in {
            "potential_constraint_overlap",
            "potential_temporal_overlap",
        }:
            raise ValueError("Unsupported Module 3 overlap group type.")
        if len(self.candidate_refs) < 2:
            raise ValueError("Potential overlap groups require at least 2 candidates.")
        if self.review_required is not True:
            raise ValueError("Potential overlap groups always require review.")
        if self.overlap_estimate_available or self.unique_reach_claimed:
            raise ValueError(
                "Module 3.3 cannot estimate overlap or claim unique reach."
            )
        for candidate_ref in self.candidate_refs:
            required_text(
                candidate_ref.get("candidate_id"),
                label="candidate_ref.candidate_id",
            )
            if int(candidate_ref.get("candidate_version") or 0) < 1:
                raise ValueError("candidate_ref.candidate_version must be >= 1.")
            required_sha256_digest(
                candidate_ref.get("candidate_fingerprint"),
                label="candidate_ref.candidate_fingerprint",
            )
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "group_id": self.group_id,
            "group_version": int(self.group_version),
            "group_fingerprint": self.group_fingerprint,
            "group_type": self.group_type,
            "normalized_signature": dict(self.normalized_signature),
            "candidate_refs": [dict(value) for value in self.candidate_refs],
            "candidate_count": len(self.candidate_refs),
            "review_required": self.review_required,
            "overlap_estimate_available": self.overlap_estimate_available,
            "unique_reach_claimed": self.unique_reach_claimed,
        }
        assert_no_raw_identifier_fields(payload)
        return payload


@dataclass(frozen=True)
class GovernedOverlapDeduplicationReport:
    status: str
    request: Module3OverlapDeduplicationRequest
    policy: Module3OverlapDeduplicationPolicy
    report_fingerprint: str
    source_candidate_count: int
    retained_candidate_count: int
    exact_duplicate_group_count: int
    suppressed_exact_duplicate_occurrence_count: int
    potential_overlap_group_count: int
    candidates_requiring_overlap_review_count: int
    retained_candidates: Sequence[Mapping[str, Any]]
    exact_duplicate_suppressions: Sequence[ExactDuplicateSuppression]
    overlap_groups: Sequence[PotentialOverlapGroup]
    safety: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.3 report status.")
        required_sha256_digest(
            self.report_fingerprint,
            label="report_fingerprint",
        )
        for label, value in (
            ("source_candidate_count", self.source_candidate_count),
            ("retained_candidate_count", self.retained_candidate_count),
            ("exact_duplicate_group_count", self.exact_duplicate_group_count),
            (
                "suppressed_exact_duplicate_occurrence_count",
                self.suppressed_exact_duplicate_occurrence_count,
            ),
            ("potential_overlap_group_count", self.potential_overlap_group_count),
            (
                "candidates_requiring_overlap_review_count",
                self.candidates_requiring_overlap_review_count,
            ),
        ):
            if int(value) < 0:
                raise ValueError(f"{label} must be >= 0.")
        if int(self.retained_candidate_count) != len(self.retained_candidates):
            raise ValueError("retained_candidate_count does not match output rows.")
        if int(self.exact_duplicate_group_count) != len(
            self.exact_duplicate_suppressions
        ):
            raise ValueError(
                "exact_duplicate_group_count does not match suppressions."
            )
        if int(self.potential_overlap_group_count) != len(self.overlap_groups):
            raise ValueError("potential_overlap_group_count does not match groups.")
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "request": self.request.to_record(),
            "policy": self.policy.to_record(),
            "report_fingerprint": self.report_fingerprint,
            "source_candidate_count": int(self.source_candidate_count),
            "retained_candidate_count": int(self.retained_candidate_count),
            "exact_duplicate_group_count": int(self.exact_duplicate_group_count),
            "suppressed_exact_duplicate_occurrence_count": int(
                self.suppressed_exact_duplicate_occurrence_count
            ),
            "potential_overlap_group_count": int(
                self.potential_overlap_group_count
            ),
            "candidates_requiring_overlap_review_count": int(
                self.candidates_requiring_overlap_review_count
            ),
            "retained_candidates": [
                dict(candidate) for candidate in self.retained_candidates
            ],
            "exact_duplicate_suppressions": [
                suppression.to_record()
                for suppression in self.exact_duplicate_suppressions
            ],
            "overlap_groups": [group.to_record() for group in self.overlap_groups],
            "safety": dict(self.safety),
        }
        assert_no_raw_identifier_fields(payload)
        return payload
