from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from app.models.audience_feature_contracts import normalize_taxonomy_value

MODULE3_POLICY_VERSION = "module3_cohort_candidate_policy_v1"
MODULE3_MIN_COHORT_SIZE = 1000
MODULE3_BLOCKED_OUTPUT_TOKENS = {
    "maid",
    "raw_maid",
    "device_id",
    "advertising_id",
    "email",
    "phone",
    "latitude",
    "longitude",
    "raw_identifier",
    "individual_id",
}

CandidateLifecycleStatus = Literal[
    "historical_preview_only",
    "quality_review_pending",
    "review_required_sensitive_poi",
    "blocked_sensitive_poi",
]


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def stable_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def finite_unit_interval(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def required_slug(value: Any, *, label: str) -> str:
    normalized = normalize_taxonomy_value(value)
    if not normalized:
        raise ValueError(f"{label} is required.")
    return normalized


def required_text(value: Any, *, label: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def assert_no_raw_identifier_fields(payload: Any) -> None:
    """Reject raw/individual identifier field names anywhere in an output payload."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = normalize_taxonomy_value(key)
            if normalized in MODULE3_BLOCKED_OUTPUT_TOKENS:
                raise ValueError(
                    "Module 3 output contains a prohibited raw identifier field: "
                    f"{key}"
                )
            assert_no_raw_identifier_fields(value)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            assert_no_raw_identifier_fields(value)


@dataclass(frozen=True)
class Module3CohortCandidatePolicy:
    min_cohort_size: int = MODULE3_MIN_COHORT_SIZE
    max_candidates: int = 250
    source_quality_weight: float = 0.45
    size_adequacy_weight: float = 0.25
    freshness_weight: float = 0.15
    completeness_weight: float = 0.10
    privacy_weight: float = 0.05
    policy_version: str = MODULE3_POLICY_VERSION

    def __post_init__(self) -> None:
        if int(self.min_cohort_size) < MODULE3_MIN_COHORT_SIZE:
            raise ValueError(
                "Governed cohort generation requires min_cohort_size >= 1000."
            )
        if not 1 <= int(self.max_candidates) <= 1000:
            raise ValueError("max_candidates must be between 1 and 1000.")
        weights = (
            self.source_quality_weight,
            self.size_adequacy_weight,
            self.freshness_weight,
            self.completeness_weight,
            self.privacy_weight,
        )
        for index, weight in enumerate(weights):
            finite_unit_interval(weight, label=f"quality weight {index}")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Module 3 quality weights must sum to 1.0.")
        required_text(self.policy_version, label="policy_version")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Module3CohortGenerationRequest:
    tenant_id: str
    feature_set_id: str
    feature_set_version: int
    execution_mode: Literal["historical_preview", "offline_evaluation", "production"]
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "feature_set_id",
            required_text(self.feature_set_id, label="feature_set_id"),
        )
        if int(self.feature_set_version) < 1:
            raise ValueError("feature_set_version must be >= 1.")
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 3 execution_mode.")
        object.__setattr__(
            self,
            "purpose",
            required_slug(self.purpose, label="purpose"),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernedCohortCandidate:
    tenant_id: str
    candidate_id: str
    candidate_version: int
    candidate_fingerprint: str
    batch_fingerprint: str
    source_feature_set_id: str
    source_feature_set_version: int
    source_feature_id: str
    location_name: str
    primary_poi_type: str
    created_day_part: str
    lookback_bucket: str | None
    cohort_size: int
    source_quality_score: float
    quality_score: float
    quality_components: Mapping[str, float]
    metric_disclosure: Mapping[str, Sequence[str]]
    privacy_status: str
    privacy_decision: str
    sensitive_poi_decision: str
    rights_status: str
    freshness_status: str
    data_use_mode: str
    lifecycle_status: CandidateLifecycleStatus
    approval_required: bool
    eligible_for_activation: bool
    eligible_for_export: bool
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        if int(self.candidate_version) < 1:
            raise ValueError("candidate_version must be >= 1.")
        if int(self.cohort_size) < MODULE3_MIN_COHORT_SIZE:
            raise ValueError("candidate cohort_size must be >= 1000.")
        finite_unit_interval(self.source_quality_score, label="source_quality_score")
        finite_unit_interval(self.quality_score, label="quality_score")
        if self.eligible_for_activation or self.eligible_for_export:
            raise ValueError(
                "Module 3.1-3.2 candidates cannot be activation/export eligible."
            )
        if len(self.candidate_fingerprint) != 64 or len(self.batch_fingerprint) != 64:
            raise ValueError("Candidate and batch fingerprints must be SHA-256 values.")
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_components"] = dict(self.quality_components)
        payload["metric_disclosure"] = {
            key: list(value) for key, value in self.metric_disclosure.items()
        }
        payload["lineage"] = dict(self.lineage)
        return payload


@dataclass(frozen=True)
class GovernedCohortCandidateBatch:
    status: str
    request: Module3CohortGenerationRequest
    policy: Module3CohortCandidatePolicy
    batch_fingerprint: str
    source_feature_count: int
    generated_candidate_count: int
    truncated_candidate_count: int
    excluded_below_k_count: int
    excluded_unsafe_privacy_count: int
    excluded_ineligible_retrieval_count: int
    excluded_invalid_count: int
    blocked_sensitive_count: int
    review_required_sensitive_count: int
    candidates: Sequence[GovernedCohortCandidate]
    safety: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "request": self.request.to_record(),
            "policy": self.policy.to_record(),
            "batch_fingerprint": self.batch_fingerprint,
            "source_feature_count": int(self.source_feature_count),
            "generated_candidate_count": int(self.generated_candidate_count),
            "truncated_candidate_count": int(self.truncated_candidate_count),
            "excluded_below_k_count": int(self.excluded_below_k_count),
            "excluded_unsafe_privacy_count": int(self.excluded_unsafe_privacy_count),
            "excluded_ineligible_retrieval_count": int(
                self.excluded_ineligible_retrieval_count
            ),
            "excluded_invalid_count": int(self.excluded_invalid_count),
            "blocked_sensitive_count": int(self.blocked_sensitive_count),
            "review_required_sensitive_count": int(
                self.review_required_sensitive_count
            ),
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "safety": dict(self.safety),
        }
        assert_no_raw_identifier_fields(payload)
        return payload
