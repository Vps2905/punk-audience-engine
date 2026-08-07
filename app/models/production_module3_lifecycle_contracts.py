from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    finite_unit_interval,
    required_slug,
    required_text,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


MODULE3_LIFECYCLE_POLICY_VERSION = "module3_lifecycle_monitoring_policy_v1"

LifecycleRecommendation = Literal[
    "historical_preview_only",
    "shadow_review_pending",
    "review_required_overlap",
    "review_required_sensitive_poi",
    "paused_stale_source",
    "paused_quality_degraded",
    "blocked_policy",
    "blocked_rights",
]


@dataclass(frozen=True)
class Module3LifecyclePolicy:
    min_quality_for_shadow_review: float = 0.70
    max_quality_drop: float = 0.15
    max_input_candidates: int = 1000
    policy_version: str = MODULE3_LIFECYCLE_POLICY_VERSION

    def __post_init__(self) -> None:
        finite_unit_interval(
            self.min_quality_for_shadow_review,
            label="min_quality_for_shadow_review",
        )
        finite_unit_interval(
            self.max_quality_drop,
            label="max_quality_drop",
        )
        if not 1 <= int(self.max_input_candidates) <= 5000:
            raise ValueError(
                "max_input_candidates must be between 1 and 5000."
            )
        required_text(self.policy_version, label="policy_version")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Module3LifecycleEvaluationRequest:
    tenant_id: str
    overlap_report_fingerprint: str
    lookalike_report_fingerprint: str
    execution_mode: Literal[
        "historical_preview",
        "offline_evaluation",
        "production",
    ]
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
        object.__setattr__(
            self,
            "lookalike_report_fingerprint",
            required_sha256_digest(
                self.lookalike_report_fingerprint,
                label="lookalike_report_fingerprint",
            ),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 3 lifecycle execution_mode.")
        object.__setattr__(
            self,
            "purpose",
            required_slug(self.purpose, label="purpose"),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernedLifecycleRecommendation:
    tenant_id: str
    evaluation_fingerprint: str
    recommendation_fingerprint: str
    candidate_id: str
    candidate_version: int
    candidate_fingerprint: str
    current_lifecycle_status: str
    recommended_lifecycle_status: LifecycleRecommendation
    quality_score: float
    previous_quality_score: float | None
    quality_delta: float | None
    reason_codes: Sequence[str]
    manual_approval_required: bool = True
    monitoring_required: bool = True
    lifecycle_mutated: bool = False
    shadow_routing_enabled: bool = False
    eligible_for_activation: bool = False
    eligible_for_export: bool = False

    def __post_init__(self) -> None:
        required_slug(self.tenant_id, label="tenant_id")
        required_sha256_digest(
            self.evaluation_fingerprint,
            label="evaluation_fingerprint",
        )
        required_sha256_digest(
            self.recommendation_fingerprint,
            label="recommendation_fingerprint",
        )
        required_text(self.candidate_id, label="candidate_id")
        if int(self.candidate_version) < 1:
            raise ValueError("candidate_version must be >= 1.")
        required_sha256_digest(
            self.candidate_fingerprint,
            label="candidate_fingerprint",
        )
        required_text(
            self.current_lifecycle_status,
            label="current_lifecycle_status",
        )
        if self.recommended_lifecycle_status not in {
            "historical_preview_only",
            "shadow_review_pending",
            "review_required_overlap",
            "review_required_sensitive_poi",
            "paused_stale_source",
            "paused_quality_degraded",
            "blocked_policy",
            "blocked_rights",
        }:
            raise ValueError("Unsupported lifecycle recommendation.")
        finite_unit_interval(self.quality_score, label="quality_score")
        if self.previous_quality_score is not None:
            finite_unit_interval(
                self.previous_quality_score,
                label="previous_quality_score",
            )
        if self.quality_delta is not None and not -1.0 <= float(
            self.quality_delta
        ) <= 1.0:
            raise ValueError("quality_delta must be between -1 and 1.")
        if not self.reason_codes:
            raise ValueError("Lifecycle recommendations require reason codes.")
        if self.manual_approval_required is not True:
            raise ValueError("Lifecycle recommendations require manual approval.")
        if self.monitoring_required is not True:
            raise ValueError("Lifecycle recommendations require monitoring.")
        if (
            self.lifecycle_mutated
            or self.shadow_routing_enabled
            or self.eligible_for_activation
            or self.eligible_for_export
        ):
            raise ValueError(
                "Module 3.5 evidence cannot mutate or enable release state."
            )
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        assert_no_raw_identifier_fields(payload)
        return payload


@dataclass(frozen=True)
class GovernedLifecycleEvaluationReport:
    status: str
    request: Module3LifecycleEvaluationRequest
    policy: Module3LifecyclePolicy
    evaluation_fingerprint: str
    source_candidate_count: int
    recommendation_count: int
    shadow_review_pending_count: int
    review_required_count: int
    paused_count: int
    blocked_count: int
    historical_preview_count: int
    recommendations: Sequence[GovernedLifecycleRecommendation]
    safety: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.5 report status.")
        required_sha256_digest(
            self.evaluation_fingerprint,
            label="evaluation_fingerprint",
        )
        for label, value in (
            ("source_candidate_count", self.source_candidate_count),
            ("recommendation_count", self.recommendation_count),
            ("shadow_review_pending_count", self.shadow_review_pending_count),
            ("review_required_count", self.review_required_count),
            ("paused_count", self.paused_count),
            ("blocked_count", self.blocked_count),
            ("historical_preview_count", self.historical_preview_count),
        ):
            if int(value) < 0:
                raise ValueError(f"{label} must be >= 0.")
        if int(self.recommendation_count) != len(self.recommendations):
            raise ValueError(
                "recommendation_count does not match recommendation rows."
            )
        assert_no_raw_identifier_fields(self.to_record())

    def to_record(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "request": self.request.to_record(),
            "policy": self.policy.to_record(),
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "source_candidate_count": int(self.source_candidate_count),
            "recommendation_count": int(self.recommendation_count),
            "shadow_review_pending_count": int(
                self.shadow_review_pending_count
            ),
            "review_required_count": int(self.review_required_count),
            "paused_count": int(self.paused_count),
            "blocked_count": int(self.blocked_count),
            "historical_preview_count": int(self.historical_preview_count),
            "recommendations": [
                recommendation.to_record()
                for recommendation in self.recommendations
            ],
            "safety": dict(self.safety),
        }
        assert_no_raw_identifier_fields(payload)
        return payload
