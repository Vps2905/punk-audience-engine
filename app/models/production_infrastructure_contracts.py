from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import required_slug


INFRASTRUCTURE_ASSESSMENT_POLICY_VERSION = (
    "production_infrastructure_assessment_policy_v1"
)
INFRASTRUCTURE_REVIEW_POLICY_VERSION = (
    "production_infrastructure_change_set_review_policy_v1"
)


@dataclass(frozen=True)
class ProductionInfrastructureAssessmentRequest:
    tenant_id: str
    assessment_id: str
    environment_name: str
    execution_mode: Literal[
        "historical_preview",
        "offline_evaluation",
        "preproduction",
    ]

    def __post_init__(self) -> None:
        for field in ("tenant_id", "assessment_id", "environment_name"):
            object.__setattr__(
                self,
                field,
                required_slug(getattr(self, field), label=field),
            )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "preproduction",
        }:
            raise ValueError("Unsupported infrastructure execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
