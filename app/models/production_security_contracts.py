from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import required_slug


SECURITY_POSTURE_POLICY_VERSION = "production_security_posture_policy_v1"
SECURITY_REVIEW_POLICY_VERSION = "production_security_manual_review_policy_v1"


@dataclass(frozen=True)
class ProductionSecurityAssessmentRequest:
    tenant_id: str
    assessment_id: str
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
            "assessment_id",
            required_slug(self.assessment_id, label="assessment_id"),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported security assessment execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
