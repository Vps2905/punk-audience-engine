from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import required_slug
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


PREPRODUCTION_DEPLOYMENT_CERTIFICATION_POLICY_VERSION = (
    "preproduction_deployment_certification_policy_v1"
)

_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MIGRATION_HEAD_PATTERN = re.compile(r"^[0-9]{4}_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class PreproductionDeploymentCertificationRequest:
    tenant_id: str
    certification_id: str
    environment_name: str
    infrastructure_review_fingerprint: str
    candidate_image_digest: str
    expected_migration_head: str
    execution_mode: Literal["preproduction"] = "preproduction"

    def __post_init__(self) -> None:
        for field in ("tenant_id", "certification_id", "environment_name"):
            object.__setattr__(
                self,
                field,
                required_slug(getattr(self, field), label=field),
            )
        if self.environment_name in {"prod", "production", "live"}:
            raise ValueError(
                "Deployment certification must target an isolated "
                "preproduction environment."
            )
        object.__setattr__(
            self,
            "infrastructure_review_fingerprint",
            required_sha256_digest(
                self.infrastructure_review_fingerprint,
                label="infrastructure_review_fingerprint",
            ),
        )
        image_digest = str(self.candidate_image_digest or "").strip()
        if not _IMAGE_DIGEST_PATTERN.fullmatch(image_digest):
            raise ValueError(
                "candidate_image_digest must be an immutable sha256 digest."
            )
        object.__setattr__(self, "candidate_image_digest", image_digest)
        migration_head = str(self.expected_migration_head or "").strip()
        if not _MIGRATION_HEAD_PATTERN.fullmatch(migration_head):
            raise ValueError("expected_migration_head is invalid.")
        object.__setattr__(self, "expected_migration_head", migration_head)
        if self.execution_mode != "preproduction":
            raise ValueError(
                "Deployment certification supports preproduction only."
            )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
