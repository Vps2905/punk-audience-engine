from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    stable_digest,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec


@dataclass(frozen=True)
class HistoricalSemanticFeatureBuildRequest:
    tenant_id: str
    source_feature_set_id: str
    source_feature_set_version: int
    model: EmbeddingModelSpec
    minimum_cohort_size: int = 1000
    batch_size: int = 256
    requested_by: str = "module5_historical_evidence_operator"
    contract_version: str = "historical-semantic-feature-build-v1"

    def __post_init__(self) -> None:
        tenant_id = normalize_taxonomy_value(self.tenant_id)
        source_feature_set_id = " ".join(
            str(self.source_feature_set_id or "").split()
        )
        requested_by = normalize_taxonomy_value(self.requested_by)
        if not tenant_id or not source_feature_set_id or not requested_by:
            raise ValueError(
                "tenant_id, source_feature_set_id and requested_by are required."
            )
        if int(self.source_feature_set_version) < 1:
            raise ValueError("source_feature_set_version must be at least 1.")
        if int(self.minimum_cohort_size) < 1000:
            raise ValueError("minimum_cohort_size must be at least 1000.")
        if not 1 <= int(self.batch_size) <= 4096:
            raise ValueError("batch_size must be between 1 and 4096.")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(
            self,
            "source_feature_set_id",
            source_feature_set_id,
        )
        object.__setattr__(
            self,
            "source_feature_set_version",
            int(self.source_feature_set_version),
        )
        object.__setattr__(
            self,
            "minimum_cohort_size",
            int(self.minimum_cohort_size),
        )
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "requested_by", requested_by)

    @property
    def request_fingerprint(self) -> str:
        return stable_digest(self.to_safe_dict(include_fingerprint=False))

    def to_safe_dict(
        self,
        *,
        include_fingerprint: bool = True,
    ) -> dict[str, Any]:
        result = {
            "contract_version": self.contract_version,
            "tenant_id": self.tenant_id,
            "source_feature_set_id": self.source_feature_set_id,
            "source_feature_set_version": self.source_feature_set_version,
            "model": self.model.to_safe_dict(),
            "minimum_cohort_size": self.minimum_cohort_size,
            "batch_size": self.batch_size,
            "requested_by": self.requested_by,
            "activation_requested": False,
        }
        if include_fingerprint:
            result["request_fingerprint"] = self.request_fingerprint
        return result
