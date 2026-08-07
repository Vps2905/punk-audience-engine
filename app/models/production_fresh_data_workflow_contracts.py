from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    stable_digest,
)
from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
    EmbeddingModelSpec,
    ProductionFeatureBuildRequest,
)
from app.models.production_module3_cohort_contracts import (
    Module3CohortCandidatePolicy,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationPolicy,
)


@dataclass(frozen=True)
class ProductionFreshDataWorkflowRequest:
    """Immutable Module 1 -> Module 2 -> Module 3 workflow contract.

    This contract accepts only a completed, privacy-safe canonical object
    produced by Module 1. It cannot request activation, export, proposal
    creation, or lookalike generation.
    """

    ingestion_id: str
    source: CanonicalFeatureSourceManifest
    model: EmbeddingModelSpec
    cohort_policy: Module3CohortCandidatePolicy = (
        Module3CohortCandidatePolicy()
    )
    overlap_policy: Module3OverlapDeduplicationPolicy = (
        Module3OverlapDeduplicationPolicy()
    )
    batch_size: int = 256
    max_features: int = 1_000_000
    requested_by: str = "fresh_data_workflow_worker"
    approval_required: bool = True
    activation_requested: bool = False
    export_requested: bool = False
    lookalike_generation_requested: bool = False
    contract_version: str = "production-fresh-data-workflow-v1"

    def __post_init__(self) -> None:
        ingestion_id = " ".join(str(self.ingestion_id or "").split())
        if not ingestion_id:
            raise ValueError("ingestion_id is required.")
        object.__setattr__(self, "ingestion_id", ingestion_id)

        requested_by = normalize_taxonomy_value(self.requested_by)
        if not requested_by:
            raise ValueError("requested_by is required.")
        object.__setattr__(self, "requested_by", requested_by)

        if not 1 <= int(self.batch_size) <= 4096:
            raise ValueError("batch_size must be between 1 and 4096.")
        if not 1 <= int(self.max_features) <= 10_000_000:
            raise ValueError("max_features must be between 1 and 10000000.")
        if self.approval_required is not True:
            raise ValueError("Fresh-data workflows always require approval.")
        if (
            self.activation_requested
            or self.export_requested
            or self.lookalike_generation_requested
        ):
            raise ValueError(
                "Fresh-data workflows cannot request lookalikes, activation, "
                "or export."
            )
        if self.source.data_use_mode not in {
            "offline_evaluation",
            "production",
        }:
            raise ValueError(
                "Fresh-data workflows require offline_evaluation or production "
                "canonical input."
            )

    @property
    def feature_build_request(self) -> ProductionFeatureBuildRequest:
        return ProductionFeatureBuildRequest(
            source=self.source,
            model=self.model,
            batch_size=int(self.batch_size),
            max_features=int(self.max_features),
            requested_by=self.requested_by,
            activation_requested=False,
        )

    @property
    def request_fingerprint(self) -> str:
        return stable_digest(
            {
                "contract_version": self.contract_version,
                "ingestion_id": self.ingestion_id,
                "source": self.source.to_safe_dict(),
                "model": self.model.to_safe_dict(),
                "cohort_policy": self.cohort_policy.to_record(),
                "overlap_policy": self.overlap_policy.to_record(),
                "batch_size": int(self.batch_size),
                "max_features": int(self.max_features),
                "requested_by": self.requested_by,
                "approval_required": True,
                "activation_requested": False,
                "export_requested": False,
                "lookalike_generation_requested": False,
            }
        )

    @property
    def workflow_id(self) -> str:
        return "fresh_data_workflow_" + self.request_fingerprint[:32]

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "workflow_id": self.workflow_id,
            "request_fingerprint": self.request_fingerprint,
            "ingestion_id": self.ingestion_id,
            "source": self.source.to_safe_dict(),
            "model": self.model.to_safe_dict(),
            "cohort_policy": self.cohort_policy.to_record(),
            "overlap_policy": self.overlap_policy.to_record(),
            "batch_size": int(self.batch_size),
            "max_features": int(self.max_features),
            "requested_by": self.requested_by,
            "approval_required": True,
            "activation_requested": False,
            "export_requested": False,
            "lookalike_generation_requested": False,
        }
