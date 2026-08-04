from __future__ import annotations

from app.models.production_audience_retrieval_contracts import (
    GovernedAudienceRetrievalRequest,
)
from app.services.production_dual_model_candidate_retrieval_service import (
    ProductionDualModelCandidateRetrievalService,
)
from app.services.production_multilingual_constraint_canonicalization_service import (
    ProductionMultilingualConstraintCanonicalizationService,
)


class ProductionGovernedAudienceRetrievalService:
    """Offline-first Module 2.1 orchestration with no proposal side effects."""

    def __init__(
        self,
        *,
        canonicalizer: ProductionMultilingualConstraintCanonicalizationService,
        candidate_retrieval: ProductionDualModelCandidateRetrievalService,
    ) -> None:
        self._canonicalizer = canonicalizer
        self._candidate_retrieval = candidate_retrieval

    def retrieve(
        self,
        request: GovernedAudienceRetrievalRequest,
    ) -> dict:
        constraints = self._canonicalizer.canonicalize(request)
        result = self._candidate_retrieval.retrieve(
            request=request,
            constraints=constraints,
        )
        result["module"] = "module_2_1_governed_multilingual_retrieval"
        result["automatic_proposal_creation_enabled"] = False
        result["existing_proposal_flow_modified"] = False
        result["model_registration_performed"] = False
        result["threshold_changed"] = False
        return result
