from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.audience_feature_proposal_service import (
    AudienceFeatureProposalService,
)
from app.services.durable_audience_feature_proposal_service import (
    DurableAudienceFeatureProposalService,
)
from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)
from app.services.punk_ai_audience_proposal_store_service import (
    ProposalIdempotencyConflictError,
    PunkAIAudienceProposalStore,
)


class CampaignBudget(BaseModel):
    currency: str | None = Field(default=None, min_length=3, max_length=8)
    daily: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)


class PunkAIAudienceProposalRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    campaign_id: str = Field(min_length=1, max_length=256)
    objective: str = Field(min_length=1, max_length=128)
    audience_intent: str = Field(min_length=1, max_length=4000)
    locations: list[str] = Field(default_factory=list, max_length=100)
    categories: list[str] = Field(default_factory=list, max_length=100)
    dayparts: list[str] = Field(default_factory=list, max_length=24)
    budget: CampaignBudget = Field(default_factory=CampaignBudget)
    exclusions: list[str] = Field(default_factory=list, max_length=100)
    destination: str = Field(default="meta", min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=256)
    execution_mode: Literal["historical_preview", "production"] = (
        "historical_preview"
    )
    feature_set_id: str | None = Field(default=None, max_length=256)
    feature_set_version: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=10, ge=1, le=50)


router = APIRouter(
    prefix="/api/audience-intelligence/punk-ai/v1",
    tags=["Punk AI Audience Proposal Contract"],
    dependencies=[Depends(require_audience_api_key)],
)


def build_audience_feature_store() -> PgvectorAudienceFeatureStore:
    database_url = os.getenv("AUDIENCE_FEATURE_DATABASE_URL")
    return PgvectorAudienceFeatureStore(
        database_url=database_url,
    )


def build_audience_proposal_store() -> PunkAIAudienceProposalStore:
    return PunkAIAudienceProposalStore(
        database_url=os.getenv("AUDIENCE_PROPOSAL_DATABASE_URL"),
    )


def build_audience_feature_proposal_service() -> DurableAudienceFeatureProposalService:
    return DurableAudienceFeatureProposalService(
        delegate=AudienceFeatureProposalService(
            feature_store=build_audience_feature_store()
        ),
        proposal_store=build_audience_proposal_store(),
    )


@router.get("/feature-sets/current")
def current_feature_set(
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
    feature_set_id: str | None = Query(default=None, max_length=256),
    feature_set_version: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    try:
        feature_set = build_audience_feature_store().get_feature_set(
            tenant_id=verified_tenant_id,
            feature_set_id=feature_set_id,
            version=feature_set_version,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    source_latest_at = feature_set.get("source_latest_at")
    if source_latest_at is not None and hasattr(source_latest_at, "isoformat"):
        source_latest_at = source_latest_at.isoformat()
    return {
        "contract_version": AudienceFeatureProposalService.CONTRACT_VERSION,
        "tenant_id": verified_tenant_id,
        "feature_set_id": feature_set.get("feature_set_id"),
        "version": feature_set.get("version"),
        "status": feature_set.get("status"),
        "source_mode": feature_set.get("source_mode"),
        "data_use_mode": feature_set.get("data_use_mode"),
        "source_latest_at": source_latest_at,
        "freshness_status": feature_set.get("freshness_status"),
        "feature_count": feature_set.get("feature_count"),
        "model_backend": feature_set.get("model_backend"),
        "model_name": feature_set.get("model_name"),
        "model_version": feature_set.get("model_version"),
        "privacy_policy_version": feature_set.get("privacy_policy_version"),
        "rights_policy_id": feature_set.get("rights_policy_id"),
        "purpose": feature_set.get("purpose"),
        "eligible_for_retrieval": bool(
            feature_set.get("eligible_for_retrieval")
        ),
        "eligible_for_activation": bool(
            feature_set.get("eligible_for_activation")
        ),
        "downstream_export_enabled": False,
    }


@router.post("/proposals")
def create_audience_proposal(
    request: PunkAIAudienceProposalRequest,
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict[str, Any]:
    """
    Versioned Punk AI -> Audience Intelligence proposal boundary.

    The authenticated tenant header must match the tenant in the request.
    This endpoint only proposes audiences. It never performs an export.
    """
    if verified_tenant_id != request.tenant_id.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant header does not match request tenant.",
        )

    try:
        return build_audience_feature_proposal_service().propose(
            request.model_dump()
        )
    except ProposalIdempotencyConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
