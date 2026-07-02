from typing import Optional, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.cohort_service import create_cohort, get_cohort
from app.services.lookalike_service import create_lookalike


router = APIRouter(tags=["Module 3 - Cohorts & Lookalike"])


class CohortCreateRequest(BaseModel):
    job_id: str = Field(..., description="Job ID from /ingest and /embed")
    name: str = Field(..., description="Human-readable cohort name")
    query: Optional[str] = Field(default=None, description="Semantic audience query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Exact metadata filters")
    top_k: int = Field(default=20, ge=1, le=100)


class LookalikeRequest(BaseModel):
    cohort_id: str = Field(..., description="Source cohort ID")
    top_k: int = Field(default=20, ge=1, le=100)


@router.post("/cohort/create")
def cohort_create(request: CohortCreateRequest):
    """
    Create a privacy-safe audience cohort.
    """
    return create_cohort(
        job_id=request.job_id,
        name=request.name,
        query=request.query,
        filters=request.filters,
        top_k=request.top_k
    )


@router.get("/cohort/query/{cohort_id}")
def cohort_query(cohort_id: str):
    """
    Read saved cohort by ID.
    """
    return get_cohort(cohort_id)


@router.post("/cohort/lookalike")
def cohort_lookalike(request: LookalikeRequest):
    """
    Create lookalike audience from existing cohort.
    """
    return create_lookalike(
        cohort_id=request.cohort_id,
        top_k=request.top_k
    )
