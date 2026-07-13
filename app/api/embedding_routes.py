from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query

from app.core.api_key_auth import require_audience_api_key
from app.services.embedding_service import embed_processed_job, embed_records, search_similar_audiences
from app.services.clustering_service import cluster_job_vectors


router = APIRouter(
    tags=["Module 2 - Embeddings & Feature Store"],
    dependencies=[
        Depends(require_audience_api_key),
    ],
)


class EmbedRecordsRequest(BaseModel):
    job_id: str = Field(..., description="Embedding job ID")
    records: List[Dict[str, Any]] = Field(..., min_length=1, description="Privacy-safe feature/cohort records")


class SimilarSearchRequest(BaseModel):
    job_id: str = Field(
        ...,
        description="Embedding job ID",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="Audience intent query",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
    )
    location_name: Optional[str] = Field(
        default=None,
        description="Exact normalized location filter",
    )
    primary_poi_type: Optional[str] = Field(
        default=None,
        description="POI taxonomy value or alias",
    )
    created_day_part: Optional[str] = Field(
        default=None,
        description="Exact daypart filter",
    )
    min_quality: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


@router.post("/embed")
def embed_records_route(request: EmbedRecordsRequest):
    """
    Production-safe embedding from already privacy-safe records.
    """
    return embed_records(
        job_id=request.job_id,
        records=request.records,
    )


@router.post("/embed/{job_id}")
def embed_job(job_id: str):
    """
    Generate embeddings for processed feature table.
    """
    return embed_processed_job(job_id)


@router.post("/search/similar")
def search_similar(
    request: SimilarSearchRequest,
):
    """
    Search privacy-safe audience rows using semantic
    similarity and structured metadata filters.
    """
    return search_similar_audiences(
        job_id=request.job_id,
        query=request.query,
        top_k=request.top_k,
        location_name=request.location_name,
        primary_poi_type=request.primary_poi_type,
        created_day_part=request.created_day_part,
        min_quality=request.min_quality,
    )


@router.post("/cluster/{job_id}")
def cluster_job(
    job_id: str,
    n_clusters: int = Query(default=3, ge=1, le=20)
):
    """
    Cluster audience vectors.
    """
    return cluster_job_vectors(
        job_id=job_id,
        n_clusters=n_clusters
    )
