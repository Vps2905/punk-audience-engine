from typing import Any, Dict, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, Query

from app.services.embedding_service import embed_processed_job, embed_records, search_similar_audiences
from app.services.clustering_service import cluster_job_vectors


router = APIRouter(tags=["Module 2 - Embeddings & Feature Store"])


class EmbedRecordsRequest(BaseModel):
    job_id: str = Field(..., description="Embedding job ID")
    records: List[Dict[str, Any]] = Field(..., min_length=1, description="Privacy-safe feature/cohort records")


class SimilarSearchRequest(BaseModel):
    job_id: str = Field(..., description="Job ID from /ingest")
    query: str = Field(..., description="Audience intent query")
    top_k: int = Field(default=5, ge=1, le=20)


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
def search_similar(request: SimilarSearchRequest):
    """
    Search similar audience rows by meaning.
    """
    return search_similar_audiences(
        job_id=request.job_id,
        query=request.query,
        top_k=request.top_k
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
