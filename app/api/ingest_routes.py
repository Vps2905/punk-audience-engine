from fastapi import APIRouter, Depends, File, UploadFile, Query

from app.services.ingestion_service import ingest_csv, get_job_status
from app.core.api_key_auth import require_audience_api_key



router = APIRouter(tags=["Module 1 - Ingestion & Privacy"], dependencies=[Depends(require_audience_api_key)])


@router.post("/ingest")
def ingest_file(
    file: UploadFile = File(...),
    k_min: int = Query(default=1000, ge=1),
    epsilon: float = Query(default=1.0, gt=0)
):
    return ingest_csv(file=file, k_min=k_min, epsilon=epsilon)


@router.get("/status/{job_id}")
def status(job_id: str):
    return get_job_status(job_id)
