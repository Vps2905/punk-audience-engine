from fastapi import APIRouter, File, UploadFile, Query

from app.services.ingestion_service import ingest_csv, get_job_status


router = APIRouter(tags=["Module 1 - Ingestion & Privacy"])


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
