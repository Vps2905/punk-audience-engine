from fastapi import APIRouter, Query

from app.services.synthetic_service import generate_synthetic_for_job


router = APIRouter(tags=["Module 1B - Synthetic Data"])


@router.post("/synthetic/generate/{job_id}")
def generate_synthetic(
    job_id: str,
    num_rows: int = Query(default=1000, ge=1, le=100000),
    use_sdv: bool = Query(default=True)
):
    """
    Generate privacy-safe synthetic audience profiles from processed features.
    """
    return generate_synthetic_for_job(
        job_id=job_id,
        num_rows=num_rows,
        use_sdv=use_sdv
    )
