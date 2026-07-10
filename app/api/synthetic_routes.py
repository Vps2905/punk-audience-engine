from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.synthetic_service import generate_synthetic_for_job
from app.core.api_key_auth import require_audience_api_key


router = APIRouter(tags=["Module 1B - Synthetic Data"], dependencies=[Depends(require_audience_api_key)])


@router.post("/synthetic/generate/{job_id}")
def generate_synthetic(
    job_id: str,
    num_rows: int = Query(default=1000, ge=1, le=100000),
    use_sdv: bool = Query(default=False),
    production_mode: bool = Query(default=True),
    allow_fallback: bool = Query(default=False),
):
    """
    Generate privacy-safe synthetic audience profiles from processed features.

    Production default:
    - aggregate-safe generation only
    - no SDV Gaussian path
    - no silent fallback
    """
    try:
        return generate_synthetic_for_job(
            job_id=job_id,
            num_rows=num_rows,
            use_sdv=use_sdv,
            production_mode=production_mode,
            allow_fallback=allow_fallback,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
