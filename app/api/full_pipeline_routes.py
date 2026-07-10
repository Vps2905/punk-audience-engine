from fastapi import APIRouter, Depends, File, UploadFile, Form

from app.services.full_pipeline_service import run_full_audience_pipeline
from app.core.api_key_auth import require_audience_api_key



router = APIRouter(tags=["Simple Product Flow"], dependencies=[Depends(require_audience_api_key)])


@router.post("/audience/generate")
def generate_audience(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    k_min: int = Form(default=1000),
    epsilon: float = Form(default=1.0),
    synthetic_rows: int = Form(default=1000),
    seed_limit: int = Form(default=1000)
):
    """
    Product-facing endpoint.

    Upload CSV + prompt -> audience cohort + Meta-safe export.
    """
    return run_full_audience_pipeline(
        file=file,
        prompt=prompt,
        k_min=k_min,
        epsilon=epsilon,
        synthetic_rows=synthetic_rows,
        seed_limit=seed_limit
    )
