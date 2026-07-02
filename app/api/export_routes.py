from fastapi import APIRouter, Query

from app.services.meta_export_service import generate_meta_safe_export, get_export_status


router = APIRouter(tags=["Module 4 - Meta Safe Export"])


@router.post("/export/meta/{cohort_id}")
def export_meta(
    cohort_id: str,
    seed_limit: int = Query(default=1000, ge=1, le=100000),
    approval_status: str = Query(default="pending_approval")
):
    """
    Generate Meta-safe export package from cohort.
    """
    return generate_meta_safe_export(
        cohort_id=cohort_id,
        seed_limit=seed_limit,
        approval_status=approval_status
    )


@router.get("/export/status/{export_id}")
def export_status(export_id: str):
    """
    Check export package status.
    """
    return get_export_status(export_id)
