from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from app.core.api_key_auth import require_audience_api_key
from app.core.production_guardrails import (
    local_file_storage_allowed,
)
from app.services.meta_export_service import (
    generate_meta_safe_export,
    get_export_status,
)


router = APIRouter(
    tags=["Module 4 - Meta Safe Export"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.post("/export/meta/{cohort_id}")
def export_meta(
    cohort_id: str,
    seed_limit: int = Query(
        default=1000,
        ge=1,
        le=100000,
    ),
    approval_status: str = Query(
        default="pending_approval",
    ),
):
    """
    Legacy local Meta package route.

    Production callers must use:
    /api/audience-intelligence/modules/export/meta/{cohort_id}
    with a run_id.
    """
    if not local_file_storage_allowed():
        raise HTTPException(
            status_code=404,
            detail=(
                "Legacy local Meta export is disabled in "
                "production. Use the DB-backed module endpoint."
            ),
        )

    if approval_status != "pending_approval":
        raise HTTPException(
            status_code=400,
            detail=(
                "Caller-supplied approval_status cannot authorize "
                "an export. Legacy packages always remain "
                "pending_approval."
            ),
        )

    return generate_meta_safe_export(
        cohort_id=cohort_id,
        seed_limit=seed_limit,
        approval_status="pending_approval",
    )


@router.get("/export/status/{export_id}")
def export_status(export_id: str):
    if not local_file_storage_allowed():
        raise HTTPException(
            status_code=404,
            detail=(
                "Legacy local export status is disabled "
                "in production."
            ),
        )

    return get_export_status(export_id)
