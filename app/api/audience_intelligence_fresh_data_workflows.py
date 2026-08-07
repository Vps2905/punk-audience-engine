from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.services.production_fresh_data_workflow_status_service import (
    ProductionFreshDataWorkflowStatusService,
)

router = APIRouter(
    prefix="/api/audience-intelligence/fresh-data-workflows",
    tags=["Audience Intelligence Fresh Data Workflows"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def fresh_data_workflow_status() -> dict:
    return ProductionFreshDataWorkflowStatusService().status()
