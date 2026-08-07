from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.production_observability_status_service import (
    ProductionObservabilityStatusService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/observability",
    tags=["Audience Intelligence Production Observability Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def production_observability_status(
    _tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict:
    return ProductionObservabilityStatusService().status()
