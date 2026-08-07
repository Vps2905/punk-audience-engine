from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.production_infrastructure_status_service import (
    ProductionInfrastructureStatusService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/infrastructure",
    tags=["Audience Intelligence Production Infrastructure Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def production_infrastructure_status(
    _tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict:
    return ProductionInfrastructureStatusService().status()
