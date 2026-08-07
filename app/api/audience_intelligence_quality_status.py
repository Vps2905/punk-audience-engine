from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.production_quality_monitoring_status_service import (
    ProductionQualityMonitoringStatusService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/production-quality",
    tags=["Audience Intelligence Production Quality Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def production_quality_status(
    _tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict:
    return ProductionQualityMonitoringStatusService().status()
