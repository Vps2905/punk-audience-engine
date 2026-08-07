from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.production_security_status_service import (
    ProductionSecurityStatusService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/security",
    tags=["Audience Intelligence Production Security Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def production_security_status(
    _tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict:
    return ProductionSecurityStatusService().status()
