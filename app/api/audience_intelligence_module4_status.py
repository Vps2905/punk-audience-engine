from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.production_module4_status_service import (
    ProductionModule4StatusService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/module-4",
    tags=["Audience Intelligence Module 4 Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def module4_status(
    _tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict:
    return ProductionModule4StatusService().status()
