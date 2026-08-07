from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.services.production_module5_status_service import (
    ProductionModule5StatusService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/module-5",
    tags=["Audience Intelligence Module 5 Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def module5_status(
    _tenant_id: str = Depends(require_verified_audience_tenant),
) -> dict:
    return ProductionModule5StatusService().status()
