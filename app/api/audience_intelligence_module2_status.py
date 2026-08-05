from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.services.production_module2_status_service import (
    ProductionModule2StatusService,
)

router = APIRouter(
    prefix="/api/audience-intelligence/module-2",
    tags=["Audience Intelligence Module 2 Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def module_2_status() -> dict:
    return ProductionModule2StatusService().status()
