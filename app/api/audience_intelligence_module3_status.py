from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key
from app.services.production_module3_status_service import (
    ProductionModule3StatusService,
)

router = APIRouter(
    prefix="/api/audience-intelligence/module-3",
    tags=["Audience Intelligence Module 3 Status"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.get("/status")
def module_3_status() -> dict:
    return ProductionModule3StatusService().status()
