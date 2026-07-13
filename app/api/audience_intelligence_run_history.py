from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.api_key_auth import require_audience_api_key
from app.services.audience_run_history_service import AudienceRunHistoryService


router = APIRouter(
    prefix="/api/audience-intelligence/runs",
    tags=["Audience Intelligence Run History"],
    dependencies=[Depends(require_audience_api_key)],
)


class RunDecisionRequest(BaseModel):
    actor: str = Field(default="internal_reviewer", min_length=2)
    note: Optional[str] = None
    downstream_export_enabled: bool = True


@router.get("")
def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    approval_status: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    result = AudienceRunHistoryService().list_runs(
        limit=limit,
        offset=offset,
        approval_status=approval_status,
        status=status,
    )

    if result.get("status") == "skipped":
        raise HTTPException(status_code=503, detail=result)

    return result


@router.get("/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    result = AudienceRunHistoryService().get_run(run_id)

    if result.get("status") == "skipped":
        raise HTTPException(status_code=503, detail=result)

    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return result


@router.get("/{run_id}/cohorts")
def list_run_cohorts(
    run_id: str,
    location: Optional[str] = None,
    poi_type: Optional[str] = None,
    daypart: Optional[str] = None,
    approval_status: Optional[str] = None,
    min_quality: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> Dict[str, Any]:
    result = AudienceRunHistoryService().list_cohorts(
        run_id=run_id,
        location=location,
        poi_type=poi_type,
        daypart=daypart,
        approval_status=approval_status,
        min_quality=min_quality,
        limit=limit,
        offset=offset,
    )

    if result.get("status") == "skipped":
        raise HTTPException(
            status_code=503,
            detail=result,
        )

    if result.get("status") == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}",
        )

    return result


@router.get("/{run_id}/audit")
def get_run_audit(run_id: str) -> Dict[str, Any]:
    result = AudienceRunHistoryService().get_audit(run_id)

    if result.get("status") == "skipped":
        raise HTTPException(status_code=503, detail=result)

    return result


@router.post("/{run_id}/approve")
def approve_run(run_id: str, request: RunDecisionRequest) -> Dict[str, Any]:
    result = AudienceRunHistoryService().approve_run(
        run_id=run_id,
        actor=request.actor,
        note=request.note,
        downstream_export_enabled=request.downstream_export_enabled,
    )

    if result.get("status") == "skipped":
        raise HTTPException(status_code=503, detail=result)

    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if result.get("status") == "blocked":
        raise HTTPException(status_code=400, detail=result)

    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result)

    return result


@router.post("/{run_id}/reject")
def reject_run(run_id: str, request: RunDecisionRequest) -> Dict[str, Any]:
    result = AudienceRunHistoryService().reject_run(
        run_id=run_id,
        actor=request.actor,
        note=request.note,
    )

    if result.get("status") == "skipped":
        raise HTTPException(status_code=503, detail=result)

    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result)

    return result
