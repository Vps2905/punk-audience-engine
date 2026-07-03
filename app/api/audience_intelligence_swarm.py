from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.agents.audience_swarm_monitor_agent import AudienceSwarmMonitorAgent

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence/swarm",
    tags=["Audience Intelligence Swarm"],
    dependencies=[Depends(require_audience_api_key)],
)


class SwarmRunRequest(BaseModel):
    stale_pending_hours: int = 24
    max_failed_jobs_warning: int = 1


@router.post("/run")
def run_swarm_monitor(request: SwarmRunRequest) -> Dict[str, Any]:
    agent = AudienceSwarmMonitorAgent()
    return agent.run_once(
        stale_pending_hours=request.stale_pending_hours,
        max_failed_jobs_warning=request.max_failed_jobs_warning,
    )


@router.get("/health")
def swarm_health() -> Dict[str, Any]:
    agent = AudienceSwarmMonitorAgent()
    return agent.run_once()
