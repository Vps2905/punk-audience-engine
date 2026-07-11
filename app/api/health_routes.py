from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.core.environment_validation import validate_environment

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "Audience Intelligence Engine",
    }


@router.get("/ready")
def ready(response: Response) -> dict:
    result = validate_environment()

    if not result.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if result.ok else "not_ready",
        "mode": result.mode,
        "missing_required": result.missing_required,
        "warnings": result.warnings,
        "env_presence": result.env_presence,
    }
