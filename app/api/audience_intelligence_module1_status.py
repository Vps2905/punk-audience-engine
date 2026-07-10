from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence/module-1",
    tags=["Audience Intelligence Module 1 Status"],
    dependencies=[Depends(require_audience_api_key)],
)


def _env_present(*names: str) -> Dict[str, bool]:
    return {name: bool(os.getenv(name)) for name in names}


@router.get("/status")
def module_1_status() -> Dict[str, Any]:
    """
    Module 1 readiness/status endpoint.

    Important:
        This endpoint reports configuration presence only.
        It never returns raw secret values.
    """
    database_envs = _env_present(
        "AUDIENCE_INGESTION_DATABASE_URL",
        "AUDIENCE_LINEAGE_DATABASE_URL",
        "AUDIENCE_PRIVACY_BUDGET_DATABASE_URL",
        "AUDIENCE_HISTORY_DATABASE_URL",
        "ECHO_DATABASE_URL",
        "DATABASE_URL",
        "POSTGRES_URL",
        "POSTGRES_DATABASE_URL",
        "SUPABASE_DB_URL",
        "DB_URL",
    )

    api_key_configured = bool(os.getenv("AUDIENCE_API_KEY"))

    has_any_database = any(database_envs.values())

    return {
        "module": "module_1_ingestion_privacy_layer",
        "status": "ready_for_preproduction_review",
        "api_key_configured": api_key_configured,
        "database_configured": has_any_database,
        "database_env_presence": database_envs,
        "privacy_controls": {
            "salted_hashing": True,
            "contribution_bounding": True,
            "k_anonymity": True,
            "differential_privacy_noise": True,
            "privacy_budget_ledger": True,
            "privacy_budget_approval_enforcement": True,
            "lineage_logging": True,
            "ingestion_job_tracking": True,
            "synthetic_generation_audit": True,
            "legacy_route_auth_protection": True,
        },
        "protected_endpoints": [
            "POST /api/audience-intelligence/ingest",
            "POST /api/audience-intelligence/ingest/csv",
            "GET /api/audience-intelligence/ingest/{job_id}/status",
            "POST /api/audience-intelligence/synthetic/generate/{job_id}",
            "GET /api/audience-intelligence/synthetic/{synthetic_job_id}/lineage",
            "GET /api/audience-intelligence/module-1/status",
            "GET /api/audience-intelligence/source-health",
        ],
        "legacy_routes_hardened": [
            "POST /ingest",
            "GET /status/{job_id}",
            "POST /synthetic/generate/{job_id}",
            "POST /audience/generate",
        ],
        "test_status": {
            "latest_known_local_result": "default local: 180 passed, 3 skipped, 3 warnings; with AUDIENCE_TEST_DATABASE_URL: 183 passed, 3 warnings",
            "postgres_only_tests": "3 Postgres integration tests skipped by default; pass when AUDIENCE_TEST_DATABASE_URL is set",
            "warnings_are_blocking": False,
        },
        "production_notes": [
            "Use a real Postgres database for production run history and approval integration tests.",
            "Rotate exposed API keys before any push or deployment.",
            "Move CREATE TABLE logic to Alembic migrations before final production release.",
            "Use production-grade secret management instead of raw env values in shared terminals.",
        ],
    }
