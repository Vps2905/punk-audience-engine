from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import create_engine, text

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence",
    tags=["Audience Intelligence Source Health"],
    dependencies=[Depends(require_audience_api_key)],
)


SOURCE_DB_ENVS = [
    "ECHO_DATABASE_URL",
    "DATABASE_URL",
    "POSTGRES_URL",
    "POSTGRES_DATABASE_URL",
    "SUPABASE_DB_URL",
    "DB_URL",
]


def _source_db_url() -> Optional[str]:
    for name in SOURCE_DB_ENVS:
        value = os.getenv(name)
        if value:
            return value
    return None


def _as_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        parsed = value

    if getattr(parsed, "tzinfo", None) is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _age_hours(value: Optional[datetime]) -> Optional[float]:
    if value is None:
        return None
    return round((datetime.now(timezone.utc) - value).total_seconds() / 3600, 2)


@router.get("/source-health")
def source_health(limit_days: int = 20) -> Dict[str, Any]:
    """
    Read-only source data health check.

    This endpoint reports freshness for the real audience source table:
    public.maid_extractions.

    It does not expose DB credentials and it does not modify source data.
    """
    db_url = _source_db_url()

    if not db_url:
        return {
            "enabled": False,
            "status": "not_configured",
            "reason": "no_source_database_url",
            "source_table": "public.maid_extractions",
            "database_env_presence": {name: bool(os.getenv(name)) for name in SOURCE_DB_ENVS},
        }

    threshold_hours = 48.0
    limit_days = max(1, min(int(limit_days or 20), 90))

    engine = create_engine(db_url)

    try:
        with engine.connect() as conn:
            summary = conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS rows,
                        COUNT(DISTINCT session_id) AS sessions,
                        COALESCE(SUM(COALESCE(maid_count, 0)), 0) AS total_maid_count,
                        MAX(created_at) AS latest_created_at,
                        MIN(created_at) AS oldest_created_at
                    FROM public.maid_extractions
                    """
                )
            ).mappings().first()

            daily_rows = conn.execute(
                text(
                    """
                    SELECT
                        DATE(created_at) AS source_date,
                        COUNT(*) AS rows,
                        COUNT(DISTINCT session_id) AS sessions,
                        COALESCE(SUM(COALESCE(maid_count, 0)), 0) AS total_maid_count,
                        MAX(created_at) AS latest_created_at
                    FROM public.maid_extractions
                    GROUP BY DATE(created_at)
                    ORDER BY source_date DESC
                    LIMIT :limit_days
                    """
                ),
                {"limit_days": limit_days},
            ).mappings().all()

    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "source_table": "public.maid_extractions",
            "error": str(exc)[:300],
            "database_env_presence": {name: bool(os.getenv(name)) for name in SOURCE_DB_ENVS},
        }

    latest = _as_utc(summary.get("latest_created_at") if summary else None)
    oldest = _as_utc(summary.get("oldest_created_at") if summary else None)
    age = _age_hours(latest)

    if latest is None:
        freshness_status = "no_data"
        reason = "No source rows found in public.maid_extractions."
    elif age is not None and age <= threshold_hours:
        freshness_status = "fresh"
        reason = "Latest source timestamp is within freshness threshold."
    else:
        freshness_status = "stale"
        reason = "Latest source timestamp is older than 48 hours."

    recent_source_dates: List[Dict[str, Any]] = []
    for row in daily_rows:
        row_latest = _as_utc(row.get("latest_created_at"))
        recent_source_dates.append(
            {
                "source_date": str(row.get("source_date")),
                "rows": int(row.get("rows") or 0),
                "sessions": int(row.get("sessions") or 0),
                "total_maid_count": int(row.get("total_maid_count") or 0),
                "latest_created_at": row_latest.isoformat() if row_latest else None,
            }
        )

    return {
        "enabled": True,
        "status": "ok",
        "source_table": "public.maid_extractions",
        "freshness_status": freshness_status,
        "freshness_threshold_hours": threshold_hours,
        "latest_source_timestamp": latest.isoformat() if latest else None,
        "oldest_source_timestamp": oldest.isoformat() if oldest else None,
        "source_age_hours": age,
        "source_rows": int(summary.get("rows") or 0) if summary else 0,
        "source_sessions": int(summary.get("sessions") or 0) if summary else 0,
        "total_maid_count": int(summary.get("total_maid_count") or 0) if summary else 0,
        "recent_source_dates": recent_source_dates,
        "reason": reason,
        "diagnosis": (
            "Source audience table is stale. Pipeline should block approval/export until upstream MAID extraction refresh runs."
            if freshness_status == "stale"
            else "Source audience table is usable for freshness gate."
        ),
        "source_owner_hint": (
            "This repo reads public.maid_extractions. It does not appear to own the upstream insert/update refresh job."
        ),
        "database_env_presence": {name: bool(os.getenv(name)) for name in SOURCE_DB_ENVS},
    }
