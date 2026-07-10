from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class LineageEvent:
    """
    One auditable transformation event.

    Example:
        source_csv -> contribution_bounded -> aggregated -> dp_safe -> synthetic_export
    """

    job_id: str
    stage: str
    transformation: str
    run_id: Optional[str] = None
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    dropped_rows: Optional[int] = None
    actor: str = "system"
    details: Optional[Dict[str, Any]] = None


class IngestionLineageService:
    """
    DB-backed lineage logger for Module 1.

    Why this exists:
        Roman/reviewers should be able to audit:

        source -> transformations -> privacy controls -> output

    Production value:
        - proves data handling path
        - shows where privacy controls were applied
        - helps debug data quality issues
        - supports compliance/review evidence
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def record_event(self, event: LineageEvent) -> Dict[str, Any]:
        self._validate_event(event)

        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for ingestion lineage.",
            }

        engine = create_engine(self._connection_url(db_url))

        created_at = datetime.now(timezone.utc).isoformat()
        clean_details = self._clean_json(event.details or {})

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)

            conn.execute(
                text(
                    """
                    INSERT INTO audience_lineage_events (
                        job_id,
                        run_id,
                        stage,
                        transformation,
                        input_ref,
                        output_ref,
                        input_rows,
                        output_rows,
                        dropped_rows,
                        actor,
                        details,
                        created_at
                    )
                    VALUES (
                        :job_id,
                        :run_id,
                        :stage,
                        :transformation,
                        :input_ref,
                        :output_ref,
                        :input_rows,
                        :output_rows,
                        :dropped_rows,
                        :actor,
                        :details,
                        :created_at
                    )
                    """
                ),
                {
                    "job_id": event.job_id,
                    "run_id": event.run_id,
                    "stage": event.stage,
                    "transformation": event.transformation,
                    "input_ref": event.input_ref,
                    "output_ref": event.output_ref,
                    "input_rows": event.input_rows,
                    "output_rows": event.output_rows,
                    "dropped_rows": event.dropped_rows,
                    "actor": event.actor,
                    "details": json.dumps(clean_details, sort_keys=True),
                    "created_at": created_at,
                },
            )

        return {
            "enabled": True,
            "status": "recorded",
            "job_id": event.job_id,
            "run_id": event.run_id,
            "stage": event.stage,
            "transformation": event.transformation,
            "created_at": created_at,
        }

    def list_events(self, *, job_id: str, limit: int = 100) -> Dict[str, Any]:
        if not job_id:
            raise ValueError("job_id is required")

        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for ingestion lineage.",
                "events": [],
            }

        engine = create_engine(self._connection_url(db_url))

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)

            rows = conn.execute(
                text(
                    """
                    SELECT
                        job_id,
                        run_id,
                        stage,
                        transformation,
                        input_ref,
                        output_ref,
                        input_rows,
                        output_rows,
                        dropped_rows,
                        actor,
                        details,
                        created_at
                    FROM audience_lineage_events
                    WHERE job_id = :job_id
                    ORDER BY id ASC
                    LIMIT :limit
                    """
                ),
                {
                    "job_id": job_id,
                    "limit": int(limit),
                },
            ).fetchall()

        events = [self._row_to_event(row) for row in rows]

        return {
            "enabled": True,
            "status": "ok",
            "job_id": job_id,
            "event_count": len(events),
            "events": events,
        }

    def build_chain_summary(self, *, job_id: str) -> Dict[str, Any]:
        listing = self.list_events(job_id=job_id)

        if not listing.get("enabled"):
            return listing

        events = listing.get("events", [])
        stages = [event.get("stage") for event in events]

        privacy_controls = []
        for event in events:
            stage = str(event.get("stage") or "").lower()
            transformation = str(event.get("transformation") or "").lower()
            details = event.get("details") or {}

            if "bounding" in stage or "bounding" in transformation:
                privacy_controls.append("contribution_bounding")

            if "dp" in stage or "differential" in transformation:
                privacy_controls.append("differential_privacy")

            if "synthetic" in stage or "synthetic" in transformation:
                privacy_controls.append("synthetic_generation")

            if details.get("k_anonymity_enforced") is True:
                privacy_controls.append("k_anonymity")

        return {
            "enabled": True,
            "status": "ok",
            "job_id": job_id,
            "event_count": len(events),
            "stages": stages,
            "privacy_controls_detected": sorted(set(privacy_controls)),
            "has_source": any(stage in {"source", "ingest", "raw_input"} for stage in stages),
            "has_privacy_step": bool(privacy_controls),
            "has_output": any(stage in {"output", "export", "synthetic_export"} for stage in stages),
        }

    def _ensure_table(self, conn, dialect_name: str) -> None:
        if dialect_name == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audience_lineage_events (
                        id BIGSERIAL PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        run_id TEXT,
                        stage TEXT NOT NULL,
                        transformation TEXT NOT NULL,
                        input_ref TEXT,
                        output_ref TEXT,
                        input_rows INTEGER,
                        output_rows INTEGER,
                        dropped_rows INTEGER,
                        actor TEXT NOT NULL DEFAULT 'system',
                        details JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audience_lineage_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        run_id TEXT,
                        stage TEXT NOT NULL,
                        transformation TEXT NOT NULL,
                        input_ref TEXT,
                        output_ref TEXT,
                        input_rows INTEGER,
                        output_rows INTEGER,
                        dropped_rows INTEGER,
                        actor TEXT NOT NULL DEFAULT 'system',
                        details TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                    """
                )
            )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_lineage_job
                ON audience_lineage_events(job_id)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_lineage_run
                ON audience_lineage_events(run_id)
                """
            )
        )

    def _row_to_event(self, row) -> Dict[str, Any]:
        data = dict(row._mapping)

        details = data.get("details")
        if isinstance(details, str):
            try:
                data["details"] = json.loads(details)
            except json.JSONDecodeError:
                data["details"] = {"raw": details}
        elif details is None:
            data["details"] = {}

        return data

    def _clean_json(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {str(key): self._clean_json(val) for key, val in value.items()}
            if isinstance(value, list):
                return [self._clean_json(item) for item in value]
            return str(value)

    def _validate_event(self, event: LineageEvent) -> None:
        if not event.job_id:
            raise ValueError("job_id is required")

        if not event.stage:
            raise ValueError("stage is required")

        if not event.transformation:
            raise ValueError("transformation is required")

        for name, value in {
            "input_rows": event.input_rows,
            "output_rows": event.output_rows,
            "dropped_rows": event.dropped_rows,
        }.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")

    def _db_url(self) -> Optional[str]:
        return (
            self._explicit_database_url
            or os.getenv("AUDIENCE_LINEAGE_DATABASE_URL")
            or os.getenv("AUDIENCE_HISTORY_DATABASE_URL")
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_DATABASE_URL")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DB_URL")
        )

    def _connection_url(self, db_url: str) -> str:
        if db_url.startswith("postgres://"):
            return "postgresql://" + db_url[len("postgres://") :]
        return db_url
