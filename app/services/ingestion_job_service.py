from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class IngestionJobCreate:
    """
    Creates a trackable ingestion job.

    source_type examples:
        csv
        api
        postgres_safe_derived
        synthetic_generation
    """

    source_type: str
    source_ref: Optional[str] = None
    run_id: Optional[str] = None
    actor: str = "system"
    metadata: Optional[Dict[str, Any]] = None


class IngestionJobService:
    """
    DB-backed ingestion job tracker for Module 1.

    Why this exists:
        In production, ingestion should not be a black box.

    It tracks:
        - job_id
        - source type
        - status
        - input/output rows
        - error message
        - metadata
        - timestamps

    This supports:
        POST /ingest
        GET /status/{job_id}
        POST /synthetic/generate
    """

    VALID_STATUSES = {"queued", "running", "completed", "failed", "blocked"}

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def create_job(self, request: IngestionJobCreate) -> Dict[str, Any]:
        if not request.source_type:
            raise ValueError("source_type is required")

        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for ingestion jobs.",
            }

        engine = create_engine(self._connection_url(db_url))
        job_id = f"ingest_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        metadata = self._clean_json(request.metadata or {})

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)

            conn.execute(
                text(
                    """
                    INSERT INTO audience_ingestion_jobs (
                        job_id,
                        run_id,
                        source_type,
                        source_ref,
                        status,
                        input_rows,
                        output_rows,
                        dropped_rows,
                        error_message,
                        actor,
                        metadata,
                        created_at,
                        started_at,
                        completed_at,
                        updated_at
                    )
                    VALUES (
                        :job_id,
                        :run_id,
                        :source_type,
                        :source_ref,
                        :status,
                        :input_rows,
                        :output_rows,
                        :dropped_rows,
                        :error_message,
                        :actor,
                        :metadata,
                        :created_at,
                        :started_at,
                        :completed_at,
                        :updated_at
                    )
                    """
                ),
                {
                    "job_id": job_id,
                    "run_id": request.run_id,
                    "source_type": request.source_type,
                    "source_ref": request.source_ref,
                    "status": "queued",
                    "input_rows": None,
                    "output_rows": None,
                    "dropped_rows": None,
                    "error_message": None,
                    "actor": request.actor,
                    "metadata": json.dumps(metadata, sort_keys=True),
                    "created_at": now,
                    "started_at": None,
                    "completed_at": None,
                    "updated_at": now,
                },
            )

        return {
            "enabled": True,
            "status": "queued",
            "job_id": job_id,
            "run_id": request.run_id,
            "source_type": request.source_type,
            "source_ref": request.source_ref,
            "actor": request.actor,
            "created_at": now,
        }

    def mark_running(self, job_id: str) -> Dict[str, Any]:
        return self.update_job(
            job_id=job_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def mark_completed(
        self,
        *,
        job_id: str,
        input_rows: int,
        output_rows: int,
        dropped_rows: int = 0,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.update_job(
            job_id=job_id,
            status="completed",
            input_rows=input_rows,
            output_rows=output_rows,
            dropped_rows=dropped_rows,
            completed_at=datetime.now(timezone.utc).isoformat(),
            metadata_update=metadata_update,
        )

    def mark_failed(self, *, job_id: str, error_message: str) -> Dict[str, Any]:
        if not error_message:
            raise ValueError("error_message is required")

        return self.update_job(
            job_id=job_id,
            status="failed",
            error_message=error_message,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def mark_blocked(self, *, job_id: str, reason: str) -> Dict[str, Any]:
        if not reason:
            raise ValueError("reason is required")

        return self.update_job(
            job_id=job_id,
            status="blocked",
            error_message=reason,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def update_job(
        self,
        *,
        job_id: str,
        status: str,
        input_rows: Optional[int] = None,
        output_rows: Optional[int] = None,
        dropped_rows: Optional[int] = None,
        error_message: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not job_id:
            raise ValueError("job_id is required")

        if status not in self.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(self.VALID_STATUSES)}")

        for name, value in {
            "input_rows": input_rows,
            "output_rows": output_rows,
            "dropped_rows": dropped_rows,
        }.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")

        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for ingestion jobs.",
            }

        engine = create_engine(self._connection_url(db_url))
        now = datetime.now(timezone.utc).isoformat()

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)

            existing = self._get_job_conn(conn, job_id)
            if not existing:
                return {
                    "enabled": True,
                    "status": "not_found",
                    "job_id": job_id,
                }

            metadata = existing.get("metadata") or {}
            if metadata_update:
                metadata.update(self._clean_json(metadata_update))

            conn.execute(
                text(
                    """
                    UPDATE audience_ingestion_jobs
                    SET status = :status,
                        input_rows = COALESCE(:input_rows, input_rows),
                        output_rows = COALESCE(:output_rows, output_rows),
                        dropped_rows = COALESCE(:dropped_rows, dropped_rows),
                        error_message = COALESCE(:error_message, error_message),
                        started_at = COALESCE(:started_at, started_at),
                        completed_at = COALESCE(:completed_at, completed_at),
                        metadata = :metadata,
                        updated_at = :updated_at
                    WHERE job_id = :job_id
                    """
                ),
                {
                    "job_id": job_id,
                    "status": status,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                    "dropped_rows": dropped_rows,
                    "error_message": error_message,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "metadata": json.dumps(metadata, sort_keys=True),
                    "updated_at": now,
                },
            )

            updated = self._get_job_conn(conn, job_id)

        return {
            "enabled": True,
            "status": status,
            "job": updated,
        }

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if not job_id:
            raise ValueError("job_id is required")

        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for ingestion jobs.",
            }

        engine = create_engine(self._connection_url(db_url))

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            job = self._get_job_conn(conn, job_id)

        if not job:
            return {
                "enabled": True,
                "status": "not_found",
                "job_id": job_id,
            }

        return {
            "enabled": True,
            "status": "ok",
            "job": job,
        }

    def _get_job_conn(self, conn, job_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            text(
                """
                SELECT
                    job_id,
                    run_id,
                    source_type,
                    source_ref,
                    status,
                    input_rows,
                    output_rows,
                    dropped_rows,
                    error_message,
                    actor,
                    metadata,
                    created_at,
                    started_at,
                    completed_at,
                    updated_at
                FROM audience_ingestion_jobs
                WHERE job_id = :job_id
                """
            ),
            {"job_id": job_id},
        ).fetchone()

        if not row:
            return None

        data = dict(row._mapping)
        metadata = data.get("metadata")

        if isinstance(metadata, str):
            try:
                data["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                data["metadata"] = {"raw": metadata}
        elif metadata is None:
            data["metadata"] = {}

        return data

    def _ensure_table(self, conn, dialect_name: str) -> None:
        if dialect_name == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audience_ingestion_jobs (
                        id BIGSERIAL PRIMARY KEY,
                        job_id TEXT UNIQUE NOT NULL,
                        run_id TEXT,
                        source_type TEXT NOT NULL,
                        source_ref TEXT,
                        status TEXT NOT NULL,
                        input_rows INTEGER,
                        output_rows INTEGER,
                        dropped_rows INTEGER,
                        error_message TEXT,
                        actor TEXT NOT NULL DEFAULT 'system',
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        started_at TIMESTAMPTZ,
                        completed_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audience_ingestion_jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT UNIQUE NOT NULL,
                        run_id TEXT,
                        source_type TEXT NOT NULL,
                        source_ref TEXT,
                        status TEXT NOT NULL,
                        input_rows INTEGER,
                        output_rows INTEGER,
                        dropped_rows INTEGER,
                        error_message TEXT,
                        actor TEXT NOT NULL DEFAULT 'system',
                        metadata TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_ingestion_jobs_job
                ON audience_ingestion_jobs(job_id)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_ingestion_jobs_run
                ON audience_ingestion_jobs(run_id)
                """
            )
        )

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

    def _db_url(self) -> Optional[str]:
        return (
            self._explicit_database_url
            or os.getenv("AUDIENCE_INGESTION_DATABASE_URL")
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
