from __future__ import annotations

import json
import math
import os
import uuid
from importlib.util import find_spec
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text


class AudienceJobStore:
    """
    Audience Intelligence asynchronous job persistence.

    Production:
        AUDIENCE_JOB_STORE_BACKEND=postgres

    Development/tests:
        AUDIENCE_JOB_STORE_BACKEND=local
    """

    POSTGRES_BACKENDS = {
        "postgres",
        "postgresql",
        "postgres_array",
        "pg",
    }

    def __init__(
        self,
        root_dir: str | Path = "data/audience_jobs",
        *,
        backend: Optional[str] = None,
        db_url: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.backend = (
            backend
            or os.getenv("AUDIENCE_JOB_STORE_BACKEND", "local")
        ).strip().lower()
        self.db_url = db_url
        self._engine_instance = None

        if self.backend in self.POSTGRES_BACKENDS:
            if not self._db_url():
                raise RuntimeError(
                    "AUDIENCE_JOB_STORE_BACKEND is Postgres, but no "
                    "ECHO_DATABASE_URL or DATABASE_URL is configured."
                )
            return

        if self.backend != "local":
            raise ValueError(
                f"Unsupported audience job-store backend: {self.backend}"
            )

        production_mode = (
            os.getenv("PRODUCTION_MODE", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        local_allowed = (
            os.getenv("ALLOW_LOCAL_FILE_STORAGE", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if production_mode and not local_allowed:
            raise RuntimeError(
                "Local AudienceJobStore is blocked in production. "
                "Set AUDIENCE_JOB_STORE_BACKEND=postgres."
            )

        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = (
            "job_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )

        record = {
            "job_id": job_id,
            "status": "queued",
            "created_at": self._now(),
            "updated_at": self._now(),
            "payload": payload,
            "progress": {
                "stage": "queued",
                "message": "Audience Intelligence job queued.",
            },
            "result": None,
            "error": None,
        }

        self.save(record)
        return record

    def get(self, job_id: str) -> Dict[str, Any]:
        if self._uses_postgres():
            return self._get_postgres(job_id)

        path = self._path(job_id)

        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")

        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, record: Dict[str, Any]) -> None:
        saved = self._clean_json(dict(record))
        saved["updated_at"] = self._now()

        if self._uses_postgres():
            self._save_postgres(saved)
        else:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._path(saved["job_id"]).write_text(
                json.dumps(saved, indent=2, allow_nan=False),
                encoding="utf-8",
            )

        record["updated_at"] = saved["updated_at"]

    def update_status(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        message: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self.get(job_id)
        record["status"] = status
        record["progress"] = {
            "stage": stage,
            "message": message,
        }

        if result is not None:
            record["result"] = result

        if error is not None:
            record["error"] = error

        self.save(record)
        return record

    def _get_postgres(self, job_id: str) -> Dict[str, Any]:
        engine = self._engine()

        with engine.begin() as conn:
            self._ensure_table(conn)

            row = conn.execute(
                text(
                    """
                    SELECT job_id, status, created_at, updated_at,
                           payload, progress, result, error
                    FROM audience_jobs
                    WHERE job_id = :job_id
                    """
                ),
                {"job_id": job_id},
            ).fetchone()

        if not row:
            raise FileNotFoundError(f"Job not found: {job_id}")

        values = dict(row._mapping)

        return {
            "job_id": values["job_id"],
            "status": values["status"],
            "created_at": self._iso(values.get("created_at")),
            "updated_at": self._iso(values.get("updated_at")),
            "payload": self._json_object(values.get("payload"), {}),
            "progress": self._json_object(values.get("progress"), {}),
            "result": self._json_object(values.get("result"), None),
            "error": values.get("error"),
        }

    def _save_postgres(self, record: Dict[str, Any]) -> None:
        engine = self._engine()

        result_json = None
        if record.get("result") is not None:
            result_json = self._json_text(record.get("result"))

        with engine.begin() as conn:
            self._ensure_table(conn)

            conn.execute(
                text(
                    """
                    INSERT INTO audience_jobs (
                        job_id,
                        status,
                        created_at,
                        updated_at,
                        payload,
                        progress,
                        result,
                        error
                    )
                    VALUES (
                        :job_id,
                        :status,
                        CAST(:created_at AS timestamptz),
                        CAST(:updated_at AS timestamptz),
                        CAST(:payload AS jsonb),
                        CAST(:progress AS jsonb),
                        CAST(:result AS jsonb),
                        :error
                    )
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload,
                        progress = EXCLUDED.progress,
                        result = EXCLUDED.result,
                        error = EXCLUDED.error;
                    """
                ),
                {
                    "job_id": record["job_id"],
                    "status": record["status"],
                    "created_at": record.get("created_at") or self._now(),
                    "updated_at": record.get("updated_at") or self._now(),
                    "payload": self._json_text(record.get("payload") or {}),
                    "progress": self._json_text(record.get("progress") or {}),
                    "result": result_json,
                    "error": record.get("error"),
                },
            )

    def _ensure_table(self, conn) -> None:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
                    result JSONB,
                    error TEXT
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_jobs_status
                ON audience_jobs(status, updated_at DESC);
                """
            )
        )

    def _engine(self):
        if self._engine_instance is None:
            self._engine_instance = create_engine(
                self._connection_url(self._db_url()),
                pool_pre_ping=True,
            )

        return self._engine_instance

    def _db_url(self) -> Optional[str]:
        return (
            self.db_url
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
        )

    def _connection_url(self, db_url: str) -> str:
        """
        Convert the configured PostgreSQL URL to an installed synchronous
        SQLAlchemy driver.

        Prefer Psycopg 3 when present, otherwise use psycopg2. The asynchronous
        asyncpg driver cannot be used by this synchronous job store.
        """
        normalized = db_url

        for prefix in (
            "postgresql+asyncpg://",
            "postgresql+psycopg://",
            "postgresql+psycopg2://",
        ):
            if normalized.startswith(prefix):
                normalized = normalized.replace(
                    prefix,
                    "postgresql://",
                    1,
                )
                break

        if normalized.startswith("postgres://"):
            normalized = normalized.replace(
                "postgres://",
                "postgresql://",
                1,
            )

        if find_spec("psycopg") is not None:
            driver = "psycopg"
        elif find_spec("psycopg2") is not None:
            driver = "psycopg2"
        else:
            raise RuntimeError(
                "No supported synchronous PostgreSQL driver is installed. "
                "Install psycopg[binary] or psycopg2-binary."
            )

        return normalized.replace(
            "postgresql://",
            f"postgresql+{driver}://",
            1,
        )

    def _uses_postgres(self) -> bool:
        return self.backend in self.POSTGRES_BACKENDS

    def _path(self, job_id: str) -> Path:
        safe_job_id = "".join(
            ch for ch in job_id
            if ch.isalnum() or ch in {"_", "-"}
        )
        return self.root_dir / f"{safe_job_id}.json"

    def _json_text(self, value: Any) -> str:
        return json.dumps(
            self._clean_json(value),
            separators=(",", ":"),
            allow_nan=False,
        )

    def _json_object(self, value: Any, default: Any) -> Any:
        if value is None:
            return default

        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default

        return value

    def _clean_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._clean_json(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._clean_json(item) for item in value]

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if hasattr(value, "item"):
            try:
                return self._clean_json(value.item())
            except Exception:
                pass

        return value

    def _iso(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
