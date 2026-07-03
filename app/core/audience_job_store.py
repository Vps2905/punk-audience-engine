from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class AudienceJobStore:
    """
    Local file-backed job store for Audience Intelligence jobs.

    Production note:
    This is intentionally simple and safe for local/internal rollout.
    Later we can replace this with Postgres/Redis/Celery without changing API shape.
    """

    def __init__(self, root_dir: str | Path = "data/audience_jobs"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

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
        path = self._path(job_id)

        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")

        return json.loads(path.read_text())

    def save(self, record: Dict[str, Any]) -> None:
        record["updated_at"] = self._now()
        self._path(record["job_id"]).write_text(json.dumps(record, indent=2, allow_nan=False))

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

    def _path(self, job_id: str) -> Path:
        safe_job_id = "".join(ch for ch in job_id if ch.isalnum() or ch in {"_", "-"})
        return self.root_dir / f"{safe_job_id}.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
