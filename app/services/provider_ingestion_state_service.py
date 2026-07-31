from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.models.provider_ingestion_contracts import ProviderObjectDescriptor


class ProviderIngestionStateService:
    """
    Durable, idempotent state for provider object processing.

    The object fingerprint is unique across all workers. A duplicate S3 event
    therefore observes the existing record instead of executing downstream
    privacy or canonical-write side effects again.
    """

    VALID_STATUSES = {
        "received",
        "validating",
        "dispatching",
        "dispatched",
        "processing",
        "completed",
        "blocked",
        "quarantined",
        "failed",
    }

    ALLOWED_TRANSITIONS = {
        "received": {"validating", "failed"},
        "validating": {
            "dispatching",
            "processing",
            "blocked",
            "quarantined",
            "failed",
        },
        "dispatching": {"dispatched", "failed"},
        "dispatched": {
            "processing",
            "completed",
            "blocked",
            "quarantined",
            "failed",
        },
        "processing": {"completed", "blocked", "quarantined", "failed"},
        "completed": set(),
        "blocked": set(),
        "quarantined": set(),
        "failed": {"validating"},
    }

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def claim_object(
        self,
        descriptor: ProviderObjectDescriptor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )

        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        ingestion_id = f"provider_ingest_{uuid.uuid4().hex}"
        now = self._now()
        payload = self._clean_json(metadata or {})

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)

            values = {
                "ingestion_id": ingestion_id,
                "fingerprint": descriptor.fingerprint,
                "tenant_id": descriptor.tenant_id,
                "provider_id": descriptor.provider_id,
                "dataset_id": descriptor.dataset_id,
                "source_ref": descriptor.source_ref,
                "object_version": descriptor.version_id,
                "checksum_sha256": descriptor.checksum_sha256,
                "status": "received",
                "reason_code": None,
                "attempt_count": 0,
                "privacy_job_id": None,
                "canonical_ref": None,
                "input_rows": None,
                "output_rows": None,
                "metadata": json.dumps(payload, sort_keys=True),
                "created_at": now,
                "updated_at": now,
                "completed_at": None,
            }

            if engine.dialect.name == "postgresql":
                result = conn.execute(
                    text(
                        """
                        INSERT INTO provider_ingestion_objects (
                            ingestion_id,
                            fingerprint,
                            tenant_id,
                            provider_id,
                            dataset_id,
                            source_ref,
                            object_version,
                            checksum_sha256,
                            status,
                            reason_code,
                            attempt_count,
                            privacy_job_id,
                            canonical_ref,
                            input_rows,
                            output_rows,
                            metadata,
                            created_at,
                            updated_at,
                            completed_at
                        )
                        VALUES (
                            :ingestion_id,
                            :fingerprint,
                            :tenant_id,
                            :provider_id,
                            :dataset_id,
                            :source_ref,
                            :object_version,
                            :checksum_sha256,
                            :status,
                            :reason_code,
                            :attempt_count,
                            :privacy_job_id,
                            :canonical_ref,
                            :input_rows,
                            :output_rows,
                            CAST(:metadata AS jsonb),
                            CAST(:created_at AS timestamptz),
                            CAST(:updated_at AS timestamptz),
                            CAST(:completed_at AS timestamptz)
                        )
                        ON CONFLICT (fingerprint) DO NOTHING
                        """
                    ),
                    values,
                )
            else:
                result = conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO provider_ingestion_objects (
                            ingestion_id,
                            fingerprint,
                            tenant_id,
                            provider_id,
                            dataset_id,
                            source_ref,
                            object_version,
                            checksum_sha256,
                            status,
                            reason_code,
                            attempt_count,
                            privacy_job_id,
                            canonical_ref,
                            input_rows,
                            output_rows,
                            metadata,
                            created_at,
                            updated_at,
                            completed_at
                        )
                        VALUES (
                            :ingestion_id,
                            :fingerprint,
                            :tenant_id,
                            :provider_id,
                            :dataset_id,
                            :source_ref,
                            :object_version,
                            :checksum_sha256,
                            :status,
                            :reason_code,
                            :attempt_count,
                            :privacy_job_id,
                            :canonical_ref,
                            :input_rows,
                            :output_rows,
                            :metadata,
                            :created_at,
                            :updated_at,
                            :completed_at
                        )
                        """
                    ),
                    values,
                )

            claimed = bool(result.rowcount)
            record = self._get_by_fingerprint_conn(
                conn,
                descriptor.fingerprint,
            )

        return {
            "claimed": claimed,
            "duplicate": not claimed,
            "record": record,
        }

    def transition(
        self,
        ingestion_id: str,
        *,
        status: str,
        reason_code: Optional[str] = None,
        privacy_job_id: Optional[str] = None,
        canonical_ref: Optional[str] = None,
        input_rows: Optional[int] = None,
        output_rows: Optional[int] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
        increment_attempt: bool = False,
    ) -> Dict[str, Any]:
        if not ingestion_id:
            raise ValueError("ingestion_id is required")
        if status not in self.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(self.VALID_STATUSES)}")
        for field_name, value in {
            "input_rows": input_rows,
            "output_rows": output_rows,
        }.items():
            if value is not None and int(value) < 0:
                raise ValueError(f"{field_name} must be >= 0")

        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )

        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        now = self._now()

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            existing = self._get_by_id_conn(conn, ingestion_id)
            if not existing:
                raise KeyError(f"Provider ingestion record not found: {ingestion_id}")

            current_status = str(existing["status"])
            if (
                status != current_status
                and status not in self.ALLOWED_TRANSITIONS[current_status]
            ):
                raise ValueError(
                    "Invalid provider ingestion transition: "
                    f"{current_status} -> {status}"
                )

            metadata = existing.get("metadata") or {}
            if metadata_update:
                metadata.update(self._clean_json(metadata_update))

            completed_at = (
                now
                if status in {"completed", "blocked", "quarantined", "failed"}
                else (
                    None
                    if current_status == "failed" and status == "validating"
                    else existing.get("completed_at")
                )
            )

            metadata_value = json.dumps(metadata, sort_keys=True)
            if engine.dialect.name == "postgresql":
                metadata_sql = "CAST(:metadata AS jsonb)"
                completed_sql = "CAST(:completed_at AS timestamptz)"
                updated_sql = "CAST(:updated_at AS timestamptz)"
            else:
                metadata_sql = ":metadata"
                completed_sql = ":completed_at"
                updated_sql = ":updated_at"

            conn.execute(
                text(
                    f"""
                    UPDATE provider_ingestion_objects
                    SET status = :status,
                        reason_code = :reason_code,
                        attempt_count = attempt_count + :attempt_increment,
                        privacy_job_id = COALESCE(:privacy_job_id, privacy_job_id),
                        canonical_ref = COALESCE(:canonical_ref, canonical_ref),
                        input_rows = COALESCE(:input_rows, input_rows),
                        output_rows = COALESCE(:output_rows, output_rows),
                        metadata = {metadata_sql},
                        updated_at = {updated_sql},
                        completed_at = {completed_sql}
                    WHERE ingestion_id = :ingestion_id
                    """
                ),
                {
                    "ingestion_id": ingestion_id,
                    "status": status,
                    "reason_code": reason_code,
                    "attempt_increment": 1 if increment_attempt else 0,
                    "privacy_job_id": privacy_job_id,
                    "canonical_ref": canonical_ref,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                    "metadata": metadata_value,
                    "updated_at": now,
                    "completed_at": completed_at,
                },
            )

            updated = self._get_by_id_conn(conn, ingestion_id)

        return updated

    def get(self, ingestion_id: str) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            record = self._get_by_id_conn(conn, ingestion_id)
        if not record:
            raise KeyError(f"Provider ingestion record not found: {ingestion_id}")
        return record

    def get_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            return self._get_by_fingerprint_conn(conn, fingerprint)

    def recover_stale_in_progress(
        self,
        *,
        stale_after_seconds: int,
    ) -> int:
        if stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be >= 1")
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        now = self._now()
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            updated_at_sql = (
                "CAST(:cutoff AS timestamptz)"
                if engine.dialect.name == "postgresql"
                else ":cutoff"
            )
            distributed_exclusion_sql = (
                "COALESCE(metadata->>'execution_mode', '') <> 'distributed'"
                if engine.dialect.name == "postgresql"
                else (
                    "COALESCE(json_extract(metadata, "
                    "'$.execution_mode'), '') <> 'distributed'"
                )
            )
            completed_at_sql = (
                "CAST(:completed_at AS timestamptz)"
                if engine.dialect.name == "postgresql"
                else ":completed_at"
            )
            result = conn.execute(
                text(
                    f"""
                    UPDATE provider_ingestion_objects
                    SET status = 'failed',
                        reason_code = 'worker_execution_stale',
                        updated_at = {completed_at_sql},
                        completed_at = {completed_at_sql}
                    WHERE status IN ('validating', 'dispatching', 'processing')
                      AND updated_at < {updated_at_sql}
                      AND {distributed_exclusion_sql}
                    """
                ),
                {
                    "cutoff": cutoff,
                    "completed_at": now,
                },
            )
        return max(0, int(result.rowcount or 0))

    def list_recent(
        self,
        *,
        tenant_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> list[Dict[str, Any]]:
        if not 1 <= int(limit) <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if status is not None and status not in self.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(self.VALID_STATUSES)}")

        clauses = ["1 = 1"]
        params: Dict[str, Any] = {"limit": int(limit)}
        for field_name, value in {
            "tenant_id": tenant_id,
            "provider_id": provider_id,
            "dataset_id": dataset_id,
            "status": status,
        }.items():
            if value is not None:
                clauses.append(f"{field_name} = :{field_name}")
                params[field_name] = str(value)

        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            rows = conn.execute(
                text(
                    f"""
                    SELECT *
                    FROM provider_ingestion_objects
                    WHERE {' AND '.join(clauses)}
                    ORDER BY updated_at DESC
                    LIMIT :limit
                    """
                ),
                params,
            ).fetchall()
        return [
            record
            for record in (self._row_to_record(row) for row in rows)
            if record is not None
        ]

    def status_counts(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, int]:
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            tenant_clause = (
                "WHERE tenant_id = :tenant_id" if tenant_id else ""
            )
            rows = conn.execute(
                text(
                    f"""
                    SELECT status, COUNT(*) AS record_count
                    FROM provider_ingestion_objects
                    {tenant_clause}
                    GROUP BY status
                    """
                ),
                {"tenant_id": tenant_id},
            ).fetchall()
        counts = {status: 0 for status in sorted(self.VALID_STATUSES)}
        for row in rows:
            counts[str(row._mapping["status"])] = int(
                row._mapping["record_count"]
            )
        return counts

    def prepare_controlled_replay(
        self,
        ingestion_id: str,
        *,
        requested_by: str,
        reason: str,
        approval_reference: str,
    ) -> Dict[str, Any]:
        for field_name, value in {
            "requested_by": requested_by,
            "reason": reason,
            "approval_reference": approval_reference,
        }.items():
            if not str(value or "").strip():
                raise ValueError(f"{field_name} is required")

        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        replay_id = f"provider_replay_{uuid.uuid4().hex}"
        now = self._now()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            existing = self._get_by_id_conn(conn, ingestion_id)
            if not existing:
                raise KeyError(
                    f"Provider ingestion record not found: {ingestion_id}"
                )
            current_status = str(existing["status"])
            if current_status not in {
                "failed",
                "blocked",
                "quarantined",
                "completed",
            }:
                raise ValueError(
                    "Only terminal provider ingestion records can be replayed."
                )

            if current_status in {"failed", "blocked", "quarantined"}:
                conn.execute(
                    text(
                        """
                        UPDATE provider_ingestion_objects
                        SET status = 'failed',
                            reason_code = 'controlled_replay_requested',
                            updated_at = :updated_at,
                            completed_at = :completed_at
                        WHERE ingestion_id = :ingestion_id
                        """
                    ),
                    {
                        "ingestion_id": ingestion_id,
                        "updated_at": now,
                        "completed_at": now,
                    },
                )

            conn.execute(
                text(
                    """
                    INSERT INTO provider_ingestion_replays (
                        replay_id,
                        ingestion_id,
                        requested_by,
                        reason,
                        approval_reference,
                        replay_mode,
                        status,
                        queue_message_id,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :replay_id,
                        :ingestion_id,
                        :requested_by,
                        :reason,
                        :approval_reference,
                        :replay_mode,
                        'pending',
                        NULL,
                        :created_at,
                        :updated_at
                    )
                    """
                ),
                {
                    "replay_id": replay_id,
                    "ingestion_id": ingestion_id,
                    "requested_by": str(requested_by).strip(),
                    "reason": str(reason).strip(),
                    "approval_reference": str(approval_reference).strip(),
                    "replay_mode": (
                        "idempotency_verification"
                        if current_status == "completed"
                        else "terminal_failure_retry"
                    ),
                    "created_at": now,
                    "updated_at": now,
                },
            )
            record = self._get_by_id_conn(conn, ingestion_id)

        return {
            "replay_id": replay_id,
            "status": "pending",
            "replay_mode": (
                "idempotency_verification"
                if current_status == "completed"
                else "terminal_failure_retry"
            ),
            "ingestion": record,
        }

    def mark_replay_enqueued(
        self,
        replay_id: str,
        *,
        queue_message_id: str,
    ) -> Dict[str, Any]:
        if not str(replay_id or "").strip():
            raise ValueError("replay_id is required")
        if not str(queue_message_id or "").strip():
            raise ValueError("queue_message_id is required")
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Provider ingestion state requires a configured PostgreSQL database."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        now = self._now()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            result = conn.execute(
                text(
                    """
                    UPDATE provider_ingestion_replays
                    SET status = 'queued',
                        queue_message_id = :queue_message_id,
                        updated_at = :updated_at
                    WHERE replay_id = :replay_id
                      AND status = 'pending'
                    """
                ),
                {
                    "replay_id": replay_id,
                    "queue_message_id": queue_message_id,
                    "updated_at": now,
                },
            )
            if not result.rowcount:
                raise KeyError("Pending provider replay request was not found.")
            row = conn.execute(
                text(
                    """
                    SELECT *
                    FROM provider_ingestion_replays
                    WHERE replay_id = :replay_id
                    """
                ),
                {"replay_id": replay_id},
            ).fetchone()
        return dict(row._mapping)

    def _ensure_table(self, conn, dialect_name: str) -> None:
        # Production identities intentionally have no schema CREATE privilege.
        if dialect_name == "postgresql":
            return
        if dialect_name == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_ingestion_objects (
                        ingestion_id TEXT PRIMARY KEY,
                        fingerprint TEXT UNIQUE NOT NULL,
                        tenant_id TEXT NOT NULL,
                        provider_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        source_ref TEXT NOT NULL,
                        object_version TEXT,
                        checksum_sha256 TEXT,
                        status TEXT NOT NULL,
                        reason_code TEXT,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        privacy_job_id TEXT,
                        canonical_ref TEXT,
                        input_rows BIGINT,
                        output_rows BIGINT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        completed_at TIMESTAMPTZ
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_ingestion_objects (
                        ingestion_id TEXT PRIMARY KEY,
                        fingerprint TEXT UNIQUE NOT NULL,
                        tenant_id TEXT NOT NULL,
                        provider_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        source_ref TEXT NOT NULL,
                        object_version TEXT,
                        checksum_sha256 TEXT,
                        status TEXT NOT NULL,
                        reason_code TEXT,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        privacy_job_id TEXT,
                        canonical_ref TEXT,
                        input_rows INTEGER,
                        output_rows INTEGER,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT
                    )
                    """
                )
            )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_ingestion_tenant_dataset
                ON provider_ingestion_objects (
                    tenant_id,
                    provider_id,
                    dataset_id,
                    updated_at
                )
                """
            )
        )
        if dialect_name == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_ingestion_replays (
                        replay_id TEXT PRIMARY KEY,
                        ingestion_id TEXT NOT NULL REFERENCES
                            provider_ingestion_objects(ingestion_id),
                        requested_by TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        approval_reference TEXT NOT NULL,
                        replay_mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        queue_message_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_ingestion_replays (
                        replay_id TEXT PRIMARY KEY,
                        ingestion_id TEXT NOT NULL,
                        requested_by TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        approval_reference TEXT NOT NULL,
                        replay_mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        queue_message_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (ingestion_id)
                            REFERENCES provider_ingestion_objects(ingestion_id)
                    )
                    """
                )
            )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_ingestion_status
                ON provider_ingestion_objects (status, updated_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_provider_ingestion_replays_ingestion
                ON provider_ingestion_replays (ingestion_id, created_at)
                """
            )
        )

    def _get_by_id_conn(
        self,
        conn,
        ingestion_id: str,
    ) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            text(
                """
                SELECT *
                FROM provider_ingestion_objects
                WHERE ingestion_id = :ingestion_id
                """
            ),
            {"ingestion_id": ingestion_id},
        ).fetchone()
        return self._row_to_record(row)

    def _get_by_fingerprint_conn(
        self,
        conn,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            text(
                """
                SELECT *
                FROM provider_ingestion_objects
                WHERE fingerprint = :fingerprint
                """
            ),
            {"fingerprint": fingerprint},
        ).fetchone()
        return self._row_to_record(row)

    def _row_to_record(self, row) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        record = dict(row._mapping)
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            try:
                record["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                record["metadata"] = {}
        elif metadata is None:
            record["metadata"] = {}
        return record

    def _db_url(self) -> Optional[str]:
        return (
            self._explicit_database_url
            or os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        )

    def _connection_url(self, db_url: str) -> str:
        if db_url.startswith("postgres://"):
            return "postgresql://" + db_url[len("postgres://") :]
        return db_url

    def _clean_json(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {str(key): self._clean_json(item) for key, item in value.items()}
            if isinstance(value, list):
                return [self._clean_json(item) for item in value]
            return str(value)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
