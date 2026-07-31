from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.models.provider_privacy_window_contracts import (
    ProviderDataRightsRequest,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


class ProviderDataRightsService:
    """
    Fail-closed deletion and opt-out propagation.

    This boundary accepts only an HMAC-derived token digest. It never accepts
    or returns a raw MAID. Because released aggregates cannot safely subtract
    one subject, affected windows are revoked and must be rebuilt from an
    authorized retained source with the suppression set applied.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        *,
        window_service: Optional[ProviderPrivacyWindowService] = None,
    ) -> None:
        self._explicit_database_url = database_url
        self._windows = window_service or ProviderPrivacyWindowService(
            database_url=database_url
        )

    def submit(
        self,
        request: ProviderDataRightsRequest,
    ) -> Dict[str, Any]:
        engine = self._engine()
        now = self._now()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            self._set_tenant(
                conn, engine.dialect.name, request.tenant_id
            )
            existing = conn.execute(
                text(
                    """
                    SELECT * FROM provider_data_rights_requests
                    WHERE request_id = :request_id
                    """
                ),
                {"request_id": request.request_id},
            ).mappings().first()
            if existing:
                self._validate_replay(dict(existing), request)
                return self._safe_result(dict(existing), replayed=True)
            conn.execute(
                text(
                    """
                    INSERT INTO provider_data_rights_requests (
                        request_id, tenant_id, provider_id, dataset_id,
                        request_type, subject_token_sha256, requested_at,
                        event_time_start, event_time_end, source_request_ref,
                        status, created_at, updated_at
                    ) VALUES (
                        :request_id, :tenant_id, :provider_id, :dataset_id,
                        :request_type, :subject_token_sha256, :requested_at,
                        :event_time_start, :event_time_end, :source_request_ref,
                        'received', :created_at, :updated_at
                    )
                    """
                ),
                {
                    **request.__dict__,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            row = conn.execute(
                text(
                    """
                    SELECT * FROM provider_data_rights_requests
                    WHERE request_id = :request_id
                    """
                ),
                {"request_id": request.request_id},
            ).mappings().one()
            return self._safe_result(dict(row), replayed=False)

    def apply(
        self,
        request_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not str(tenant_id or "").strip():
            raise ValueError("tenant_id is required")
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, str(tenant_id))
            row = conn.execute(
                text(
                    """
                    SELECT * FROM provider_data_rights_requests
                    WHERE request_id = :request_id
                      AND tenant_id = :tenant_id
                    """
                ),
                {"request_id": request_id, "tenant_id": tenant_id},
            ).mappings().first()
        if not row:
            raise KeyError("Data-rights request was not found")
        record = dict(row)
        if record["status"] == "applied":
            return self._safe_result(record, replayed=True)
        if record["status"] != "received":
            raise ValueError("Data-rights request is not applicable")

        revocation = self._windows.revoke_scope(
            tenant_id=record["tenant_id"],
            provider_id=record["provider_id"],
            dataset_id=record["dataset_id"],
            reason_code=f"data_rights_{record['request_type']}",
            event_time_start=record.get("event_time_start"),
            event_time_end=record.get("event_time_end"),
        )
        now = self._now()
        with engine.begin() as conn:
            self._set_tenant(conn, engine.dialect.name, str(tenant_id))
            conn.execute(
                text(
                    """
                    UPDATE provider_data_rights_requests
                    SET status = 'applied',
                        affected_window_count = :affected_window_count,
                        rebuild_required = 1,
                        applied_at = :applied_at,
                        updated_at = :updated_at
                    WHERE request_id = :request_id
                      AND status = 'received'
                      AND tenant_id = :tenant_id
                    """
                ),
                {
                    "request_id": request_id,
                    "tenant_id": tenant_id,
                    "affected_window_count": revocation[
                        "revoked_window_count"
                    ],
                    "applied_at": now,
                    "updated_at": now,
                },
            )
            updated = conn.execute(
                text(
                    """
                    SELECT * FROM provider_data_rights_requests
                    WHERE request_id = :request_id
                    """
                ),
                {"request_id": request_id},
            ).mappings().one()
        result = self._safe_result(dict(updated), replayed=False)
        result["activation_blocked"] = True
        result["rebuild_required"] = True
        return result

    def active_token_digests(
        self,
        *,
        tenant_id: str,
        provider_id: str,
        dataset_id: str,
        limit: int = 100_000,
    ) -> tuple[str, ...]:
        """
        Internal privacy-boundary lookup for bounded in-process ingestion.

        Distributed ingestion uses an encrypted suppression snapshot rather
        than loading an unbounded suppression set into the API worker.
        """
        if not 1 <= int(limit) <= 100_000:
            raise ValueError("limit must be between 1 and 100000")
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, tenant_id)
            rows = conn.execute(
                text(
                    """
                    SELECT subject_token_sha256
                    FROM provider_data_rights_requests
                    WHERE tenant_id = :tenant_id
                      AND provider_id = :provider_id
                      AND dataset_id = :dataset_id
                      AND status = 'applied'
                    ORDER BY requested_at
                    LIMIT :limit
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "provider_id": provider_id,
                    "dataset_id": dataset_id,
                    "limit": int(limit),
                },
            ).fetchall()
        return tuple(str(row._mapping["subject_token_sha256"]) for row in rows)

    def _validate_replay(
        self,
        existing: Dict[str, Any],
        request: ProviderDataRightsRequest,
    ) -> None:
        for field_name, expected in request.__dict__.items():
            actual = existing.get(field_name)
            if actual is None and expected is None:
                continue
            if str(actual) != str(expected):
                raise ValueError(
                    "Idempotent data-rights request parameters changed"
                )

    def _safe_result(
        self,
        record: Dict[str, Any],
        *,
        replayed: bool,
    ) -> Dict[str, Any]:
        return {
            "request_id": record["request_id"],
            "tenant_id": record["tenant_id"],
            "provider_id": record["provider_id"],
            "dataset_id": record["dataset_id"],
            "request_type": record["request_type"],
            "status": record["status"],
            "affected_window_count": int(
                record.get("affected_window_count") or 0
            ),
            "rebuild_required": bool(record.get("rebuild_required") or 0),
            "subject_token_exposed": False,
            "raw_identifier_accepted": False,
            "replayed": replayed,
        }

    def _ensure_table(self, conn: Any, dialect: str) -> None:
        # Runtime identities must not create or alter production schema.
        if dialect == "postgresql":
            return
        timestamp = "TIMESTAMPTZ" if dialect == "postgresql" else "TEXT"
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS provider_data_rights_requests (
                    request_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    subject_token_sha256 TEXT NOT NULL,
                    requested_at {timestamp} NOT NULL,
                    event_time_start {timestamp},
                    event_time_end {timestamp},
                    source_request_ref TEXT,
                    status TEXT NOT NULL,
                    affected_window_count INTEGER NOT NULL DEFAULT 0,
                    rebuild_required INTEGER NOT NULL DEFAULT 0,
                    created_at {timestamp} NOT NULL,
                    updated_at {timestamp} NOT NULL,
                    applied_at {timestamp}
                )
                """
            )
        )

    def _set_tenant(self, conn: Any, dialect: str, tenant_id: str) -> None:
        normalized = str(tenant_id or "").strip().lower()
        if not normalized:
            raise ValueError("tenant_id is required")
        if dialect == "postgresql":
            conn.execute(
                text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                {"tenant_id": normalized},
            )

    def _engine(self) -> Any:
        database_url = (
            self._explicit_database_url
            or os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        )
        if not database_url:
            raise RuntimeError("Data-rights processing requires PostgreSQL.")
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
