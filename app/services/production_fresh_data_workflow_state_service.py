from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_fresh_data_workflow_contracts import (
    ProductionFreshDataWorkflowRequest,
)


_TERMINAL_STATUSES = {
    "awaiting_review",
    "blocked",
    "quarantined",
    "failed",
}

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "claimed": {"validating_ingestion", "blocked", "failed"},
    "validating_ingestion": {
        "reading_canonical",
        "blocked",
        "quarantined",
        "failed",
    },
    "reading_canonical": {
        "building_features",
        "quarantined",
        "failed",
    },
    "building_features": {
        "generating_candidates",
        "blocked",
        "quarantined",
        "failed",
    },
    "generating_candidates": {
        "analyzing_overlap",
        "blocked",
        "failed",
    },
    "analyzing_overlap": {
        "awaiting_review",
        "blocked",
        "failed",
    },
    "awaiting_review": set(),
    "blocked": set(),
    "quarantined": set(),
    "failed": set(),
}


class ProductionFreshDataWorkflowStateService:
    """Durable tenant-isolated workflow state with worker leasing and audit events."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        engine: Engine | None = None,
        now_fn: Any = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def claim(
        self,
        request: ProductionFreshDataWorkflowRequest,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        clean_worker = normalize_taxonomy_value(worker_id)
        if not clean_worker:
            raise ValueError("worker_id is required.")
        if not 30 <= int(lease_seconds) <= 3600:
            raise ValueError("lease_seconds must be between 30 and 3600.")

        now = self._now_fn()
        lease_expires_at = now + timedelta(seconds=int(lease_seconds))
        engine = self._engine()
        with engine.begin() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, request.source.tenant_id)
            inserted = connection.execute(
                text(
                    """
                    INSERT INTO audience_fresh_data_workflows (
                        tenant_id,
                        workflow_id,
                        request_fingerprint,
                        ingestion_id,
                        provider_id,
                        dataset_id,
                        canonical_source_fingerprint,
                        feature_build_id,
                        status,
                        reason_code,
                        attempt_count,
                        lease_owner,
                        lease_expires_at,
                        request_manifest,
                        approval_required,
                        activation_requested,
                        export_requested,
                        lookalike_generation_requested,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :tenant_id,
                        :workflow_id,
                        :request_fingerprint,
                        :ingestion_id,
                        :provider_id,
                        :dataset_id,
                        :canonical_source_fingerprint,
                        :feature_build_id,
                        'claimed',
                        NULL,
                        1,
                        :lease_owner,
                        :lease_expires_at,
                        CAST(:request_manifest AS JSONB),
                        TRUE,
                        FALSE,
                        FALSE,
                        FALSE,
                        :now,
                        :now
                    )
                    ON CONFLICT (tenant_id, request_fingerprint) DO NOTHING
                    RETURNING workflow_id
                    """
                ),
                {
                    "tenant_id": request.source.tenant_id,
                    "workflow_id": request.workflow_id,
                    "request_fingerprint": request.request_fingerprint,
                    "ingestion_id": request.ingestion_id,
                    "provider_id": request.source.provider_id,
                    "dataset_id": request.source.dataset_id,
                    "canonical_source_fingerprint": (
                        request.source.source_fingerprint
                    ),
                    "feature_build_id": request.feature_build_request.build_id,
                    "lease_owner": clean_worker,
                    "lease_expires_at": lease_expires_at,
                    "request_manifest": json.dumps(
                        request.to_safe_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "now": now,
                },
            ).scalar_one_or_none()

            record = connection.execute(
                text(
                    """
                    SELECT *
                    FROM audience_fresh_data_workflows
                    WHERE tenant_id = :tenant_id
                      AND request_fingerprint = :request_fingerprint
                    FOR UPDATE
                    """
                ),
                {
                    "tenant_id": request.source.tenant_id,
                    "request_fingerprint": request.request_fingerprint,
                },
            ).mappings().one()
            safe_record = self._record(record)
            self._verify_identity(safe_record, request)

            if inserted is not None:
                self._append_event(
                    connection,
                    tenant_id=request.source.tenant_id,
                    workflow_id=request.workflow_id,
                    from_status=None,
                    to_status="claimed",
                    actor=clean_worker,
                    reason_code=None,
                    details={"lease_seconds": int(lease_seconds)},
                    occurred_at=now,
                )
                return {
                    "acquired": True,
                    "duplicate": False,
                    "busy": False,
                    "record": safe_record,
                }

            status = str(safe_record["status"])
            if status in _TERMINAL_STATUSES:
                return {
                    "acquired": False,
                    "duplicate": True,
                    "busy": False,
                    "record": safe_record,
                }

            existing_owner = str(safe_record.get("lease_owner") or "")
            existing_expiry = safe_record.get("lease_expires_at")
            lease_active = bool(
                existing_expiry
                and existing_expiry > now
                and existing_owner
                and existing_owner != clean_worker
            )
            if lease_active:
                return {
                    "acquired": False,
                    "duplicate": True,
                    "busy": True,
                    "record": safe_record,
                }

            record = connection.execute(
                text(
                    """
                    UPDATE audience_fresh_data_workflows
                    SET
                        lease_owner = :lease_owner,
                        lease_expires_at = :lease_expires_at,
                        attempt_count = attempt_count + 1,
                        updated_at = :now
                    WHERE tenant_id = :tenant_id
                      AND workflow_id = :workflow_id
                    RETURNING *
                    """
                ),
                {
                    "lease_owner": clean_worker,
                    "lease_expires_at": lease_expires_at,
                    "now": now,
                    "tenant_id": request.source.tenant_id,
                    "workflow_id": request.workflow_id,
                },
            ).mappings().one()
            self._append_event(
                connection,
                tenant_id=request.source.tenant_id,
                workflow_id=request.workflow_id,
                from_status=status,
                to_status=status,
                actor=clean_worker,
                reason_code="lease_reacquired",
                details={"lease_seconds": int(lease_seconds)},
                occurred_at=now,
            )
            return {
                "acquired": True,
                "duplicate": True,
                "busy": False,
                "record": self._record(record),
            }

    def transition(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
        worker_id: str,
        status: str,
        reason_code: str | None = None,
        feature_receipt: Mapping[str, Any] | None = None,
        candidate_report: Mapping[str, Any] | None = None,
        overlap_report: Mapping[str, Any] | None = None,
        result_receipt: Mapping[str, Any] | None = None,
        event_details: Mapping[str, Any] | None = None,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        clean_tenant = normalize_taxonomy_value(tenant_id)
        clean_worker = normalize_taxonomy_value(worker_id)
        clean_status = normalize_taxonomy_value(status)
        if clean_status not in _ALLOWED_TRANSITIONS:
            raise ValueError("Unsupported fresh-data workflow status.")
        if not clean_tenant or not clean_worker:
            raise ValueError("tenant_id and worker_id are required.")
        if not 30 <= int(lease_seconds) <= 3600:
            raise ValueError("lease_seconds must be between 30 and 3600.")

        now = self._now_fn()
        engine = self._engine()
        with engine.begin() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, clean_tenant)
            current = connection.execute(
                text(
                    """
                    SELECT *
                    FROM audience_fresh_data_workflows
                    WHERE tenant_id = :tenant_id
                      AND workflow_id = :workflow_id
                    FOR UPDATE
                    """
                ),
                {"tenant_id": clean_tenant, "workflow_id": workflow_id},
            ).mappings().one_or_none()
            if current is None:
                raise FileNotFoundError("Fresh-data workflow was not found.")
            current_record = self._record(current)
            current_status = str(current_record["status"])
            if clean_status not in _ALLOWED_TRANSITIONS[current_status]:
                raise RuntimeError(
                    "Invalid fresh-data workflow transition: "
                    f"{current_status} -> {clean_status}."
                )
            if str(current_record.get("lease_owner") or "") != clean_worker:
                raise RuntimeError("Fresh-data workflow lease is not owned by worker.")
            lease_expiry = current_record.get("lease_expires_at")
            if lease_expiry is None or lease_expiry <= now:
                raise RuntimeError("Fresh-data workflow lease has expired.")

            terminal = clean_status in _TERMINAL_STATUSES
            if clean_status == "awaiting_review" and not result_receipt:
                raise ValueError(
                    "awaiting_review requires a verified result receipt."
                )

            updated = connection.execute(
                text(
                    """
                    UPDATE audience_fresh_data_workflows
                    SET
                        status = :status,
                        reason_code = :reason_code,
                        feature_receipt = COALESCE(
                            CAST(:feature_receipt AS JSONB),
                            feature_receipt
                        ),
                        candidate_report = COALESCE(
                            CAST(:candidate_report AS JSONB),
                            candidate_report
                        ),
                        overlap_report = COALESCE(
                            CAST(:overlap_report AS JSONB),
                            overlap_report
                        ),
                        result_receipt = COALESCE(
                            CAST(:result_receipt AS JSONB),
                            result_receipt
                        ),
                        lease_owner = CASE WHEN :terminal THEN NULL ELSE lease_owner END,
                        lease_expires_at = CASE
                            WHEN :terminal THEN NULL
                            ELSE :lease_expires_at
                        END,
                        updated_at = :now,
                        completed_at = CASE WHEN :terminal THEN :now ELSE NULL END
                    WHERE tenant_id = :tenant_id
                      AND workflow_id = :workflow_id
                    RETURNING *
                    """
                ),
                {
                    "status": clean_status,
                    "reason_code": (
                        normalize_taxonomy_value(reason_code)
                        if reason_code
                        else None
                    ),
                    "feature_receipt": self._json(feature_receipt),
                    "candidate_report": self._json(candidate_report),
                    "overlap_report": self._json(overlap_report),
                    "result_receipt": self._json(result_receipt),
                    "terminal": terminal,
                    "lease_expires_at": now
                    + timedelta(seconds=int(lease_seconds)),
                    "now": now,
                    "tenant_id": clean_tenant,
                    "workflow_id": workflow_id,
                },
            ).mappings().one()
            self._append_event(
                connection,
                tenant_id=clean_tenant,
                workflow_id=workflow_id,
                from_status=current_status,
                to_status=clean_status,
                actor=clean_worker,
                reason_code=reason_code,
                details=dict(event_details or {}),
                occurred_at=now,
            )
            return self._record(updated)

    def get(self, *, tenant_id: str, workflow_id: str) -> dict[str, Any]:
        clean_tenant = normalize_taxonomy_value(tenant_id)
        engine = self._engine()
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                self._assert_schema_ready(connection)
                self._set_tenant_context(connection, clean_tenant)
                row = connection.execute(
                    text(
                        """
                        SELECT *
                        FROM audience_fresh_data_workflows
                        WHERE tenant_id = :tenant_id
                          AND workflow_id = :workflow_id
                        """
                    ),
                    {"tenant_id": clean_tenant, "workflow_id": workflow_id},
                ).mappings().one_or_none()
            finally:
                transaction.rollback()
        if row is None:
            raise FileNotFoundError("Fresh-data workflow was not found.")
        return self._record(row)

    def _append_event(
        self,
        connection: Any,
        *,
        tenant_id: str,
        workflow_id: str,
        from_status: str | None,
        to_status: str,
        actor: str,
        reason_code: str | None,
        details: Mapping[str, Any],
        occurred_at: datetime,
    ) -> None:
        connection.execute(
            text(
                """
                INSERT INTO audience_fresh_data_workflow_events (
                    tenant_id,
                    workflow_id,
                    from_status,
                    to_status,
                    actor,
                    reason_code,
                    details,
                    occurred_at
                ) VALUES (
                    :tenant_id,
                    :workflow_id,
                    :from_status,
                    :to_status,
                    :actor,
                    :reason_code,
                    CAST(:details AS JSONB),
                    :occurred_at
                )
                """
            ),
            {
                "tenant_id": tenant_id,
                "workflow_id": workflow_id,
                "from_status": from_status,
                "to_status": to_status,
                "actor": actor,
                "reason_code": (
                    normalize_taxonomy_value(reason_code)
                    if reason_code
                    else None
                ),
                "details": json.dumps(
                    dict(details),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "occurred_at": occurred_at,
            },
        )

    def _verify_identity(
        self,
        record: Mapping[str, Any],
        request: ProductionFreshDataWorkflowRequest,
    ) -> None:
        expected = {
            "tenant_id": request.source.tenant_id,
            "workflow_id": request.workflow_id,
            "request_fingerprint": request.request_fingerprint,
            "ingestion_id": request.ingestion_id,
            "provider_id": request.source.provider_id,
            "dataset_id": request.source.dataset_id,
            "canonical_source_fingerprint": request.source.source_fingerprint,
            "feature_build_id": request.feature_build_request.build_id,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "Existing fresh-data workflow identity conflicts with request."
            )

    def _record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(row)
        for key in ("attempt_count",):
            if record.get(key) is not None:
                record[key] = int(record[key])
        for key in (
            "request_manifest",
            "feature_receipt",
            "candidate_report",
            "overlap_report",
            "result_receipt",
        ):
            record[key] = dict(record.get(key) or {})
        record["terminal"] = record.get("status") in _TERMINAL_STATUSES
        return record

    def _assert_schema_ready(self, connection: Any) -> None:
        ready = connection.execute(
            text(
                """
                SELECT
                    to_regclass('public.audience_fresh_data_workflows') IS NOT NULL
                    AND to_regclass(
                        'public.audience_fresh_data_workflow_events'
                    ) IS NOT NULL
                """
            )
        ).scalar()
        if not ready:
            raise RuntimeError(
                "Fresh-data workflow migration 0014 is not ready."
            )

    def _set_tenant_context(self, connection: Any, tenant_id: str) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required.")
        connection.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": tenant_id},
        )

    def _json(self, value: Mapping[str, Any] | None) -> str | None:
        if value is None:
            return None
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        database_url = (
            self._database_url
            or os.getenv("AUDIENCE_FEATURE_WRITER_DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError(
                "AUDIENCE_FEATURE_WRITER_DATABASE_URL is required for fresh-data "
                "workflow state."
            )
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)
