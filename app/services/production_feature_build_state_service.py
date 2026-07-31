from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_feature_build_contracts import (
    ProductionFeatureBuildRequest,
)


_ALLOWED_TRANSITIONS = {
    "claimed": {"validating", "blocked", "quarantined", "failed"},
    "validating": {"embedding", "blocked", "quarantined", "failed"},
    "embedding": {"publishing", "blocked", "quarantined", "failed"},
    "publishing": {"completed", "blocked", "quarantined", "failed"},
    "completed": set(),
    "blocked": set(),
    "quarantined": set(),
    "failed": set(),
}
_TERMINAL_STATUSES = {"completed", "blocked", "quarantined", "failed"}


class ProductionFeatureBuildStateService:
    """Tenant-isolated durable idempotency and lifecycle state for builds."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine

    def claim(
        self,
        request: ProductionFeatureBuildRequest,
    ) -> dict[str, Any]:
        tenant_id = request.source.tenant_id
        engine = self._engine()
        with engine.begin() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, tenant_id)
            inserted = connection.execute(
                text(
                    """
                    INSERT INTO audience_feature_build_jobs (
                        tenant_id,
                        feature_build_id,
                        request_fingerprint,
                        canonical_source_fingerprint,
                        source_ref,
                        source_version,
                        model_fingerprint,
                        data_use_mode,
                        status,
                        expected_feature_count,
                        request_manifest
                    )
                    VALUES (
                        :tenant_id,
                        :feature_build_id,
                        :request_fingerprint,
                        :canonical_source_fingerprint,
                        :source_ref,
                        :source_version,
                        :model_fingerprint,
                        :data_use_mode,
                        'claimed',
                        :expected_feature_count,
                        CAST(:request_manifest AS JSONB)
                    )
                    ON CONFLICT (
                        tenant_id,
                        request_fingerprint
                    )
                    DO NOTHING
                    RETURNING feature_build_id
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "feature_build_id": request.build_id,
                    "request_fingerprint": request.request_fingerprint,
                    "canonical_source_fingerprint": (
                        request.source.source_fingerprint
                    ),
                    "source_ref": request.source.source_ref,
                    "source_version": request.source.source_version,
                    "model_fingerprint": request.model.fingerprint,
                    "data_use_mode": request.source.data_use_mode,
                    "expected_feature_count": (
                        request.source.expected_row_count
                    ),
                    "request_manifest": json.dumps(
                        request.to_safe_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ).mappings().first()
            record = connection.execute(
                text(
                    """
                    SELECT
                        tenant_id,
                        feature_build_id,
                        request_fingerprint,
                        canonical_source_fingerprint,
                        source_ref,
                        source_version,
                        model_fingerprint,
                        data_use_mode,
                        status,
                        reason_code,
                        expected_feature_count,
                        processed_feature_count,
                        feature_set_id,
                        feature_set_version,
                        result_receipt,
                        created_at,
                        updated_at,
                        completed_at
                    FROM audience_feature_build_jobs
                    WHERE tenant_id = :tenant_id
                      AND request_fingerprint = :request_fingerprint
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "request_fingerprint": request.request_fingerprint,
                },
            ).mappings().one()
        safe_record = self._record(record)
        self._verify_claim(safe_record, request)
        return {
            "duplicate": inserted is None,
            "record": safe_record,
        }

    def transition(
        self,
        *,
        tenant_id: str,
        feature_build_id: str,
        status: str,
        reason_code: str | None = None,
        processed_feature_count: int | None = None,
        feature_set_id: str | None = None,
        feature_set_version: int | None = None,
        result_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_tenant_id = normalize_taxonomy_value(tenant_id)
        clean_status = normalize_taxonomy_value(status)
        if clean_status not in _ALLOWED_TRANSITIONS:
            raise ValueError("Unsupported feature-build status.")
        engine = self._engine()
        with engine.begin() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, clean_tenant_id)
            current = connection.execute(
                text(
                    """
                    SELECT
                        status,
                        expected_feature_count
                    FROM audience_feature_build_jobs
                    WHERE tenant_id = :tenant_id
                      AND feature_build_id = :feature_build_id
                    FOR UPDATE
                    """
                ),
                {
                    "tenant_id": clean_tenant_id,
                    "feature_build_id": feature_build_id,
                },
            ).mappings().first()
            if not current:
                raise FileNotFoundError(
                    "Production feature-build job was not found."
                )
            current_status = str(current["status"])
            if clean_status not in _ALLOWED_TRANSITIONS[current_status]:
                raise RuntimeError(
                    f"Invalid feature-build transition: "
                    f"{current_status} -> {clean_status}."
                )
            expected_count = int(current["expected_feature_count"])
            processed_count = (
                int(processed_feature_count)
                if processed_feature_count is not None
                else 0
            )
            if not 0 <= processed_count <= expected_count:
                raise ValueError(
                    "processed_feature_count is outside the expected range."
                )
            if clean_status == "completed":
                if (
                    processed_count != expected_count
                    or not feature_set_id
                    or not feature_set_version
                    or not result_receipt
                ):
                    raise ValueError(
                        "Completed feature builds require a verified full receipt."
                    )
            record = connection.execute(
                text(
                    """
                    UPDATE audience_feature_build_jobs
                    SET
                        status = :status,
                        reason_code = :reason_code,
                        processed_feature_count =
                            :processed_feature_count,
                        feature_set_id = :feature_set_id,
                        feature_set_version = :feature_set_version,
                        result_receipt = CASE
                            WHEN :result_receipt IS NULL THEN NULL
                            ELSE CAST(:result_receipt AS JSONB)
                        END,
                        updated_at = now(),
                        completed_at = CASE
                            WHEN :status = 'completed' THEN now()
                            ELSE NULL
                        END
                    WHERE tenant_id = :tenant_id
                      AND feature_build_id = :feature_build_id
                    RETURNING
                        tenant_id,
                        feature_build_id,
                        request_fingerprint,
                        canonical_source_fingerprint,
                        source_ref,
                        source_version,
                        model_fingerprint,
                        data_use_mode,
                        status,
                        reason_code,
                        expected_feature_count,
                        processed_feature_count,
                        feature_set_id,
                        feature_set_version,
                        result_receipt,
                        created_at,
                        updated_at,
                        completed_at
                    """
                ),
                {
                    "tenant_id": clean_tenant_id,
                    "feature_build_id": feature_build_id,
                    "status": clean_status,
                    "reason_code": (
                        normalize_taxonomy_value(reason_code)
                        if reason_code
                        else None
                    ),
                    "processed_feature_count": processed_count,
                    "feature_set_id": feature_set_id,
                    "feature_set_version": feature_set_version,
                    "result_receipt": (
                        json.dumps(
                            dict(result_receipt),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if result_receipt is not None
                        else None
                    ),
                },
            ).mappings().one()
        return self._record(record)

    def _verify_claim(
        self,
        record: dict[str, Any],
        request: ProductionFeatureBuildRequest,
    ) -> None:
        expected = {
            "tenant_id": request.source.tenant_id,
            "feature_build_id": request.build_id,
            "request_fingerprint": request.request_fingerprint,
            "canonical_source_fingerprint": (
                request.source.source_fingerprint
            ),
            "source_ref": request.source.source_ref,
            "source_version": request.source.source_version,
            "model_fingerprint": request.model.fingerprint,
            "data_use_mode": request.source.data_use_mode,
            "expected_feature_count": request.source.expected_row_count,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "Existing feature-build claim conflicts with the request."
            )

    def _record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(row)
        for key in (
            "expected_feature_count",
            "processed_feature_count",
            "feature_set_version",
        ):
            if record.get(key) is not None:
                record[key] = int(record[key])
        for key in ("created_at", "updated_at", "completed_at"):
            if hasattr(record.get(key), "isoformat"):
                record[key] = record[key].isoformat()
        record["result_receipt"] = dict(
            record.get("result_receipt") or {}
        )
        record["terminal"] = record.get("status") in _TERMINAL_STATUSES
        return record

    def _assert_schema_ready(self, connection: Any) -> None:
        ready = connection.execute(
            text(
                """
                SELECT
                    to_regclass(
                        'public.audience_feature_build_jobs'
                    ) IS NOT NULL AS ready
                """
            )
        ).scalar()
        if not ready:
            raise RuntimeError(
                "Production feature-build registry migration 0008 is not ready."
            )

    def _set_tenant_context(
        self,
        connection: Any,
        tenant_id: str,
    ) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required.")
        connection.execute(
            text(
                """
                SELECT set_config(
                    'app.tenant_id',
                    :tenant_id,
                    true
                )
                """
            ),
            {"tenant_id": tenant_id},
        )

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
                "AUDIENCE_FEATURE_WRITER_DATABASE_URL is required for "
                "feature-build state."
            )
        if database_url.startswith("postgres://"):
            database_url = (
                "postgresql://"
                + database_url[len("postgres://") :]
            )
        return create_engine(database_url, pool_pre_ping=True)
