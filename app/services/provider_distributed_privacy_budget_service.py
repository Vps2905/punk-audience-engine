from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.models.provider_scale_contracts import (
    ProviderDistributedJobRequest,
)


class ProviderDistributedPrivacyBudgetService:
    """
    Atomic, idempotent privacy-budget reservation per immutable source object.

    Reserved and completed releases both count against the budget. A failed
    release remains charged conservatively because the control plane cannot
    prove that no privacy-safe output was observed before the failure.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def reserve(
        self,
        request: ProviderDistributedJobRequest,
    ) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Distributed privacy accounting requires PostgreSQL."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        release_id = (
            "provider_privacy_"
            + hashlib.sha256(
                (
                    f"{request.ingestion_id}\x1f"
                    f"{request.fingerprint}\x1f"
                    f"{request.dispatch_attempt}"
                ).encode("utf-8")
            ).hexdigest()[:32]
        )
        scope = self._scope(request)
        composition_group = self._composition_group(request)
        privacy_partition_index = self._partition_index(request)
        now = datetime.now(timezone.utc).isoformat()

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            if engine.dialect.name == "postgresql":
                conn.execute(
                    text(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtextextended(:budget_scope, 0)
                        )
                        """
                    ),
                    {"budget_scope": scope},
                )

            existing = conn.execute(
                text(
                    """
                    SELECT *
                    FROM provider_privacy_releases
                    WHERE ingestion_id = :ingestion_id
                      AND dispatch_attempt = :dispatch_attempt
                    """
                ),
                {
                    "ingestion_id": request.ingestion_id,
                    "dispatch_attempt": request.dispatch_attempt,
                },
            ).mappings().first()
            if existing:
                self._validate_replay(
                    dict(existing),
                    request=request,
                    budget_scope=scope,
                )
                return self._result(dict(existing), replayed=True)

            used_before = float(
                conn.execute(
                    text(
                        """
                        SELECT COALESCE(SUM(charged_epsilon), 0)
                        FROM provider_privacy_releases
                        WHERE budget_scope = :budget_scope
                          AND decision = 'allowed'
                        """
                    ),
                    {"budget_scope": scope},
                ).scalar()
                or 0.0
            )
            epsilon = float(request.contract.epsilon)
            existing_group = conn.execute(
                text(
                    """
                    SELECT epsilon, delta, sensitivity, mechanism
                    FROM provider_privacy_releases
                    WHERE budget_scope = :budget_scope
                      AND composition_group = :composition_group
                      AND decision = 'allowed'
                    LIMIT 1
                    """
                ),
                {
                    "budget_scope": scope,
                    "composition_group": composition_group,
                },
            ).mappings().first()
            if existing_group:
                expected_group = {
                    "epsilon": epsilon,
                    "delta": float(request.contract.delta),
                    "sensitivity": float(request.contract.sensitivity),
                    "mechanism": request.contract.mechanism,
                }
                for key, expected in expected_group.items():
                    actual = existing_group[key]
                    if str(actual) != str(expected):
                        raise ValueError(
                            "Parallel privacy composition parameters changed"
                        )
                same_partition_release = conn.execute(
                    text(
                        """
                        SELECT 1
                        FROM provider_privacy_releases
                        WHERE budget_scope = :budget_scope
                          AND composition_group = :composition_group
                          AND privacy_partition_index = :partition_index
                          AND decision = 'allowed'
                        LIMIT 1
                        """
                    ),
                    {
                        "budget_scope": scope,
                        "composition_group": composition_group,
                        "partition_index": privacy_partition_index,
                    },
                ).first()
                charged_epsilon = epsilon if same_partition_release else 0.0
            else:
                charged_epsilon = epsilon
            max_budget = float(
                request.contract.max_cumulative_epsilon
            )
            if used_before + charged_epsilon > max_budget:
                decision = "blocked"
                status = "blocked"
                reason_code = "privacy_budget_exceeded"
                budget_after = used_before
                charged_epsilon = 0.0
            else:
                decision = "allowed"
                status = "reserved"
                reason_code = None
                budget_after = used_before + charged_epsilon

            values = {
                "release_id": release_id,
                "ingestion_id": request.ingestion_id,
                "fingerprint": request.fingerprint,
                "dispatch_attempt": request.dispatch_attempt,
                "tenant_id": request.contract.tenant_id,
                "provider_id": request.contract.provider_id,
                "dataset_id": request.contract.dataset_id,
                "schema_version": request.contract.schema_version,
                "budget_scope": scope,
                "composition_group": composition_group,
                "privacy_partition_index": privacy_partition_index,
                "epsilon": epsilon,
                "charged_epsilon": charged_epsilon,
                "delta": float(request.contract.delta),
                "sensitivity": float(request.contract.sensitivity),
                "mechanism": request.contract.mechanism,
                "max_budget": max_budget,
                "budget_before": used_before,
                "budget_after": budget_after,
                "decision": decision,
                "status": status,
                "reason_code": reason_code,
                "created_at": now,
                "updated_at": now,
            }
            conn.execute(
                text(
                    """
                    INSERT INTO provider_privacy_releases (
                        release_id,
                        ingestion_id,
                        fingerprint,
                        dispatch_attempt,
                        tenant_id,
                        provider_id,
                        dataset_id,
                        schema_version,
                        budget_scope,
                        composition_group,
                        privacy_partition_index,
                        epsilon,
                        charged_epsilon,
                        delta,
                        sensitivity,
                        mechanism,
                        max_budget,
                        budget_before,
                        budget_after,
                        decision,
                        status,
                        reason_code,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :release_id,
                        :ingestion_id,
                        :fingerprint,
                        :dispatch_attempt,
                        :tenant_id,
                        :provider_id,
                        :dataset_id,
                        :schema_version,
                        :budget_scope,
                        :composition_group,
                        :privacy_partition_index,
                        :epsilon,
                        :charged_epsilon,
                        :delta,
                        :sensitivity,
                        :mechanism,
                        :max_budget,
                        :budget_before,
                        :budget_after,
                        :decision,
                        :status,
                        :reason_code,
                        :created_at,
                        :updated_at
                    )
                    """
                ),
                values,
            )
        return self._result(values, replayed=False)

    def mark_terminal(
        self,
        release_id: str,
        *,
        status: str,
        canonical_ref: Optional[str] = None,
        reason_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        if status not in {"completed", "blocked", "failed"}:
            raise ValueError(
                "Privacy release status must be completed, blocked, or failed"
            )
        db_url = self._db_url()
        if not db_url:
            raise RuntimeError(
                "Distributed privacy accounting requires PostgreSQL."
            )
        engine = create_engine(self._connection_url(db_url), pool_pre_ping=True)
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            existing = conn.execute(
                text(
                    """
                    SELECT *
                    FROM provider_privacy_releases
                    WHERE release_id = :release_id
                    """
                ),
                {"release_id": release_id},
            ).mappings().first()
            if not existing:
                raise KeyError(
                    f"Privacy release was not found: {release_id}"
                )
            existing = dict(existing)
            if existing["status"] in {"completed", "blocked", "failed"}:
                if (
                    existing["status"] == status
                    and existing.get("canonical_ref") == canonical_ref
                    and existing.get("reason_code") == reason_code
                ):
                    return self._result(existing, replayed=True)
                raise ValueError(
                    "A terminal privacy release cannot be replaced"
                )
            if existing["decision"] == "blocked" and status != "blocked":
                raise ValueError(
                    "A blocked privacy reservation cannot publish output"
                )
            if status == "completed" and not canonical_ref:
                raise ValueError(
                    "Completed privacy release requires a canonical reference"
                )
            if status != "completed" and canonical_ref:
                raise ValueError(
                    "Non-completed privacy release cannot publish output"
                )
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                text(
                    """
                    UPDATE provider_privacy_releases
                    SET status = :status,
                        canonical_ref = :canonical_ref,
                        reason_code = :reason_code,
                        updated_at = :updated_at,
                        completed_at = :completed_at
                    WHERE release_id = :release_id
                      AND status = 'reserved'
                    """
                ),
                {
                    "release_id": release_id,
                    "status": status,
                    "canonical_ref": canonical_ref,
                    "reason_code": reason_code,
                    "updated_at": now,
                    "completed_at": now,
                },
            )
            updated = conn.execute(
                text(
                    """
                    SELECT *
                    FROM provider_privacy_releases
                    WHERE release_id = :release_id
                    """
                ),
                {"release_id": release_id},
            ).mappings().one()
        return self._result(dict(updated), replayed=False)

    def _scope(self, request: ProviderDistributedJobRequest) -> str:
        return (
            f"provider:{request.contract.tenant_id}:"
            f"{request.contract.provider_id}:"
            f"{request.contract.dataset_id}:"
            f"{request.contract.schema_version}"
        )

    def _composition_group(
        self,
        request: ProviderDistributedJobRequest,
    ) -> str:
        manifest = request.manifest
        if (
            request.contract.distributed_partition_strategy
            == "entity_hash_v1"
            and manifest.delivery_window_id
        ):
            return f"window:{manifest.delivery_window_id}"
        return f"object:{request.fingerprint}:attempt:{request.dispatch_attempt}"

    def _partition_index(
        self,
        request: ProviderDistributedJobRequest,
    ) -> int:
        value = request.manifest.partition_index
        return int(value) if value is not None else -1

    def _validate_replay(
        self,
        existing: Dict[str, Any],
        *,
        request: ProviderDistributedJobRequest,
        budget_scope: str,
    ) -> None:
        expected = {
            "fingerprint": request.fingerprint,
            "dispatch_attempt": int(request.dispatch_attempt),
            "budget_scope": budget_scope,
            "composition_group": self._composition_group(request),
            "privacy_partition_index": self._partition_index(request),
            "epsilon": float(request.contract.epsilon),
            "delta": float(request.contract.delta),
            "sensitivity": float(request.contract.sensitivity),
            "mechanism": request.contract.mechanism,
            "max_budget": float(
                request.contract.max_cumulative_epsilon
            ),
        }
        for key, value in expected.items():
            actual = existing.get(key)
            if isinstance(value, float):
                if float(actual) != value:
                    raise ValueError(
                        "Idempotent privacy reservation parameters changed"
                    )
            elif str(actual) != str(value):
                raise ValueError(
                    "Idempotent privacy reservation parameters changed"
                )

    def _result(
        self,
        record: Dict[str, Any],
        *,
        replayed: bool,
    ) -> Dict[str, Any]:
        return {
            "release_id": record["release_id"],
            "ingestion_id": record["ingestion_id"],
            "fingerprint": record["fingerprint"],
            "dispatch_attempt": int(record["dispatch_attempt"]),
            "budget_scope": record["budget_scope"],
            "composition_group": record["composition_group"],
            "privacy_partition_index": int(
                record["privacy_partition_index"]
            ),
            "epsilon": float(record["epsilon"]),
            "charged_epsilon": float(record["charged_epsilon"]),
            "delta": float(record["delta"]),
            "sensitivity": float(record["sensitivity"]),
            "mechanism": record["mechanism"],
            "max_budget": float(record["max_budget"]),
            "budget_before": float(record["budget_before"]),
            "budget_after": float(record["budget_after"]),
            "decision": record["decision"],
            "status": record["status"],
            "reason_code": record.get("reason_code"),
            "replayed": replayed,
        }

    def _ensure_table(self, conn: Any, dialect: str) -> None:
        # PostgreSQL schema changes are restricted to controlled migrations.
        if dialect == "postgresql":
            return
        if dialect == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_privacy_releases (
                        release_id TEXT PRIMARY KEY,
                        ingestion_id TEXT NOT NULL,
                        fingerprint CHAR(64) NOT NULL,
                        dispatch_attempt INTEGER NOT NULL,
                        tenant_id TEXT NOT NULL,
                        provider_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        budget_scope TEXT NOT NULL,
                        composition_group TEXT NOT NULL,
                        privacy_partition_index INTEGER NOT NULL,
                        epsilon DOUBLE PRECISION NOT NULL,
                        charged_epsilon DOUBLE PRECISION NOT NULL,
                        delta DOUBLE PRECISION NOT NULL,
                        sensitivity DOUBLE PRECISION NOT NULL,
                        mechanism TEXT NOT NULL,
                        max_budget DOUBLE PRECISION NOT NULL,
                        budget_before DOUBLE PRECISION NOT NULL,
                        budget_after DOUBLE PRECISION NOT NULL,
                        decision TEXT NOT NULL,
                        status TEXT NOT NULL,
                        canonical_ref TEXT,
                        reason_code TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        completed_at TIMESTAMPTZ,
                        UNIQUE (ingestion_id, dispatch_attempt)
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_privacy_releases (
                        release_id TEXT PRIMARY KEY,
                        ingestion_id TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        dispatch_attempt INTEGER NOT NULL,
                        tenant_id TEXT NOT NULL,
                        provider_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        budget_scope TEXT NOT NULL,
                        composition_group TEXT NOT NULL,
                        privacy_partition_index INTEGER NOT NULL,
                        epsilon REAL NOT NULL,
                        charged_epsilon REAL NOT NULL,
                        delta REAL NOT NULL,
                        sensitivity REAL NOT NULL,
                        mechanism TEXT NOT NULL,
                        max_budget REAL NOT NULL,
                        budget_before REAL NOT NULL,
                        budget_after REAL NOT NULL,
                        decision TEXT NOT NULL,
                        status TEXT NOT NULL,
                        canonical_ref TEXT,
                        reason_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        UNIQUE (ingestion_id, dispatch_attempt)
                    )
                    """
                )
            )

    def _db_url(self) -> Optional[str]:
        return (
            self._explicit_database_url
            or os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        )

    def _connection_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
