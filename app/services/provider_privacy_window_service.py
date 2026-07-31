from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.models.provider_privacy_window_contracts import (
    ProviderPrivacyPartitionRegistration,
)


class ProviderPrivacyWindowService:
    """
    Durable coordination for entity-disjoint provider privacy partitions.

    A window is publishable only after every current partition has completed
    its privacy transformation. A changed partition must be declared as a
    correction and must name the exact fingerprint it supersedes.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def register_partition(
        self,
        registration: ProviderPrivacyPartitionRegistration,
    ) -> Dict[str, Any]:
        engine = self._engine()
        now = self._now()
        with engine.begin() as conn:
            self._ensure_tables(conn, engine.dialect.name)
            self._set_tenant(
                conn, engine.dialect.name, registration.tenant_id
            )
            self._lock(conn, engine.dialect.name, registration.window_key)
            window = self._window(conn, registration.window_key)
            if window is None:
                conn.execute(
                    text(
                        """
                        INSERT INTO provider_privacy_windows (
                            window_key, tenant_id, provider_id, dataset_id,
                            schema_version, delivery_window_id,
                            event_time_start, event_time_end,
                            partition_count, partition_algorithm, status,
                            created_at, updated_at
                        ) VALUES (
                            :window_key, :tenant_id, :provider_id, :dataset_id,
                            :schema_version, :delivery_window_id,
                            :event_time_start, :event_time_end,
                            :partition_count, :partition_algorithm, 'open',
                            :created_at, :updated_at
                        )
                        """
                    ),
                    {
                        **registration.to_safe_dict(),
                        "window_key": registration.window_key,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            else:
                self._validate_window(window, registration)
                if window["status"] == "revoked":
                    raise ValueError(
                        "A revoked privacy window cannot accept data"
                    )
                if (
                    window["status"] == "sealed"
                    and registration.delivery_type != "correction"
                ):
                    raise ValueError(
                        "A sealed privacy window accepts only exact corrections"
                    )

            current = conn.execute(
                text(
                    """
                    SELECT *
                    FROM provider_privacy_window_partitions
                    WHERE window_key = :window_key
                      AND partition_index = :partition_index
                      AND is_current = 1
                    """
                ),
                {
                    "window_key": registration.window_key,
                    "partition_index": registration.partition_index,
                },
            ).mappings().first()
            if current:
                current = dict(current)
                if current["fingerprint"] == registration.fingerprint:
                    return self._result(
                        conn,
                        registration.window_key,
                        replayed=True,
                    )
                if (
                    registration.delivery_type != "correction"
                    or registration.supersedes_fingerprint
                    != current["fingerprint"]
                ):
                    raise ValueError(
                        "A partition replacement requires an exact correction"
                    )
                if window and window["status"] == "sealed":
                    conn.execute(
                        text(
                            """
                            UPDATE provider_privacy_windows
                            SET status = 'open', sealed_at = NULL,
                                updated_at = :updated_at
                            WHERE window_key = :window_key
                              AND status = 'sealed'
                            """
                        ),
                        {
                            "window_key": registration.window_key,
                            "updated_at": now,
                        },
                    )
                    conn.execute(
                        text(
                            """
                            UPDATE provider_canonical_partitions
                            SET status = 'candidate',
                                updated_at = :updated_at
                            WHERE window_key = :window_key
                              AND status = 'active'
                            """
                        ),
                        {
                            "window_key": registration.window_key,
                            "updated_at": now,
                        },
                    )
                conn.execute(
                    text(
                        """
                        UPDATE provider_privacy_window_partitions
                        SET is_current = 0,
                            status = 'superseded',
                            updated_at = :updated_at
                        WHERE fingerprint = :fingerprint
                        """
                    ),
                    {
                        "fingerprint": current["fingerprint"],
                        "updated_at": now,
                    },
                )
                conn.execute(
                    text(
                        """
                        UPDATE provider_canonical_partitions
                        SET status = 'revoked',
                            revocation_reason = 'superseded_correction',
                            updated_at = :updated_at
                        WHERE canonical_partition_id = :canonical_partition_id
                          AND status IN ('candidate', 'active')
                        """
                    ),
                    {
                        "canonical_partition_id": (
                            f"canonical:{current['fingerprint']}"
                        ),
                        "updated_at": now,
                    },
                )

            conn.execute(
                text(
                    """
                    INSERT INTO provider_privacy_window_partitions (
                        fingerprint, window_key, partition_index,
                        partition_count, ingestion_id, row_count,
                        delivery_type, supersedes_fingerprint, status,
                        is_current, created_at, updated_at
                    ) VALUES (
                        :fingerprint, :window_key, :partition_index,
                        :partition_count, :ingestion_id, :row_count,
                        :delivery_type, :supersedes_fingerprint, 'registered',
                        1, :created_at, :updated_at
                    )
                    """
                ),
                {
                    **registration.to_safe_dict(),
                    "window_key": registration.window_key,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            conn.execute(
                text(
                    """
                    UPDATE provider_privacy_windows
                    SET updated_at = :updated_at
                    WHERE window_key = :window_key
                    """
                ),
                {
                    "window_key": registration.window_key,
                    "updated_at": now,
                },
            )
            return self._result(
                conn,
                registration.window_key,
                replayed=False,
            )

    def mark_partition_ready(
        self,
        *,
        tenant_id: str,
        fingerprint: str,
        canonical_ref: str,
        canonical_checksum_sha256: str,
        privacy_release_id: str,
        output_rows: int,
    ) -> Dict[str, Any]:
        if not canonical_ref or not privacy_release_id:
            raise ValueError(
                "Canonical reference and privacy release are required"
            )
        if int(output_rows) < 0:
            raise ValueError("output_rows must be >= 0")
        checksum = str(canonical_checksum_sha256 or "").strip().lower()
        if len(checksum) != 64 or any(
            value not in "0123456789abcdef" for value in checksum
        ):
            raise ValueError("canonical_checksum_sha256 must be SHA-256")
        engine = self._engine()
        now = self._now()
        with engine.begin() as conn:
            self._ensure_tables(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, tenant_id)
            row = conn.execute(
                text(
                    """
                    SELECT *
                    FROM provider_privacy_window_partitions
                    WHERE fingerprint = :fingerprint
                      AND is_current = 1
                    """
                ),
                {"fingerprint": fingerprint},
            ).mappings().first()
            if not row:
                raise KeyError("Current privacy partition was not found")
            row = dict(row)
            window = self._window(conn, row["window_key"])
            if not window or window["tenant_id"] != tenant_id:
                raise KeyError("Current privacy partition was not found")
            if row["status"] == "ready":
                if (
                    row["canonical_ref"] == canonical_ref
                    and row["privacy_release_id"] == privacy_release_id
                    and int(row["output_rows"]) == int(output_rows)
                ):
                    return self._result(
                        conn,
                        row["window_key"],
                        replayed=True,
                    )
                raise ValueError(
                    "A ready privacy partition cannot be replaced"
                )
            if row["status"] != "registered":
                raise ValueError("Privacy partition is not publishable")
            conn.execute(
                text(
                    """
                    UPDATE provider_privacy_window_partitions
                    SET status = 'ready',
                        canonical_ref = :canonical_ref,
                        privacy_release_id = :privacy_release_id,
                        output_rows = :output_rows,
                        updated_at = :updated_at
                    WHERE fingerprint = :fingerprint
                      AND status = 'registered'
                      AND is_current = 1
                    """
                ),
                {
                    "fingerprint": fingerprint,
                    "canonical_ref": canonical_ref,
                    "privacy_release_id": privacy_release_id,
                    "output_rows": int(output_rows),
                    "updated_at": now,
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO provider_canonical_partitions (
                        canonical_partition_id, window_key,
                        partition_index, canonical_ref,
                        checksum_sha256, status, created_at, updated_at
                    ) VALUES (
                        :canonical_partition_id, :window_key,
                        :partition_index, :canonical_ref,
                        :checksum_sha256, 'candidate', :created_at, :updated_at
                    )
                    """
                ),
                {
                    "canonical_partition_id": f"canonical:{fingerprint}",
                    "window_key": row["window_key"],
                    "partition_index": int(row["partition_index"]),
                    "canonical_ref": canonical_ref,
                    "checksum_sha256": checksum,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            return self._result(
                conn,
                row["window_key"],
                replayed=False,
            )

    def seal_window(
        self,
        window_key: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        engine = self._engine()
        now = self._now()
        with engine.begin() as conn:
            self._ensure_tables(conn, engine.dialect.name)
            self._set_tenant(
                conn,
                engine.dialect.name,
                tenant_id or self._tenant_from_window_key(window_key),
            )
            self._lock(conn, engine.dialect.name, window_key)
            window = self._window(conn, window_key)
            if not window:
                raise KeyError("Privacy window was not found")
            if window["status"] == "sealed":
                return self._result(conn, window_key, replayed=True)
            if window["status"] != "open":
                raise ValueError("Privacy window cannot be sealed")
            rows = conn.execute(
                text(
                    """
                    SELECT partition_index, status
                    FROM provider_privacy_window_partitions
                    WHERE window_key = :window_key
                      AND is_current = 1
                    ORDER BY partition_index
                    """
                ),
                {"window_key": window_key},
            ).mappings().all()
            expected = list(range(int(window["partition_count"])))
            actual = [int(row["partition_index"]) for row in rows]
            if actual != expected or any(row["status"] != "ready" for row in rows):
                raise ValueError(
                    "Privacy window is incomplete or contains unready partitions"
                )
            conn.execute(
                text(
                    """
                    UPDATE provider_privacy_windows
                    SET status = 'sealed',
                        sealed_at = :sealed_at,
                        updated_at = :updated_at
                    WHERE window_key = :window_key
                      AND status = 'open'
                    """
                ),
                {
                    "window_key": window_key,
                    "sealed_at": now,
                    "updated_at": now,
                },
            )
            conn.execute(
                text(
                    """
                    UPDATE provider_canonical_partitions
                    SET status = 'active', updated_at = :updated_at
                    WHERE window_key = :window_key
                      AND status = 'candidate'
                    """
                ),
                {"window_key": window_key, "updated_at": now},
            )
            return self._result(conn, window_key, replayed=False)

    def revoke_scope(
        self,
        *,
        tenant_id: str,
        provider_id: str,
        dataset_id: str,
        reason_code: str,
        event_time_start: Optional[str] = None,
        event_time_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not reason_code:
            raise ValueError("reason_code is required")
        clauses = [
            "tenant_id = :tenant_id",
            "provider_id = :provider_id",
            "dataset_id = :dataset_id",
            "status IN ('open', 'sealed')",
        ]
        params: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "provider_id": provider_id,
            "dataset_id": dataset_id,
            "reason_code": reason_code,
            "updated_at": self._now(),
        }
        if event_time_start and event_time_end:
            clauses.extend(
                [
                    "event_time_end > :event_time_start",
                    "event_time_start < :event_time_end",
                ]
            )
            params.update(
                {
                    "event_time_start": event_time_start,
                    "event_time_end": event_time_end,
                }
            )
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_tables(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, tenant_id)
            result = conn.execute(
                text(
                    f"""
                    UPDATE provider_privacy_windows
                    SET status = 'revoked',
                        revocation_reason = :reason_code,
                        updated_at = :updated_at
                    WHERE {' AND '.join(clauses)}
                    """
                ),
                params,
            )
            conn.execute(
                text(
                    """
                    UPDATE provider_canonical_partitions
                    SET status = 'revoked',
                        revocation_reason = :reason_code,
                        updated_at = :updated_at
                    WHERE window_key IN (
                        SELECT window_key
                        FROM provider_privacy_windows
                        WHERE tenant_id = :tenant_id
                          AND provider_id = :provider_id
                          AND dataset_id = :dataset_id
                          AND status = 'revoked'
                    )
                      AND status = 'active'
                    """
                ),
                params,
            )
            return {
                "status": "revoked",
                "revoked_window_count": max(0, int(result.rowcount or 0)),
                "reason_code": reason_code,
            }

    def get_window(
        self,
        window_key: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_tables(conn, engine.dialect.name)
            self._set_tenant(
                conn,
                engine.dialect.name,
                tenant_id or self._tenant_from_window_key(window_key),
            )
            if self._window(conn, window_key) is None:
                raise KeyError("Privacy window was not found")
            return self._result(conn, window_key, replayed=False)

    def publication_health(
        self,
        *,
        tenant_id: str,
        stale_after_seconds: int = 7_200,
        limit: int = 200,
    ) -> Dict[str, Any]:
        if int(stale_after_seconds) < 60:
            raise ValueError("stale_after_seconds must be >= 60")
        if not 1 <= int(limit) <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        engine = self._engine()
        now = datetime.now(timezone.utc)
        with engine.begin() as conn:
            self._ensure_tables(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, tenant_id)
            rows = conn.execute(
                text(
                    """
                    SELECT window_key, provider_id, dataset_id,
                           partition_count, status, updated_at
                    FROM provider_privacy_windows
                    WHERE tenant_id = :tenant_id
                      AND status = 'open'
                    ORDER BY updated_at
                    LIMIT :limit
                    """
                ),
                {"tenant_id": tenant_id, "limit": int(limit)},
            ).mappings().all()
            items = []
            for raw in rows:
                row = dict(raw)
                state = self._result(
                    conn, row["window_key"], replayed=False
                )
                updated = datetime.fromisoformat(
                    str(row["updated_at"]).replace("Z", "+00:00")
                )
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                age = max(0.0, (now - updated).total_seconds())
                items.append(
                    {
                        "window_key": row["window_key"],
                        "provider_id": row["provider_id"],
                        "dataset_id": row["dataset_id"],
                        "partition_count": int(row["partition_count"]),
                        "registered_partition_count": state[
                            "registered_partition_count"
                        ],
                        "ready_partition_count": state[
                            "ready_partition_count"
                        ],
                        "age_seconds": round(age, 3),
                        "stale": age > int(stale_after_seconds),
                    }
                )
        return {
            "open_window_count": len(items),
            "stale_open_window_count": sum(
                bool(item["stale"]) for item in items
            ),
            "windows": items,
        }

    def _result(
        self,
        conn: Any,
        window_key: str,
        *,
        replayed: bool,
    ) -> Dict[str, Any]:
        window = self._window(conn, window_key)
        if not window:
            raise KeyError("Privacy window was not found")
        counts = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) AS registered_count,
                    SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END)
                        AS ready_count,
                    COALESCE(SUM(output_rows), 0) AS output_rows
                FROM provider_privacy_window_partitions
                WHERE window_key = :window_key
                  AND is_current = 1
                """
            ),
            {"window_key": window_key},
        ).mappings().one()
        return {
            "window_key": window["window_key"],
            "tenant_id": window["tenant_id"],
            "provider_id": window["provider_id"],
            "dataset_id": window["dataset_id"],
            "schema_version": window["schema_version"],
            "delivery_window_id": window["delivery_window_id"],
            "status": window["status"],
            "partition_count": int(window["partition_count"]),
            "registered_partition_count": int(
                counts["registered_count"] or 0
            ),
            "ready_partition_count": int(counts["ready_count"] or 0),
            "output_rows": int(counts["output_rows"] or 0),
            "event_time_start": str(window["event_time_start"]),
            "event_time_end": str(window["event_time_end"]),
            "replayed": replayed,
            "canonical_publication_status": (
                "active" if window["status"] == "sealed" else "blocked"
            ),
        }

    def _validate_window(
        self,
        window: Dict[str, Any],
        registration: ProviderPrivacyPartitionRegistration,
    ) -> None:
        expected = {
            "tenant_id": registration.tenant_id,
            "provider_id": registration.provider_id,
            "dataset_id": registration.dataset_id,
            "schema_version": registration.schema_version,
            "delivery_window_id": registration.delivery_window_id,
            "event_time_start": registration.event_time_start,
            "event_time_end": registration.event_time_end,
            "partition_count": registration.partition_count,
            "partition_algorithm": registration.partition_algorithm,
        }
        for field_name, expected_value in expected.items():
            actual = window[field_name]
            if field_name.startswith("event_time_"):
                actual = self._normalized_timestamp(actual)
                expected_value = self._normalized_timestamp(expected_value)
            if str(actual) != str(expected_value):
                raise ValueError(
                    "Privacy window parameters changed after registration"
                )

    def _window(self, conn: Any, window_key: str) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            text(
                """
                SELECT * FROM provider_privacy_windows
                WHERE window_key = :window_key
                """
            ),
            {"window_key": window_key},
        ).mappings().first()
        return dict(row) if row else None

    def _lock(self, conn: Any, dialect: str, window_key: str) -> None:
        if dialect == "postgresql":
            conn.execute(
                text(
                    "SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"
                ),
                {"key": window_key},
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

    def _tenant_from_window_key(self, window_key: str) -> str:
        tenant_id, separator, _ = str(window_key or "").partition(":")
        if not separator or not tenant_id:
            raise ValueError("Privacy window key does not contain a tenant")
        return tenant_id

    def _normalized_timestamp(self, value: Any) -> str:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(
                str(value).strip().replace("Z", "+00:00")
            )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def _ensure_tables(self, conn: Any, dialect: str) -> None:
        # PostgreSQL tables and tenant policies are migration-owned.
        if dialect == "postgresql":
            return
        timestamp = "TIMESTAMPTZ" if dialect == "postgresql" else "TEXT"
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS provider_privacy_windows (
                    window_key TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    delivery_window_id TEXT NOT NULL,
                    event_time_start {timestamp} NOT NULL,
                    event_time_end {timestamp} NOT NULL,
                    partition_count INTEGER NOT NULL,
                    partition_algorithm TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revocation_reason TEXT,
                    created_at {timestamp} NOT NULL,
                    updated_at {timestamp} NOT NULL,
                    sealed_at {timestamp}
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS provider_privacy_window_partitions (
                    fingerprint TEXT PRIMARY KEY,
                    window_key TEXT NOT NULL,
                    partition_index INTEGER NOT NULL,
                    partition_count INTEGER NOT NULL,
                    ingestion_id TEXT NOT NULL,
                    row_count INTEGER,
                    delivery_type TEXT NOT NULL,
                    supersedes_fingerprint TEXT,
                    status TEXT NOT NULL,
                    is_current INTEGER NOT NULL,
                    canonical_ref TEXT,
                    privacy_release_id TEXT,
                    output_rows INTEGER,
                    created_at {timestamp} NOT NULL,
                    updated_at {timestamp} NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS provider_canonical_partitions (
                    canonical_partition_id TEXT PRIMARY KEY,
                    window_key TEXT NOT NULL,
                    partition_index INTEGER NOT NULL,
                    canonical_ref TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revocation_reason TEXT,
                    created_at {timestamp} NOT NULL,
                    updated_at {timestamp} NOT NULL
                )
                """
            )
        )

    def _engine(self) -> Any:
        database_url = (
            self._explicit_database_url
            or os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        )
        if not database_url:
            raise RuntimeError(
                "Provider privacy windows require PostgreSQL."
            )
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
