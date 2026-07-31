from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.models.provider_ingestion_contracts import ProviderDatasetContract


class ProviderContractRegistryService:
    """
    Durable registry for versioned, non-secret provider dataset contracts.

    A registered contract version is immutable. Contract changes require a new
    schema version, which keeps historical replay reproducible.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def register(
        self,
        contract: ProviderDatasetContract,
        *,
        actor: str,
    ) -> Dict[str, Any]:
        if not str(actor or "").strip():
            raise ValueError("actor is required")
        payload = contract.to_safe_dict()
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        engine = self._engine()
        now = datetime.now(timezone.utc).isoformat()

        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            values = {
                "contract_key": contract.contract_key,
                "tenant_id": contract.tenant_id,
                "provider_id": contract.provider_id,
                "dataset_id": contract.dataset_id,
                "schema_version": contract.schema_version,
                "contract_checksum": checksum,
                "contract": serialized,
                "active": True,
                "actor": str(actor).strip(),
                "created_at": now,
                "updated_at": now,
            }
            if engine.dialect.name == "postgresql":
                result = conn.execute(
                    text(
                        """
                        INSERT INTO provider_dataset_contracts (
                            contract_key,
                            tenant_id,
                            provider_id,
                            dataset_id,
                            schema_version,
                            contract_checksum,
                            contract,
                            active,
                            actor,
                            created_at,
                            updated_at
                        )
                        VALUES (
                            :contract_key,
                            :tenant_id,
                            :provider_id,
                            :dataset_id,
                            :schema_version,
                            :contract_checksum,
                            CAST(:contract AS jsonb),
                            :active,
                            :actor,
                            CAST(:created_at AS timestamptz),
                            CAST(:updated_at AS timestamptz)
                        )
                        ON CONFLICT (contract_key) DO NOTHING
                        """
                    ),
                    values,
                )
            else:
                result = conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO provider_dataset_contracts (
                            contract_key,
                            tenant_id,
                            provider_id,
                            dataset_id,
                            schema_version,
                            contract_checksum,
                            contract,
                            active,
                            actor,
                            created_at,
                            updated_at
                        )
                        VALUES (
                            :contract_key,
                            :tenant_id,
                            :provider_id,
                            :dataset_id,
                            :schema_version,
                            :contract_checksum,
                            :contract,
                            :active,
                            :actor,
                            :created_at,
                            :updated_at
                        )
                        """
                    ),
                    values,
                )

            record = self._get_conn(conn, contract.contract_key)
            if record["contract_checksum"] != checksum:
                raise ValueError(
                    "A different immutable provider contract is already "
                    "registered for this schema version."
                )

        return {
            "created": bool(result.rowcount),
            "contract": record,
        }

    def get_active(
        self,
        *,
        tenant_id: str,
        provider_id: str,
        dataset_id: str,
        schema_version: str,
    ) -> ProviderDatasetContract:
        contract_key = ":".join((tenant_id, provider_id, dataset_id, schema_version))
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            record = self._get_conn(conn, contract_key)
        if not record or not record["active"]:
            raise KeyError("No active provider dataset contract was found.")
        return ProviderDatasetContract(**record["contract"])

    def deactivate(self, contract_key: str, *, actor: str) -> Dict[str, Any]:
        if not str(actor or "").strip():
            raise ValueError("actor is required")
        engine = self._engine()
        now = datetime.now(timezone.utc).isoformat()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            existing = self._get_conn(conn, contract_key)
            if not existing:
                raise KeyError("Provider dataset contract was not found.")
            conn.execute(
                text(
                    """
                    UPDATE provider_dataset_contracts
                    SET active = :active,
                        actor = :actor,
                        updated_at = :updated_at
                    WHERE contract_key = :contract_key
                    """
                ),
                {
                    "contract_key": contract_key,
                    "active": False,
                    "actor": str(actor).strip(),
                    "updated_at": now,
                },
            )
            return self._get_conn(conn, contract_key)

    def resolve_for_object(
        self,
        *,
        bucket: str,
        key: str,
        schema_version: str,
    ) -> ProviderDatasetContract:
        normalized_bucket = str(bucket or "").strip()
        normalized_key = str(key or "").strip().lstrip("/")
        normalized_schema = str(schema_version or "").strip()
        if not normalized_bucket or not normalized_key or not normalized_schema:
            raise ValueError(
                "bucket, key, and schema_version are required for contract resolution"
            )

        candidates = [
            contract
            for contract in self.list_active()
            if contract.allowed_bucket == normalized_bucket
            and contract.schema_version == normalized_schema
            and (
                not contract.allowed_prefix
                or normalized_key.startswith(contract.allowed_prefix)
            )
        ]
        if not candidates:
            raise KeyError(
                "No active provider contract matches the object location and schema."
            )

        longest_prefix = max(len(item.allowed_prefix) for item in candidates)
        most_specific = [
            item
            for item in candidates
            if len(item.allowed_prefix) == longest_prefix
        ]
        if len(most_specific) != 1:
            raise ValueError(
                "Provider contract resolution is ambiguous for this object."
            )
        return most_specific[0]

    def list_active(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> list[ProviderDatasetContract]:
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            tenant_clause = (
                "AND tenant_id = :tenant_id" if tenant_id else ""
            )
            rows = conn.execute(
                text(
                    f"""
                    SELECT contract
                    FROM provider_dataset_contracts
                    WHERE active = :active
                      {tenant_clause}
                    ORDER BY tenant_id, provider_id, dataset_id, schema_version
                    """
                ),
                {"active": True, "tenant_id": tenant_id},
            ).fetchall()

        contracts: list[ProviderDatasetContract] = []
        for row in rows:
            payload = row._mapping["contract"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            contracts.append(ProviderDatasetContract(**payload))
        return contracts

    def _get_conn(self, conn, contract_key: str) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            text(
                """
                SELECT *
                FROM provider_dataset_contracts
                WHERE contract_key = :contract_key
                """
            ),
            {"contract_key": contract_key},
        ).fetchone()
        if not row:
            return None
        record = dict(row._mapping)
        if isinstance(record.get("contract"), str):
            record["contract"] = json.loads(record["contract"])
        record["active"] = bool(record.get("active"))
        return record

    def _ensure_table(self, conn, dialect_name: str) -> None:
        # PostgreSQL schema is operator-owned and installed by migrations.
        if dialect_name == "postgresql":
            return
        if dialect_name == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_dataset_contracts (
                        contract_key TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        provider_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        contract_checksum TEXT NOT NULL,
                        contract JSONB NOT NULL,
                        active BOOLEAN NOT NULL DEFAULT TRUE,
                        actor TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (
                            tenant_id,
                            provider_id,
                            dataset_id,
                            schema_version
                        )
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS provider_dataset_contracts (
                        contract_key TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        provider_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        contract_checksum TEXT NOT NULL,
                        contract TEXT NOT NULL,
                        active INTEGER NOT NULL DEFAULT 1,
                        actor TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (
                            tenant_id,
                            provider_id,
                            dataset_id,
                            schema_version
                        )
                    )
                    """
                )
            )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_contract_lookup
                ON provider_dataset_contracts (
                    tenant_id,
                    provider_id,
                    dataset_id,
                    schema_version,
                    active
                )
                """
            )
        )

    def _engine(self):
        database_url = (
            self._explicit_database_url
            or os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        )
        if not database_url:
            raise RuntimeError(
                "Provider contract registry requires the explicit Punk-owned "
                "PROVIDER_INGESTION_DATABASE_URL."
            )
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)
