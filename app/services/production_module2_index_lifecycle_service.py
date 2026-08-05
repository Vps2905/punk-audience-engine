from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import stable_digest
from app.models.production_module2_completion_contracts import (
    Module2IndexManifest,
    Module2IndexValidationEvidence,
)
from app.services.production_module2_certification_service import (
    validate_module2_certification_report,
)
from app.services.production_module2_shadow_serving_service import (
    validate_module2_shadow_readiness_report,
)


class Module2IndexLifecycleRepository(Protocol):
    def create(self, record: Mapping[str, Any]) -> dict[str, Any]: ...

    def get(self, *, tenant_id: str, index_id: str, index_version: int) -> dict[str, Any] | None: ...

    def transition(
        self,
        *,
        tenant_id: str,
        index_id: str,
        index_version: int,
        expected_status: str,
        next_status: str,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def activate_atomic(
        self,
        *,
        tenant_id: str,
        index_id: str,
        index_version: int,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def rollback_atomic(
        self,
        *,
        tenant_id: str,
        target_index_id: str,
        target_index_version: int,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]: ...


class InMemoryModule2IndexLifecycleRepository:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str, int], dict[str, Any]] = {}

    def create(self, record: Mapping[str, Any]) -> dict[str, Any]:
        safe = deepcopy(dict(record))
        key = self._key(safe["tenant_id"], safe["index_id"], safe["index_version"])
        existing = self._records.get(key)
        if existing is not None:
            if existing["manifest_fingerprint"] != safe["manifest_fingerprint"]:
                raise RuntimeError("Index identity already exists with different manifest.")
            return deepcopy(existing)
        self._records[key] = safe
        return deepcopy(safe)

    def get(self, *, tenant_id: str, index_id: str, index_version: int) -> dict[str, Any] | None:
        value = self._records.get(self._key(tenant_id, index_id, index_version))
        return deepcopy(value) if value is not None else None

    def transition(
        self,
        *,
        tenant_id: str,
        index_id: str,
        index_version: int,
        expected_status: str,
        next_status: str,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        key = self._key(tenant_id, index_id, index_version)
        record = self._records.get(key)
        if record is None:
            raise RuntimeError("Index record was not found.")
        if record["status"] != expected_status:
            raise RuntimeError(
                f"Index status transition requires {expected_status}; observed {record['status']}."
            )
        record = deepcopy(record)
        record["status"] = next_status
        record["latest_evidence"] = deepcopy(dict(evidence))
        record["event_count"] = int(record.get("event_count", 0)) + 1
        self._records[key] = record
        return deepcopy(record)

    def activate_atomic(
        self,
        *,
        tenant_id: str,
        index_id: str,
        index_version: int,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_key = self._key(tenant_id, index_id, index_version)
        target = self._records.get(target_key)
        if target is None or target["status"] != "shadow":
            raise RuntimeError("Only a shadow index may be activated.")
        for key, record in list(self._records.items()):
            if key[0] == tenant_id and record["status"] == "active":
                retired = deepcopy(record)
                retired["status"] = "retired"
                retired["latest_evidence"] = {
                    "reason_code": "superseded_by_atomic_activation",
                    "replacement_manifest_fingerprint": target[
                        "manifest_fingerprint"
                    ],
                }
                retired["event_count"] = int(retired.get("event_count", 0)) + 1
                self._records[key] = retired
        activated = deepcopy(target)
        activated["status"] = "active"
        activated["latest_evidence"] = deepcopy(dict(evidence))
        activated["event_count"] = int(activated.get("event_count", 0)) + 1
        self._records[target_key] = activated
        return deepcopy(activated)

    def rollback_atomic(
        self,
        *,
        tenant_id: str,
        target_index_id: str,
        target_index_version: int,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_key = self._key(tenant_id, target_index_id, target_index_version)
        target = self._records.get(target_key)
        if target is None or target["status"] not in {"retired", "shadow"}:
            raise RuntimeError("Rollback target must be a retired or shadow index.")
        for key, record in list(self._records.items()):
            if key[0] == tenant_id and record["status"] == "active":
                current = deepcopy(record)
                current["status"] = "retired"
                current["latest_evidence"] = {
                    "reason_code": "retired_by_atomic_rollback",
                    "rollback_target": target["manifest_fingerprint"],
                }
                current["event_count"] = int(current.get("event_count", 0)) + 1
                self._records[key] = current
        restored = deepcopy(target)
        restored["status"] = "active"
        restored["latest_evidence"] = deepcopy(dict(evidence))
        restored["event_count"] = int(restored.get("event_count", 0)) + 1
        self._records[target_key] = restored
        return deepcopy(restored)

    @staticmethod
    def _key(tenant_id: str, index_id: str, index_version: int) -> tuple[str, str, int]:
        return (str(tenant_id), str(index_id), int(index_version))


class JsonModule2IndexLifecycleRepository(InMemoryModule2IndexLifecycleRepository):
    """Atomic offline ledger used for engineering and operator rehearsal."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._load()

    def create(self, record: Mapping[str, Any]) -> dict[str, Any]:
        result = super().create(record)
        self._save()
        return result

    def transition(self, **kwargs: Any) -> dict[str, Any]:
        result = super().transition(**kwargs)
        self._save()
        return result

    def activate_atomic(self, **kwargs: Any) -> dict[str, Any]:
        result = super().activate_atomic(**kwargs)
        self._save()
        return result

    def rollback_atomic(self, **kwargs: Any) -> dict[str, Any]:
        result = super().rollback_atomic(**kwargs)
        self._save()
        return result

    def _load(self) -> None:
        if not self._path.exists():
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        for record in payload.get("records") or ():
            key = self._key(record["tenant_id"], record["index_id"], record["index_version"])
            self._records[key] = dict(record)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract_version": "module2-index-lifecycle-ledger-v1",
            "records": sorted(
                (deepcopy(value) for value in self._records.values()),
                key=lambda value: (
                    value["tenant_id"],
                    value["index_id"],
                    value["index_version"],
                ),
            ),
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
        }
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, self._path)


class PostgresModule2IndexLifecycleRepository:
    """Transactional repository for migration 0011.

    Activation and rollback are atomic and tenant-scoped. Runtime callers still
    need an external release gate; this repository never decides eligibility.
    """

    def __init__(self, *, engine: Engine | None = None, database_url: str | None = None) -> None:
        self._engine_override = engine
        self._database_url = str(database_url or "").strip()

    def create(self, record: Mapping[str, Any]) -> dict[str, Any]:
        engine = self._engine()
        dispose = self._engine_override is None
        try:
            with engine.begin() as connection:
                self._set_tenant(connection, record["tenant_id"])
                self._assert_schema(connection)
                connection.execute(
                    text(
                        """
                        INSERT INTO audience_retrieval_indexes (
                            tenant_id, index_id, index_version,
                            manifest_fingerprint, manifest, status,
                            event_count, latest_evidence
                        ) VALUES (
                            :tenant_id, :index_id, :index_version,
                            :manifest_fingerprint,
                            CAST(:manifest AS JSONB), :status, 0,
                            '{}'::jsonb
                        )
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "tenant_id": record["tenant_id"],
                        "index_id": record["index_id"],
                        "index_version": record["index_version"],
                        "manifest_fingerprint": record["manifest_fingerprint"],
                        "manifest": json.dumps(record["manifest"], sort_keys=True),
                        "status": record["status"],
                    },
                )
                return self._fetch(connection, record["tenant_id"], record["index_id"], record["index_version"])
        finally:
            if dispose:
                engine.dispose()

    def get(self, *, tenant_id: str, index_id: str, index_version: int) -> dict[str, Any] | None:
        engine = self._engine()
        dispose = self._engine_override is None
        try:
            with engine.connect() as connection:
                self._set_tenant(connection, tenant_id)
                self._assert_schema(connection)
                return self._fetch(connection, tenant_id, index_id, index_version, required=False)
        finally:
            if dispose:
                engine.dispose()

    def transition(self, *, tenant_id: str, index_id: str, index_version: int, expected_status: str, next_status: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        engine = self._engine()
        dispose = self._engine_override is None
        try:
            with engine.begin() as connection:
                self._set_tenant(connection, tenant_id)
                self._assert_schema(connection)
                row = connection.execute(
                    text(
                        """
                        UPDATE audience_retrieval_indexes
                        SET status = :next_status,
                            latest_evidence = CAST(:evidence AS JSONB),
                            event_count = event_count + 1,
                            updated_at = now()
                        WHERE tenant_id = :tenant_id
                          AND index_id = :index_id
                          AND index_version = :index_version
                          AND status = :expected_status
                        RETURNING *
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "index_id": index_id,
                        "index_version": int(index_version),
                        "expected_status": expected_status,
                        "next_status": next_status,
                        "evidence": json.dumps(dict(evidence), sort_keys=True),
                    },
                ).mappings().first()
                if row is None:
                    raise RuntimeError("Index state transition was rejected.")
                self._insert_event(connection, dict(row), expected_status, next_status, evidence)
                return self._row(dict(row))
        finally:
            if dispose:
                engine.dispose()

    def activate_atomic(self, *, tenant_id: str, index_id: str, index_version: int, evidence: Mapping[str, Any]) -> dict[str, Any]:
        return self._atomic_switch(
            tenant_id=tenant_id,
            target_index_id=index_id,
            target_index_version=index_version,
            allowed_target_statuses=("shadow",),
            evidence=evidence,
            reason="activation",
        )

    def rollback_atomic(self, *, tenant_id: str, target_index_id: str, target_index_version: int, evidence: Mapping[str, Any]) -> dict[str, Any]:
        return self._atomic_switch(
            tenant_id=tenant_id,
            target_index_id=target_index_id,
            target_index_version=target_index_version,
            allowed_target_statuses=("retired", "shadow"),
            evidence=evidence,
            reason="rollback",
        )

    def _atomic_switch(self, *, tenant_id: str, target_index_id: str, target_index_version: int, allowed_target_statuses: tuple[str, ...], evidence: Mapping[str, Any], reason: str) -> dict[str, Any]:
        engine = self._engine()
        dispose = self._engine_override is None
        try:
            with engine.begin() as connection:
                self._set_tenant(connection, tenant_id)
                self._assert_schema(connection)
                target = self._fetch(connection, tenant_id, target_index_id, target_index_version)
                if target["status"] not in allowed_target_statuses:
                    raise RuntimeError("Index target is not eligible for atomic switch.")
                connection.execute(
                    text(
                        """
                        UPDATE audience_retrieval_indexes
                        SET status = 'retired',
                            latest_evidence = jsonb_build_object(
                                'reason_code', :retire_reason,
                                'replacement_manifest_fingerprint', :replacement
                            ),
                            event_count = event_count + 1,
                            retired_at = now(), updated_at = now()
                        WHERE tenant_id = :tenant_id AND status = 'active'
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "retire_reason": f"retired_by_atomic_{reason}",
                        "replacement": target["manifest_fingerprint"],
                    },
                )
                row = connection.execute(
                    text(
                        """
                        UPDATE audience_retrieval_indexes
                        SET status = 'active',
                            latest_evidence = CAST(:evidence AS JSONB),
                            event_count = event_count + 1,
                            activated_at = now(), retired_at = NULL,
                            updated_at = now()
                        WHERE tenant_id = :tenant_id
                          AND index_id = :index_id
                          AND index_version = :index_version
                        RETURNING *
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "index_id": target_index_id,
                        "index_version": int(target_index_version),
                        "evidence": json.dumps(dict(evidence), sort_keys=True),
                    },
                ).mappings().one()
                self._insert_event(connection, dict(row), target["status"], "active", evidence)
                return self._row(dict(row))
        finally:
            if dispose:
                engine.dispose()

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        url = self._database_url or os.getenv("AUDIENCE_FEATURE_WRITER_DATABASE_URL", "")
        if not url:
            raise RuntimeError("AUDIENCE_FEATURE_WRITER_DATABASE_URL is required.")
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        return create_engine(url, pool_pre_ping=True)

    @staticmethod
    def _set_tenant(connection: Any, tenant_id: str) -> None:
        connection.execute(text("SELECT set_config('app.tenant_id', :tenant_id, true)"), {"tenant_id": tenant_id})

    @staticmethod
    def _assert_schema(connection: Any) -> None:
        ready = connection.execute(
            text("SELECT to_regclass('public.audience_retrieval_indexes') IS NOT NULL")
        ).scalar()
        if not ready:
            raise RuntimeError("Module 2 lifecycle migration 0011 is not ready.")

    @staticmethod
    def _fetch(connection: Any, tenant_id: str, index_id: str, index_version: int, required: bool = True) -> dict[str, Any] | None:
        row = connection.execute(
            text(
                """
                SELECT * FROM audience_retrieval_indexes
                WHERE tenant_id = :tenant_id
                  AND index_id = :index_id
                  AND index_version = :index_version
                """
            ),
            {"tenant_id": tenant_id, "index_id": index_id, "index_version": int(index_version)},
        ).mappings().first()
        if row is None and required:
            raise RuntimeError("Index record was not found.")
        return PostgresModule2IndexLifecycleRepository._row(dict(row)) if row is not None else None

    @staticmethod
    def _row(row: MutableMapping[str, Any]) -> dict[str, Any]:
        manifest = dict(row.get("manifest") or {})
        return {
            "tenant_id": row["tenant_id"],
            "index_id": row["index_id"],
            "index_version": int(row["index_version"]),
            "manifest_fingerprint": row["manifest_fingerprint"],
            "manifest": manifest,
            "status": row["status"],
            "event_count": int(row.get("event_count") or 0),
            "latest_evidence": dict(row.get("latest_evidence") or {}),
        }

    @staticmethod
    def _insert_event(connection: Any, row: Mapping[str, Any], previous_status: str, next_status: str, evidence: Mapping[str, Any]) -> None:
        connection.execute(
            text(
                """
                INSERT INTO audience_retrieval_index_events (
                    tenant_id, index_id, index_version, event_fingerprint,
                    previous_status, next_status, evidence
                ) VALUES (
                    :tenant_id, :index_id, :index_version,
                    :event_fingerprint, :previous_status, :next_status,
                    CAST(:evidence AS JSONB)
                )
                """
            ),
            {
                "tenant_id": row["tenant_id"],
                "index_id": row["index_id"],
                "index_version": row["index_version"],
                "event_fingerprint": stable_digest(
                    {
                        "manifest_fingerprint": row["manifest_fingerprint"],
                        "previous_status": previous_status,
                        "next_status": next_status,
                        "evidence": dict(evidence),
                        "event_count": int(row.get("event_count") or 0),
                    }
                ),
                "previous_status": previous_status,
                "next_status": next_status,
                "evidence": json.dumps(dict(evidence), sort_keys=True),
            },
        )


class ProductionModule2IndexLifecycleService:
    def __init__(self, *, repository: Module2IndexLifecycleRepository) -> None:
        self._repository = repository

    def register_candidate(self, manifest: Module2IndexManifest) -> dict[str, Any]:
        record = {
            "tenant_id": manifest.tenant_id,
            "index_id": manifest.index_id,
            "index_version": manifest.index_version,
            "manifest_fingerprint": manifest.fingerprint,
            "manifest": manifest.to_safe_dict(),
            "status": "candidate",
            "event_count": 0,
            "latest_evidence": {},
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
        }
        created = self._repository.create(record)
        if created.get("manifest_fingerprint") != manifest.fingerprint:
            raise RuntimeError("Index identity exists with a conflicting manifest.")
        return created

    def mark_building(self, *, manifest: Module2IndexManifest, worker_id: str) -> dict[str, Any]:
        return self._transition(
            manifest=manifest,
            expected="candidate",
            next_status="building",
            evidence={"worker_id": str(worker_id), "reason_code": "build_claimed"},
        )

    def mark_built(self, *, manifest: Module2IndexManifest, observed_document_count: int, artifact_checksum: str) -> dict[str, Any]:
        import re

        if int(observed_document_count) != manifest.expected_document_count:
            raise RuntimeError("Built index document count does not match manifest.")
        checksum = str(artifact_checksum or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("artifact_checksum must be a lowercase SHA-256 digest.")
        evidence = {
            "reason_code": "index_build_completed",
            "observed_document_count": int(observed_document_count),
            "artifact_checksum": checksum,
        }
        return self._transition(manifest=manifest, expected="building", next_status="built", evidence=evidence)

    def record_validation(self, *, manifest: Module2IndexManifest, validation: Module2IndexValidationEvidence) -> dict[str, Any]:
        if validation.manifest_fingerprint != manifest.fingerprint:
            raise ValueError("Index validation does not match manifest.")
        if validation.observed_document_count != manifest.expected_document_count:
            raise RuntimeError("Validated document count does not match manifest.")
        next_status = "validated" if validation.passed else "failed"
        return self._transition(
            manifest=manifest,
            expected="built",
            next_status=next_status,
            evidence=validation.to_safe_dict(),
        )

    def promote_to_shadow(self, *, manifest: Module2IndexManifest, approved_by: str, certification_report: Mapping[str, Any]) -> dict[str, Any]:
        certification_report = validate_module2_certification_report(
            certification_report
        )
        if not bool(certification_report.get("module2_evidence_ready")):
            raise RuntimeError("Shadow promotion requires complete Module 2 evidence.")
        evidence = {
            "reason_code": "approved_for_shadow_serving",
            "approved_by": str(approved_by),
            "certification_report_fingerprint": certification_report.get("report_fingerprint"),
            "production_routing_enabled": False,
        }
        return self._transition(manifest=manifest, expected="validated", next_status="shadow", evidence=evidence)

    def promote_to_active(self, *, manifest: Module2IndexManifest, approved_by: str, certification_report: Mapping[str, Any], shadow_report: Mapping[str, Any], production_routing_enabled: bool = False) -> dict[str, Any]:
        certification_report = validate_module2_certification_report(
            certification_report
        )
        shadow_report = validate_module2_shadow_readiness_report(shadow_report)
        if not production_routing_enabled:
            raise RuntimeError("Explicit production routing approval is required.")
        if manifest.data_use_mode != "production":
            raise RuntimeError("Only a production data-use index may be activated.")
        if not bool(certification_report.get("production_certification_ready")):
            raise RuntimeError("Production certification is not ready.")
        if not bool(shadow_report.get("shadow_release_ready")):
            raise RuntimeError("Shadow release evidence is not ready.")
        evidence = {
            "reason_code": "atomic_production_activation_approved",
            "approved_by": str(approved_by),
            "certification_report_fingerprint": certification_report.get("report_fingerprint"),
            "shadow_report_fingerprint": shadow_report.get("report_fingerprint"),
            "production_routing_enabled": True,
            "downstream_export_enabled": False,
        }
        return self._repository.activate_atomic(
            tenant_id=manifest.tenant_id,
            index_id=manifest.index_id,
            index_version=manifest.index_version,
            evidence=evidence,
        )

    def rollback(self, *, target_manifest: Module2IndexManifest, approved_by: str, incident_id: str, production_routing_enabled: bool = False) -> dict[str, Any]:
        if not production_routing_enabled:
            raise RuntimeError("Explicit production rollback approval is required.")
        evidence = {
            "reason_code": "atomic_index_rollback_approved",
            "approved_by": str(approved_by),
            "incident_id": str(incident_id),
            "production_routing_enabled": True,
            "downstream_export_enabled": False,
        }
        return self._repository.rollback_atomic(
            tenant_id=target_manifest.tenant_id,
            target_index_id=target_manifest.index_id,
            target_index_version=target_manifest.index_version,
            evidence=evidence,
        )

    def plan_incremental_update(self, *, base_manifest: Module2IndexManifest, delta_source_fingerprint: str, insert_count: int, update_count: int, delete_count: int) -> dict[str, Any]:
        import re

        delta_fingerprint = str(delta_source_fingerprint or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", delta_fingerprint):
            raise ValueError("delta_source_fingerprint must be a lowercase SHA-256 digest.")
        for value in (insert_count, update_count, delete_count):
            if int(value) < 0:
                raise ValueError("Incremental index counts cannot be negative.")
        plan = {
            "contract_version": "module2-index-incremental-update-plan-v1",
            "base_manifest_fingerprint": base_manifest.fingerprint,
            "delta_source_fingerprint": delta_fingerprint,
            "insert_count": int(insert_count),
            "update_count": int(update_count),
            "delete_count": int(delete_count),
            "requires_new_immutable_index_version": True,
            "in_place_mutation_allowed": False,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
        }
        plan["idempotency_key"] = stable_digest(plan)
        return plan

    def _transition(self, *, manifest: Module2IndexManifest, expected: str, next_status: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        current = self._repository.get(
            tenant_id=manifest.tenant_id,
            index_id=manifest.index_id,
            index_version=manifest.index_version,
        )
        if current is None:
            raise RuntimeError("Index record was not found.")
        if current["manifest_fingerprint"] != manifest.fingerprint:
            raise RuntimeError("Index manifest fingerprint conflict.")
        return self._repository.transition(
            tenant_id=manifest.tenant_id,
            index_id=manifest.index_id,
            index_version=manifest.index_version,
            expected_status=expected,
            next_status=next_status,
            evidence=dict(evidence),
        )
