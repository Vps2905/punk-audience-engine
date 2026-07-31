from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.models.provider_scale_acceptance_contracts import (
    ProviderScaleAcceptanceEvidence,
    ProviderScaleAcceptancePolicy,
)


class ProviderScaleAcceptanceService:
    """Fail-closed evidence gate for the billion-event production claim."""

    def __init__(
        self,
        policy: ProviderScaleAcceptancePolicy | None = None,
        *,
        database_url: Optional[str] = None,
    ) -> None:
        self._policy = policy or ProviderScaleAcceptancePolicy()
        self._explicit_database_url = database_url

    def evaluate(
        self,
        evidence: ProviderScaleAcceptanceEvidence,
    ) -> Dict[str, Any]:
        policy = self._policy
        blockers: list[str] = []
        calculated_average = (
            evidence.total_events / evidence.duration_seconds
        )
        average_tolerance = max(1.0, calculated_average * 0.01)
        if abs(
            evidence.average_events_per_second - calculated_average
        ) > average_tolerance:
            blockers.append("average_throughput_measurement_inconsistent")
        if evidence.peak_events_per_second < evidence.average_events_per_second:
            blockers.append("peak_throughput_below_average")
        if evidence.total_events < policy.target_events_per_day:
            blockers.append("target_event_volume_not_reached")
        if (
            evidence.average_events_per_second
            < policy.minimum_average_events_per_second
        ):
            blockers.append("average_throughput_below_target")
        if (
            evidence.peak_events_per_second
            < policy.minimum_peak_events_per_second
        ):
            blockers.append("peak_throughput_below_target")
        if evidence.queue_age_p99_seconds > policy.maximum_queue_age_p99_seconds:
            blockers.append("queue_age_slo_exceeded")
        if (
            evidence.end_to_end_p99_seconds
            > policy.maximum_end_to_end_p99_seconds
        ):
            blockers.append("end_to_end_slo_exceeded")
        if (
            evidence.rights_propagation_seconds
            > policy.maximum_rights_propagation_seconds
        ):
            blockers.append("data_rights_slo_exceeded")
        normalized_cost = (
            evidence.cost_usd
            * policy.target_events_per_day
            / max(1, evidence.total_events)
        )
        if normalized_cost > policy.maximum_cost_per_billion_usd:
            blockers.append("cost_per_billion_exceeded")
        zero_tolerance = {
            "duplicate_side_effect_count": evidence.duplicate_side_effect_count,
            "raw_identifier_output_count": evidence.raw_identifier_output_count,
            "privacy_partition_violation_count": (
                evidence.privacy_partition_violation_count
            ),
            "unrecovered_failure_count": evidence.unrecovered_failure_count,
            "canonical_checksum_failure_count": (
                evidence.canonical_checksum_failure_count
            ),
            "stale_data_activated_count": evidence.stale_data_activated_count,
        }
        for field_name, value in zero_tolerance.items():
            if int(value) != 0:
                blockers.append(field_name)
        required_proofs = {
            "source_replay_not_verified": evidence.source_replay_verified,
            "worker_restart_not_verified": evidence.worker_restart_verified,
            "backup_restore_not_verified": evidence.backup_restore_verified,
        }
        for reason, verified in required_proofs.items():
            if not verified:
                blockers.append(reason)

        return {
            "status": (
                "production_scale_gate_passed"
                if not blockers
                else "blocked_production_scale_gate"
            ),
            "production_scale_certified": not blockers,
            "blockers": blockers,
            "evidence_id": evidence.evidence_id,
            "environment": evidence.environment,
            "measured_events": int(evidence.total_events),
            "measured_average_events_per_second": float(
                evidence.average_events_per_second
            ),
            "measured_peak_events_per_second": float(
                evidence.peak_events_per_second
            ),
            "normalized_cost_per_billion_usd": round(normalized_cost, 4),
            "raw_identifiers_allowed": False,
            "activation_or_export_performed": False,
        }

    def record(
        self,
        *,
        tenant_id: str,
        evidence: ProviderScaleAcceptanceEvidence,
    ) -> Dict[str, Any]:
        """Persist immutable, tenant-scoped measured acceptance evidence."""
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            raise ValueError("tenant_id is required")
        report = self.evaluate(evidence)
        encoded = json.dumps(
            report, sort_keys=True, separators=(",", ":")
        )
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, tenant)
            existing = conn.execute(
                text(
                    """
                    SELECT report_fingerprint, report
                    FROM provider_scale_acceptance_runs
                    WHERE evidence_id = :evidence_id
                      AND tenant_id = :tenant_id
                    """
                ),
                {
                    "evidence_id": evidence.evidence_id,
                    "tenant_id": tenant,
                },
            ).mappings().first()
            if existing:
                if existing["report_fingerprint"] != fingerprint:
                    raise ValueError(
                        "Scale evidence ID cannot be reused with changed evidence"
                    )
                stored = existing["report"]
                if isinstance(stored, str):
                    stored = json.loads(stored)
                return {**dict(stored), "recorded": True, "replayed": True}
            conn.execute(
                text(
                    """
                    INSERT INTO provider_scale_acceptance_runs (
                        evidence_id, tenant_id, environment,
                        report_fingerprint, status, measured_events,
                        blockers, report, created_at
                    ) VALUES (
                        :evidence_id, :tenant_id, :environment,
                        :report_fingerprint, :status, :measured_events,
                        :blockers, :report, :created_at
                    )
                    """
                ),
                {
                    "evidence_id": evidence.evidence_id,
                    "tenant_id": tenant,
                    "environment": evidence.environment,
                    "report_fingerprint": fingerprint,
                    "status": report["status"],
                    "measured_events": report["measured_events"],
                    "blockers": json.dumps(report["blockers"]),
                    "report": encoded,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        return {**report, "recorded": True, "replayed": False}

    def latest(self, *, tenant_id: str) -> Dict[str, Any]:
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            raise ValueError("tenant_id is required")
        engine = self._engine()
        with engine.begin() as conn:
            self._ensure_table(conn, engine.dialect.name)
            self._set_tenant(conn, engine.dialect.name, tenant)
            row = conn.execute(
                text(
                    """
                    SELECT report, created_at
                    FROM provider_scale_acceptance_runs
                    WHERE tenant_id = :tenant_id
                    ORDER BY created_at DESC, evidence_id DESC
                    LIMIT 1
                    """
                ),
                {"tenant_id": tenant},
            ).mappings().first()
        if not row:
            return {
                "status": "scale_evidence_not_recorded",
                "production_scale_certified": False,
                "blockers": ["measured_scale_evidence_missing"],
                "activation_or_export_performed": False,
            }
        report = row["report"]
        if isinstance(report, str):
            report = json.loads(report)
        return {**dict(report), "recorded_at": str(row["created_at"])}

    def _engine(self) -> Any:
        database_url = (
            self._explicit_database_url
            or os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        )
        if not database_url:
            raise RuntimeError("Scale evidence requires provider PostgreSQL")
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)

    def _set_tenant(self, conn: Any, dialect: str, tenant_id: str) -> None:
        if dialect == "postgresql":
            conn.execute(
                text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                {"tenant_id": tenant_id},
            )

    def _ensure_table(self, conn: Any, dialect: str) -> None:
        # Production PostgreSQL schema is installed by operator migrations.
        if dialect == "postgresql":
            return
        timestamp = "TIMESTAMPTZ" if dialect == "postgresql" else "TEXT"
        json_type = "JSONB" if dialect == "postgresql" else "TEXT"
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS provider_scale_acceptance_runs (
                    evidence_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    report_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    measured_events BIGINT NOT NULL,
                    blockers {json_type} NOT NULL,
                    report {json_type} NOT NULL,
                    created_at {timestamp} NOT NULL,
                    PRIMARY KEY (tenant_id, evidence_id),
                    UNIQUE (tenant_id, report_fingerprint)
                )
                """
            )
        )
