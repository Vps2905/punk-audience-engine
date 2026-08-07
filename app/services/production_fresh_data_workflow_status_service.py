from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.services.production_fresh_data_workflow_readiness_service import (
    ProductionFreshDataWorkflowReadinessService,
)


class ProductionFreshDataWorkflowStatusService:
    """Fail-closed runtime status for the Module 1 -> 2 -> 3 bridge."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        readiness_probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._environment = dict(environment or os.environ)
        self._readiness_probe = readiness_probe or (
            lambda: ProductionFreshDataWorkflowReadinessService(
                environment=self._environment
            ).inspect()
        )

    def status(self) -> dict[str, Any]:
        enabled = self._flag("FRESH_DATA_WORKFLOW_ENABLED")
        automatic_trigger = self._flag(
            "FRESH_DATA_WORKFLOW_AUTOMATIC_TRIGGER_ENABLED"
        )
        activation = self._flag("AUDIENCE_ACTIVATION_ENABLED")
        export = self._flag("AUDIENCE_DOWNSTREAM_EXPORT_ENABLED")
        migration_present = Path(
            "migrations/0014_production_fresh_data_workflows.sql"
        ).is_file()
        required_configuration = {
            "provider_database": self._configured(
                "PROVIDER_INGESTION_DATABASE_URL"
            ),
            "feature_reader_database": self._configured(
                "AUDIENCE_FEATURE_DATABASE_URL"
            ),
            "feature_writer_database": self._configured(
                "AUDIENCE_FEATURE_WRITER_DATABASE_URL"
            ),
        }
        release_unsafe = activation or export
        config_ready = all(required_configuration.values())
        readiness = self._not_checked_readiness()

        if release_unsafe:
            status = "unsafe_release_configuration_blocked"
        elif not enabled:
            status = "fresh_data_workflow_disabled"
        elif not config_ready:
            status = "fresh_data_workflow_configuration_incomplete"
        elif not migration_present:
            status = "fresh_data_workflow_migration_file_missing"
        else:
            readiness = self._safe_probe()
            status = (
                "fresh_data_workflow_control_plane_ready"
                if readiness.get("ready")
                else "fresh_data_workflow_database_not_ready"
            )

        remaining_gates = self._remaining_gates(
            enabled=enabled,
            automatic_trigger=automatic_trigger,
            migration_present=migration_present,
            config_ready=config_ready,
            readiness=readiness,
        )
        return {
            "status": status,
            "enabled": enabled,
            "automatic_trigger_enabled": automatic_trigger,
            "migration_present": migration_present,
            "configuration": required_configuration,
            "database_readiness": readiness,
            "components": {
                "module1_completed_ingestion_validation": True,
                "canonical_object_integrity_validation": True,
                "durable_worker_leasing": True,
                "module2_feature_build_bridge": True,
                "module3_candidate_generation_bridge": True,
                "module3_overlap_review_bridge": True,
                "append_only_transition_audit": True,
                "completed_ingestion_trigger_factory": True,
                "bounded_trigger_batching": True,
            },
            "safety": {
                "approval_required": True,
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "candidate_database_write_performed": False,
                "lookalike_generation_performed": False,
                "production_routing_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "secret_values_returned": False,
            },
            "remaining_external_gates": remaining_gates,
        }

    def _safe_probe(self) -> dict[str, Any]:
        try:
            report = dict(self._readiness_probe())
        except Exception:
            return {
                "ready": False,
                "read_only": True,
                "credentials_exposed": False,
                "components": {},
                "blockers": ["readiness_probe_failed"],
                "error_code": "readiness_probe_failed",
            }
        report.pop("database_url", None)
        report.pop("credentials", None)
        report["credentials_exposed"] = False
        return report

    def _remaining_gates(
        self,
        *,
        enabled: bool,
        automatic_trigger: bool,
        migration_present: bool,
        config_ready: bool,
        readiness: Mapping[str, Any],
    ) -> list[str]:
        gates: list[str] = []
        if not enabled:
            gates.append("enable the workflow after deployment review")
        if not config_ready:
            gates.append("configure all provider and feature database boundaries")
        if not migration_present:
            gates.append("ship migration 0014 with the release artifact")
        for blocker in readiness.get("blockers") or []:
            gates.append(f"resolve database readiness blocker: {blocker}")
        if not automatic_trigger:
            gates.append("configure and validate the production trigger/queue worker")
        gates.extend(
            [
                "validate with a fresh provider delivery in staging",
                "complete load, recovery, privacy, and security certification",
            ]
        )
        return gates

    def _not_checked_readiness(self) -> dict[str, Any]:
        return {
            "ready": False,
            "checked": False,
            "read_only": True,
            "credentials_exposed": False,
            "components": {},
            "blockers": [],
        }

    def _configured(self, key: str) -> bool:
        return bool(str(self._environment.get(key) or "").strip())

    def _flag(self, key: str) -> bool:
        return str(self._environment.get(key) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
