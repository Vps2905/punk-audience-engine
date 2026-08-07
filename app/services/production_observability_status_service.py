from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from app.services.production_observability_service import (
    ProductionIncidentReviewPlanningService,
    ProductionObservabilitySnapshotService,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionObservabilityStatusService:
    """Report SLO/telemetry engineering readiness without signal disclosure."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        snapshot = self._validated(
            "OBSERVABILITY_SNAPSHOT_EVIDENCE_PATH",
            ProductionObservabilitySnapshotService().validate_report,
        )
        incidents = self._validated(
            "OBSERVABILITY_INCIDENT_PLAN_EVIDENCE_PATH",
            ProductionIncidentReviewPlanningService().validate_report,
        )
        snapshot_valid = snapshot is not None
        incident_valid = bool(
            snapshot_valid
            and incidents
            and incidents.get("tenant_id")
            == snapshot.get("request", {}).get("tenant_id")
            and incidents.get("source_observability_snapshot_fingerprint")
            == snapshot.get("observability_snapshot_fingerprint")
        )
        monitoring_enabled = self._flag("OBSERVABILITY_MONITORING_ENABLED")
        alerting_enabled = self._flag("OBSERVABILITY_ALERTING_ENABLED")
        remediation_enabled = self._flag(
            "OBSERVABILITY_AUTOMATIC_REMEDIATION_ENABLED"
        )
        routing_enabled = self._flag("OBSERVABILITY_PRODUCTION_ROUTING_ENABLED")
        if remediation_enabled or routing_enabled:
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif monitoring_enabled and not snapshot_valid:
            readiness = "unsafe_configuration_observability_evidence_missing"
        elif alerting_enabled and not incident_valid:
            readiness = "unsafe_configuration_incident_plan_evidence_missing"
        elif incident_valid:
            readiness = "observability_engineering_evidence_ready"
        elif snapshot_valid:
            readiness = "observability_snapshot_engineering_evidence_ready"
        else:
            readiness = "observability_evidence_pending"
        return {
            "module": "production_observability_and_slo_governance",
            "status": readiness,
            "components": {
                "aggregate_telemetry_and_slo_snapshot": snapshot_valid,
                "review_only_incident_planning": incident_valid,
            },
            "observability_engineering_evidence_ready": incident_valid,
            "live_production_certified": False,
            "evidence_presence": {
                "observability_snapshot": snapshot is not None,
                "incident_review_plan": incidents is not None,
            },
            "feature_flags": {
                "monitoring_enabled": monitoring_enabled,
                "alerting_enabled": alerting_enabled,
                "automatic_remediation_enabled": remediation_enabled,
                "production_routing_enabled": routing_enabled,
            },
            "safety": {
                "aggregate_telemetry_only": True,
                "raw_identifiers_exposed": False,
                "prompt_content_exposed": False,
                "request_or_response_payload_exposed": False,
                "external_alert_dispatched": False,
                "automatic_remediation_performed": False,
                "incident_declared_automatically": False,
                "manual_approval_required": True,
                "production_routing_enabled": False,
            },
        }

    def _flag(self, key: str) -> bool:
        return _truthy(self._environment.get(key))

    def _validated(
        self,
        key: str,
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any] | None:
        value = self._read_json(key)
        if value is None:
            return None
        try:
            return validator(value)
        except (TypeError, ValueError):
            return None

    def _read_json(self, key: str) -> dict[str, Any] | None:
        raw = str(self._environment.get(key) or "").strip()
        if not raw or not Path(raw).is_file():
            return None
        try:
            value = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, Mapping) else None
