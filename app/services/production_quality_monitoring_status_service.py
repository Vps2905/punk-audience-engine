from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from app.services.production_quality_monitoring_service import (
    ProductionQualityAlertPlanningService,
    ProductionQualityDriftService,
    ProductionQualitySnapshotService,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionQualityMonitoringStatusService:
    """Report aggregate quality-monitoring readiness without metric disclosure."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        snapshot = self._validated(
            "QUALITY_MONITORING_EVIDENCE_PATH",
            ProductionQualitySnapshotService().validate_report,
        )
        drift = self._validated(
            "QUALITY_DRIFT_EVIDENCE_PATH",
            ProductionQualityDriftService().validate_report,
        )
        alerts = self._validated(
            "QUALITY_ALERT_PLAN_EVIDENCE_PATH",
            ProductionQualityAlertPlanningService().validate_report,
        )
        snapshot_valid = snapshot is not None
        drift_valid = bool(
            snapshot_valid
            and drift
            and drift.get("request", {}).get("tenant_id")
            == snapshot.get("request", {}).get("tenant_id")
            and drift.get("request", {}).get("current_snapshot_fingerprint")
            == snapshot.get("quality_snapshot_fingerprint")
        )
        alert_valid = bool(
            drift_valid
            and alerts
            and alerts.get("tenant_id") == drift.get("request", {}).get("tenant_id")
            and alerts.get("source_quality_drift_report_fingerprint")
            == drift.get("quality_drift_report_fingerprint")
        )
        monitoring_enabled = self._flag("QUALITY_MONITORING_ENABLED")
        alerting_enabled = self._flag("QUALITY_ALERTING_ENABLED")
        remediation_enabled = self._flag("QUALITY_AUTOMATIC_REMEDIATION_ENABLED")
        routing_enabled = self._flag("QUALITY_PRODUCTION_ROUTING_ENABLED")
        if remediation_enabled or routing_enabled:
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif monitoring_enabled and not snapshot_valid:
            readiness = "unsafe_configuration_quality_evidence_missing"
        elif alerting_enabled and not alert_valid:
            readiness = "unsafe_configuration_alert_evidence_missing"
        elif alert_valid:
            readiness = "quality_monitoring_engineering_evidence_ready"
        elif drift_valid:
            readiness = "quality_drift_engineering_evidence_ready"
        elif snapshot_valid:
            readiness = "quality_snapshot_engineering_evidence_ready"
        else:
            readiness = "quality_monitoring_evidence_pending"
        return {
            "module": "production_aggregate_quality_monitoring",
            "status": readiness,
            "components": {
                "aggregate_quality_snapshot": snapshot_valid,
                "aggregate_quality_drift": drift_valid,
                "review_only_alert_planning": alert_valid,
            },
            "quality_monitoring_engineering_evidence_ready": alert_valid,
            "live_production_certified": False,
            "evidence_presence": {
                "quality_snapshot": snapshot is not None,
                "quality_drift": drift is not None,
                "quality_alert_plan": alerts is not None,
            },
            "feature_flags": {
                "monitoring_enabled": monitoring_enabled,
                "alerting_enabled": alerting_enabled,
                "automatic_remediation_enabled": remediation_enabled,
                "production_routing_enabled": routing_enabled,
            },
            "safety": {
                "aggregate_metrics_only": True,
                "raw_identifiers_exposed": False,
                "individual_records_read": False,
                "automatic_remediation_performed": False,
                "external_alert_dispatched": False,
                "manual_approval_required": True,
                "production_routing_enabled": False,
                "activation_or_export_performed": False,
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
