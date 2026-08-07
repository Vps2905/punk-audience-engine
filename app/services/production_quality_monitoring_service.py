from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)
from app.models.production_quality_monitoring_contracts import (
    QUALITY_ALERT_POLICY_VERSION,
    QUALITY_DRIFT_POLICY_VERSION,
    QUALITY_SNAPSHOT_POLICY_VERSION,
    AggregateQualityMetric,
    ProductionQualityDriftRequest,
    ProductionQualitySnapshotRequest,
)


_FALSE_SAFETY_FIELDS = (
    "raw_identifiers_read",
    "raw_identifiers_stored",
    "raw_identifiers_returned",
    "individual_records_read",
    "individual_behavior_inferred",
    "automatic_remediation_performed",
    "external_alert_dispatched",
    "production_routing_enabled",
    "activation_or_export_performed",
    "downstream_export_enabled",
)


def _safety() -> dict[str, Any]:
    return {
        "raw_identifiers_read": False,
        "raw_identifiers_stored": False,
        "raw_identifiers_returned": False,
        "individual_records_read": False,
        "individual_behavior_inferred": False,
        "aggregate_metrics_only": True,
        "automatic_remediation_performed": False,
        "external_alert_dispatched": False,
        "manual_approval_required": True,
        "monitoring_required": True,
        "production_routing_enabled": False,
        "activation_or_export_performed": False,
        "downstream_export_enabled": False,
    }


def _validate_safety(report: Mapping[str, Any], label: str) -> None:
    safety = report.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError(f"{label} safety evidence is required.")
    for field in _FALSE_SAFETY_FIELDS:
        if safety.get(field) is not False:
            raise ValueError(f"Unsafe {label} safety field: {field}.")
    if safety.get("aggregate_metrics_only") is not True:
        raise ValueError(f"{label} must remain aggregate-only.")
    if safety.get("manual_approval_required") is not True:
        raise ValueError(f"{label} manual approval must remain required.")
    if safety.get("monitoring_required") is not True:
        raise ValueError(f"{label} monitoring must remain required.")


class ProductionQualitySnapshotService:
    """Evaluate generic aggregate quality signals across production domains."""

    def build(
        self,
        *,
        request: ProductionQualitySnapshotRequest,
        metrics: Sequence[AggregateQualityMetric | Mapping[str, Any]],
    ) -> dict[str, Any]:
        observations = [
            value if isinstance(value, AggregateQualityMetric)
            else AggregateQualityMetric(**dict(value))
            for value in metrics
        ]
        if not observations:
            raise ValueError("At least one aggregate quality metric is required.")
        keys = [(value.domain, value.metric_name) for value in observations]
        if len(set(keys)) != len(keys):
            raise ValueError("Aggregate quality metric keys must be unique.")
        rows = [self._assess(value) for value in observations]
        rows.sort(key=lambda value: (value["domain"], value["metric_name"]))
        counts = Counter(value["health_status"] for value in rows)
        overall = (
            "critical" if counts["critical"]
            else "warning" if counts["warning"]
            else "healthy"
        )
        fingerprint = stable_fingerprint({
            "policy_version": QUALITY_SNAPSHOT_POLICY_VERSION,
            "request": request.to_record(),
            "metric_observations": rows,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": QUALITY_SNAPSHOT_POLICY_VERSION,
            "request": request.to_record(),
            "quality_snapshot_fingerprint": fingerprint,
            "metric_count": len(rows),
            "healthy_count": counts["healthy"],
            "warning_count": counts["warning"],
            "critical_count": counts["critical"],
            "overall_health": overall,
            "metric_observations": rows,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported quality snapshot status.")
        if payload.get("policy_version") != QUALITY_SNAPSHOT_POLICY_VERSION:
            raise ValueError("Unsupported quality snapshot policy version.")
        request = ProductionQualitySnapshotRequest(**dict(payload.get("request") or {}))
        rows = payload.get("metric_observations")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Quality metric observations are required.")
        rebuilt = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Quality metric observation is invalid.")
            metric = AggregateQualityMetric(**{
                key: row.get(key)
                for key in (
                    "metric_name",
                    "domain",
                    "observed_value",
                    "target_min",
                    "target_max",
                    "critical_min",
                    "critical_max",
                    "drift_warning_delta",
                    "drift_critical_delta",
                    "sample_count",
                    "source_evidence_fingerprint",
                )
            })
            expected = self._assess(metric)
            if dict(row) != expected:
                raise ValueError("Quality metric assessment is inconsistent.")
            rebuilt.append(expected)
        if rebuilt != sorted(
            rebuilt, key=lambda value: (value["domain"], value["metric_name"])
        ):
            raise ValueError("Quality metric observations must be ordered.")
        keys = [(value["domain"], value["metric_name"]) for value in rebuilt]
        if len(set(keys)) != len(keys):
            raise ValueError("Quality metric keys must be unique.")
        counts = Counter(value["health_status"] for value in rebuilt)
        expected_overall = (
            "critical" if counts["critical"]
            else "warning" if counts["warning"]
            else "healthy"
        )
        expected_counts = {
            "metric_count": len(rebuilt),
            "healthy_count": counts["healthy"],
            "warning_count": counts["warning"],
            "critical_count": counts["critical"],
        }
        for field, value in expected_counts.items():
            if int(payload.get(field) or 0) != value:
                raise ValueError(f"Quality snapshot {field} is inconsistent.")
        if payload.get("overall_health") != expected_overall:
            raise ValueError("Quality snapshot overall health is inconsistent.")
        expected_fingerprint = stable_fingerprint({
            "policy_version": QUALITY_SNAPSHOT_POLICY_VERSION,
            "request": request.to_record(),
            "metric_observations": rebuilt,
        })
        if payload.get("quality_snapshot_fingerprint") != expected_fingerprint:
            raise ValueError("Quality snapshot fingerprint mismatch.")
        _validate_safety(payload, "quality snapshot")
        return payload

    def _assess(self, metric: AggregateQualityMetric) -> dict[str, Any]:
        value = metric.observed_value
        if value < metric.critical_min or value > metric.critical_max:
            status = "critical"
            reason = "outside_critical_range"
        elif value < metric.target_min or value > metric.target_max:
            status = "warning"
            reason = "outside_target_range"
        else:
            status = "healthy"
            reason = "within_target_range"
        return {
            **metric.to_record(),
            "health_status": status,
            "reason_code": reason,
        }


class ProductionQualityDriftService:
    """Compare immutable aggregate snapshots without accessing source rows."""

    def analyze(
        self,
        *,
        request: ProductionQualityDriftRequest,
        baseline_snapshot: Mapping[str, Any],
        current_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        snapshots = ProductionQualitySnapshotService()
        baseline = snapshots.validate_report(baseline_snapshot)
        current = snapshots.validate_report(current_snapshot)
        self._validate_lineage(request, baseline, current)
        baseline_rows = self._by_key(baseline)
        current_rows = self._by_key(current)
        entries = [
            self._entry(key, baseline_rows.get(key), current_rows.get(key))
            for key in sorted(set(baseline_rows) | set(current_rows))
        ]
        counts = Counter(value["severity"] for value in entries)
        fingerprint = stable_fingerprint({
            "policy_version": QUALITY_DRIFT_POLICY_VERSION,
            "request": request.to_record(),
            "drift_entries": entries,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": QUALITY_DRIFT_POLICY_VERSION,
            "request": request.to_record(),
            "quality_drift_report_fingerprint": fingerprint,
            "metric_count": len(entries),
            "stable_count": counts["none"],
            "warning_count": counts["warning"],
            "critical_count": counts["critical"],
            "drift_detected": bool(counts["warning"] or counts["critical"]),
            "drift_entries": entries,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported quality drift status.")
        if payload.get("policy_version") != QUALITY_DRIFT_POLICY_VERSION:
            raise ValueError("Unsupported quality drift policy version.")
        request = ProductionQualityDriftRequest(**dict(payload.get("request") or {}))
        entries = payload.get("drift_entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Quality drift entries are required.")
        if any(
            not isinstance(value, Mapping)
            or value.get("change_type") not in {
                "new", "missing", "changed", "unchanged"
            }
            or value.get("severity") not in {"none", "warning", "critical"}
            or not value.get("reason_codes")
            for value in entries
        ):
            raise ValueError("Quality drift entry is invalid.")
        expected = stable_fingerprint({
            "policy_version": QUALITY_DRIFT_POLICY_VERSION,
            "request": request.to_record(),
            "drift_entries": entries,
        })
        if payload.get("quality_drift_report_fingerprint") != expected:
            raise ValueError("Quality drift fingerprint mismatch.")
        counts = Counter(value["severity"] for value in entries)
        expected_counts = {
            "metric_count": len(entries),
            "stable_count": counts["none"],
            "warning_count": counts["warning"],
            "critical_count": counts["critical"],
        }
        for field, value in expected_counts.items():
            if int(payload.get(field) or 0) != value:
                raise ValueError(f"Quality drift {field} is inconsistent.")
        expected_detected = bool(counts["warning"] or counts["critical"])
        if bool(payload.get("drift_detected")) != expected_detected:
            raise ValueError("Quality drift result is inconsistent.")
        _validate_safety(payload, "quality drift")
        return payload

    def _validate_lineage(self, request, baseline, current) -> None:
        for label, report, fingerprint in (
            ("baseline", baseline, request.baseline_snapshot_fingerprint),
            ("current", current, request.current_snapshot_fingerprint),
        ):
            if report.get("quality_snapshot_fingerprint") != fingerprint:
                raise ValueError(f"Quality {label} fingerprint mismatch.")
            if report.get("request", {}).get("tenant_id") != request.tenant_id:
                raise ValueError(f"Quality {label} tenant mismatch.")

    def _by_key(self, report):
        return {
            (value["domain"], value["metric_name"]): dict(value)
            for value in report.get("metric_observations") or []
        }

    def _entry(self, key, baseline, current):
        domain, metric_name = key
        if baseline is None:
            return {
                "domain": domain,
                "metric_name": metric_name,
                "change_type": "new",
                "severity": "warning",
                "baseline_value": None,
                "current_value": current["observed_value"],
                "absolute_delta": None,
                "reason_codes": ["new_metric_requires_baseline_review"],
            }
        if current is None:
            return {
                "domain": domain,
                "metric_name": metric_name,
                "change_type": "missing",
                "severity": "critical",
                "baseline_value": baseline["observed_value"],
                "current_value": None,
                "absolute_delta": None,
                "reason_codes": ["metric_missing_from_current_snapshot"],
            }
        delta = round(current["observed_value"] - baseline["observed_value"], 10)
        magnitude = abs(delta)
        thresholds_changed = any(
            baseline[field] != current[field]
            for field in (
                "target_min", "target_max", "critical_min", "critical_max",
                "drift_warning_delta", "drift_critical_delta",
            )
        )
        reasons = []
        if current["health_status"] == "critical":
            severity = "critical"
            reasons.append("current_metric_outside_critical_range")
        elif magnitude >= current["drift_critical_delta"]:
            severity = "critical"
            reasons.append("critical_drift_delta_exceeded")
        elif current["health_status"] == "warning":
            severity = "warning"
            reasons.append("current_metric_outside_target_range")
        elif magnitude >= current["drift_warning_delta"]:
            severity = "warning"
            reasons.append("warning_drift_delta_exceeded")
        elif thresholds_changed:
            severity = "warning"
            reasons.append("quality_thresholds_changed")
        else:
            severity = "none"
            reasons.append("aggregate_metric_stable")
        return {
            "domain": domain,
            "metric_name": metric_name,
            "change_type": "changed" if delta or thresholds_changed else "unchanged",
            "severity": severity,
            "baseline_value": baseline["observed_value"],
            "current_value": current["observed_value"],
            "absolute_delta": magnitude,
            "reason_codes": reasons,
        }


class ProductionQualityAlertPlanningService:
    """Create review-only alert plans without dispatching or remediating."""

    def plan(self, *, drift_report: Mapping[str, Any]) -> dict[str, Any]:
        drift = ProductionQualityDriftService().validate_report(drift_report)
        alerts = []
        for entry in drift["drift_entries"]:
            action = {
                "critical": "quarantine_source_and_manual_review",
                "warning": "investigate_quality_signal",
                "none": "continue_monitoring",
            }[entry["severity"]]
            core = {
                "domain": entry["domain"],
                "metric_name": entry["metric_name"],
                "severity": entry["severity"],
                "recommended_action": action,
                "reason_codes": list(entry["reason_codes"]),
                "automatic_remediation_performed": False,
                "external_alert_dispatched": False,
                "manual_approval_required": True,
                "production_routing_enabled": False,
            }
            alerts.append({
                **core,
                "alert_fingerprint": stable_fingerprint(core),
            })
        fingerprint = stable_fingerprint({
            "policy_version": QUALITY_ALERT_POLICY_VERSION,
            "tenant_id": drift["request"]["tenant_id"],
            "source_quality_drift_report_fingerprint": drift[
                "quality_drift_report_fingerprint"
            ],
            "alerts": alerts,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": QUALITY_ALERT_POLICY_VERSION,
            "tenant_id": drift["request"]["tenant_id"],
            "source_quality_drift_report_fingerprint": drift[
                "quality_drift_report_fingerprint"
            ],
            "quality_alert_plan_fingerprint": fingerprint,
            "alert_count": len(alerts),
            "alerts": alerts,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported quality alert plan status.")
        if payload.get("policy_version") != QUALITY_ALERT_POLICY_VERSION:
            raise ValueError("Unsupported quality alert policy version.")
        alerts = payload.get("alerts")
        if not isinstance(alerts, list) or not alerts:
            raise ValueError("Quality alert plan entries are required.")
        for alert in alerts:
            core = {
                key: alert.get(key)
                for key in (
                    "domain", "metric_name", "severity", "recommended_action",
                    "reason_codes", "automatic_remediation_performed",
                    "external_alert_dispatched", "manual_approval_required",
                    "production_routing_enabled",
                )
            }
            expected_action = {
                "critical": "quarantine_source_and_manual_review",
                "warning": "investigate_quality_signal",
                "none": "continue_monitoring",
            }.get(core["severity"])
            if core["recommended_action"] != expected_action:
                raise ValueError("Quality alert action is invalid.")
            for field in (
                "automatic_remediation_performed",
                "external_alert_dispatched",
                "production_routing_enabled",
            ):
                if core[field] is not False:
                    raise ValueError(f"Quality alert cannot set {field}.")
            if core["manual_approval_required"] is not True:
                raise ValueError("Quality alert manual approval is required.")
            if alert.get("alert_fingerprint") != stable_fingerprint(core):
                raise ValueError("Quality alert fingerprint mismatch.")
        if int(payload.get("alert_count") or 0) != len(alerts):
            raise ValueError("Quality alert count is inconsistent.")
        tenant_id = required_slug(payload.get("tenant_id"), label="tenant_id")
        source_fingerprint = required_sha256_digest(
            payload.get("source_quality_drift_report_fingerprint"),
            label="source_quality_drift_report_fingerprint",
        )
        expected = stable_fingerprint({
            "policy_version": QUALITY_ALERT_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_quality_drift_report_fingerprint": source_fingerprint,
            "alerts": alerts,
        })
        if payload.get("quality_alert_plan_fingerprint") != expected:
            raise ValueError("Quality alert plan fingerprint mismatch.")
        _validate_safety(payload, "quality alert plan")
        return payload
