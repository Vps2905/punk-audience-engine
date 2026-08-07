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
from app.models.production_observability_contracts import (
    OBSERVABILITY_INCIDENT_POLICY_VERSION,
    OBSERVABILITY_SNAPSHOT_POLICY_VERSION,
    AggregateTelemetryCoverage,
    ProductionObservabilitySnapshotRequest,
    ServiceLevelObservation,
)


_FALSE_SAFETY_FIELDS = (
    "raw_identifiers_read",
    "raw_identifiers_stored",
    "raw_identifiers_returned",
    "prompt_content_stored",
    "request_payload_stored",
    "response_payload_stored",
    "sensitive_attributes_exported",
    "external_alert_dispatched",
    "automatic_remediation_performed",
    "incident_declared_automatically",
    "production_routing_enabled",
    "activation_or_export_performed",
)


def _safety() -> dict[str, Any]:
    return {
        "raw_identifiers_read": False,
        "raw_identifiers_stored": False,
        "raw_identifiers_returned": False,
        "prompt_content_stored": False,
        "request_payload_stored": False,
        "response_payload_stored": False,
        "sensitive_attributes_exported": False,
        "aggregate_telemetry_only": True,
        "external_alert_dispatched": False,
        "automatic_remediation_performed": False,
        "incident_declared_automatically": False,
        "manual_approval_required": True,
        "monitoring_required": True,
        "production_routing_enabled": False,
        "activation_or_export_performed": False,
    }


def _validate_safety(report: Mapping[str, Any], label: str) -> None:
    safety = report.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError(f"{label} safety evidence is required.")
    for field in _FALSE_SAFETY_FIELDS:
        if safety.get(field) is not False:
            raise ValueError(f"Unsafe {label} safety field: {field}.")
    if safety.get("aggregate_telemetry_only") is not True:
        raise ValueError(f"{label} must remain aggregate-only.")
    if safety.get("manual_approval_required") is not True:
        raise ValueError(f"{label} manual approval must remain required.")
    if safety.get("monitoring_required") is not True:
        raise ValueError(f"{label} monitoring must remain required.")


class ProductionObservabilitySnapshotService:
    """Evaluate aggregate telemetry coverage and service-level objectives."""

    def build(
        self,
        *,
        request: ProductionObservabilitySnapshotRequest,
        telemetry_coverage: Sequence[
            AggregateTelemetryCoverage | Mapping[str, Any]
        ],
        service_levels: Sequence[
            ServiceLevelObservation | Mapping[str, Any]
        ],
    ) -> dict[str, Any]:
        coverage = [
            value if isinstance(value, AggregateTelemetryCoverage)
            else AggregateTelemetryCoverage(**dict(value))
            for value in telemetry_coverage
        ]
        objectives = [
            value if isinstance(value, ServiceLevelObservation)
            else ServiceLevelObservation(**dict(value))
            for value in service_levels
        ]
        if not coverage or not objectives:
            raise ValueError(
                "Observability evidence requires telemetry and service levels."
            )
        coverage_keys = [
            (value.component_name, value.signal_type) for value in coverage
        ]
        slo_keys = [(value.service_name, value.sli_name) for value in objectives]
        if len(set(coverage_keys)) != len(coverage_keys):
            raise ValueError("Telemetry coverage keys must be unique.")
        if len(set(slo_keys)) != len(slo_keys):
            raise ValueError("Service-level objective keys must be unique.")
        coverage_rows = [self._assess_coverage(value) for value in coverage]
        slo_rows = [self._assess_slo(value) for value in objectives]
        coverage_rows.sort(
            key=lambda value: (value["component_name"], value["signal_type"])
        )
        slo_rows.sort(key=lambda value: (value["service_name"], value["sli_name"]))
        counts = Counter(
            [value["coverage_status"] for value in coverage_rows]
            + [value["objective_status"] for value in slo_rows]
        )
        overall = self._overall_status(counts)
        fingerprint = stable_fingerprint({
            "policy_version": OBSERVABILITY_SNAPSHOT_POLICY_VERSION,
            "request": request.to_record(),
            "telemetry_coverage": coverage_rows,
            "service_levels": slo_rows,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": OBSERVABILITY_SNAPSHOT_POLICY_VERSION,
            "request": request.to_record(),
            "observability_snapshot_fingerprint": fingerprint,
            "telemetry_coverage_count": len(coverage_rows),
            "service_level_count": len(slo_rows),
            "healthy_count": counts["healthy"] + counts["met"],
            "at_risk_count": counts["at_risk"],
            "warning_count": counts["warning"],
            "critical_count": counts["critical"],
            "insufficient_data_count": counts["insufficient_data"],
            "overall_status": overall,
            "telemetry_coverage": coverage_rows,
            "service_levels": slo_rows,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported observability snapshot status.")
        if payload.get("policy_version") != OBSERVABILITY_SNAPSHOT_POLICY_VERSION:
            raise ValueError("Unsupported observability snapshot policy version.")
        request = ProductionObservabilitySnapshotRequest(
            **dict(payload.get("request") or {})
        )
        coverage = payload.get("telemetry_coverage")
        service_levels = payload.get("service_levels")
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("Telemetry coverage evidence is required.")
        if not isinstance(service_levels, list) or not service_levels:
            raise ValueError("Service-level evidence is required.")
        rebuilt_coverage = []
        for row in coverage:
            observation = AggregateTelemetryCoverage(**{
                key: row.get(key)
                for key in (
                    "component_name", "signal_type", "emitted_event_count",
                    "accepted_event_count", "rejected_sensitive_attribute_count",
                    "correlation_coverage_rate", "export_success_rate",
                    "minimum_correlation_coverage_rate",
                    "minimum_export_success_rate", "source_evidence_fingerprint",
                )
            })
            expected = self._assess_coverage(observation)
            if dict(row) != expected:
                raise ValueError("Telemetry coverage assessment is inconsistent.")
            rebuilt_coverage.append(expected)
        rebuilt_slos = []
        for row in service_levels:
            observation = ServiceLevelObservation(**{
                key: row.get(key)
                for key in (
                    "service_name", "sli_name", "objective_direction",
                    "observed_value", "target_value", "warning_boundary",
                    "critical_boundary", "sample_count", "minimum_sample_count",
                    "window_seconds", "source_evidence_fingerprint",
                )
            })
            expected = self._assess_slo(observation)
            if dict(row) != expected:
                raise ValueError("Service-level assessment is inconsistent.")
            rebuilt_slos.append(expected)
        expected_coverage = sorted(
            rebuilt_coverage,
            key=lambda value: (value["component_name"], value["signal_type"]),
        )
        expected_slos = sorted(
            rebuilt_slos,
            key=lambda value: (value["service_name"], value["sli_name"]),
        )
        if rebuilt_coverage != expected_coverage or rebuilt_slos != expected_slos:
            raise ValueError("Observability evidence must be deterministically ordered.")
        if len({
            (value["component_name"], value["signal_type"])
            for value in rebuilt_coverage
        }) != len(rebuilt_coverage):
            raise ValueError("Telemetry coverage keys must be unique.")
        if len({
            (value["service_name"], value["sli_name"])
            for value in rebuilt_slos
        }) != len(rebuilt_slos):
            raise ValueError("Service-level keys must be unique.")
        counts = Counter(
            [value["coverage_status"] for value in rebuilt_coverage]
            + [value["objective_status"] for value in rebuilt_slos]
        )
        expected_counts = {
            "telemetry_coverage_count": len(rebuilt_coverage),
            "service_level_count": len(rebuilt_slos),
            "healthy_count": counts["healthy"] + counts["met"],
            "at_risk_count": counts["at_risk"],
            "warning_count": counts["warning"],
            "critical_count": counts["critical"],
            "insufficient_data_count": counts["insufficient_data"],
        }
        for field, value in expected_counts.items():
            if int(payload.get(field) or 0) != value:
                raise ValueError(f"Observability {field} is inconsistent.")
        if payload.get("overall_status") != self._overall_status(counts):
            raise ValueError("Observability overall status is inconsistent.")
        expected_fingerprint = stable_fingerprint({
            "policy_version": OBSERVABILITY_SNAPSHOT_POLICY_VERSION,
            "request": request.to_record(),
            "telemetry_coverage": rebuilt_coverage,
            "service_levels": rebuilt_slos,
        })
        if payload.get("observability_snapshot_fingerprint") != expected_fingerprint:
            raise ValueError("Observability snapshot fingerprint mismatch.")
        _validate_safety(payload, "observability snapshot")
        return payload

    def _assess_coverage(self, observation: AggregateTelemetryCoverage):
        if observation.emitted_event_count == 0 or observation.accepted_event_count == 0:
            status, reason = "critical", "telemetry_signal_missing"
        elif (
            observation.correlation_coverage_rate
            < observation.minimum_correlation_coverage_rate
            or observation.export_success_rate
            < observation.minimum_export_success_rate
        ):
            status, reason = "warning", "telemetry_coverage_below_target"
        else:
            status, reason = "healthy", "telemetry_coverage_healthy"
        return {
            **observation.to_record(),
            "coverage_status": status,
            "reason_code": reason,
        }

    def _assess_slo(self, observation: ServiceLevelObservation):
        if observation.sample_count < observation.minimum_sample_count:
            status, reason = "insufficient_data", "minimum_sample_not_met"
        elif observation.objective_direction == "at_least":
            if observation.observed_value < observation.critical_boundary:
                status, reason = "critical", "critical_lower_boundary_breached"
            elif observation.observed_value < observation.warning_boundary:
                status, reason = "warning", "warning_lower_boundary_breached"
            elif observation.observed_value < observation.target_value:
                status, reason = "at_risk", "objective_lower_target_missed"
            else:
                status, reason = "met", "objective_met"
        else:
            if observation.observed_value > observation.critical_boundary:
                status, reason = "critical", "critical_upper_boundary_breached"
            elif observation.observed_value > observation.warning_boundary:
                status, reason = "warning", "warning_upper_boundary_breached"
            elif observation.observed_value > observation.target_value:
                status, reason = "at_risk", "objective_upper_target_missed"
            else:
                status, reason = "met", "objective_met"
        objective_gap = (
            observation.observed_value - observation.target_value
            if observation.objective_direction == "at_least"
            else observation.target_value - observation.observed_value
        )
        return {
            **observation.to_record(),
            "objective_status": status,
            "objective_gap": round(objective_gap, 10),
            "reason_code": reason,
        }

    def _overall_status(self, counts: Counter) -> str:
        if counts["critical"]:
            return "critical"
        if counts["warning"]:
            return "warning"
        if counts["at_risk"]:
            return "at_risk"
        if counts["insufficient_data"]:
            return "insufficient_data"
        return "healthy"


class ProductionIncidentReviewPlanningService:
    """Translate observability findings into non-dispatched review actions."""

    def plan(self, *, observability_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = ProductionObservabilitySnapshotService().validate_report(
            observability_snapshot
        )
        actions = []
        for row in snapshot["telemetry_coverage"]:
            action = {
                "critical": "declare_observability_incident_review",
                "warning": "repair_telemetry_pipeline_review",
                "healthy": "continue_monitoring",
            }[row["coverage_status"]]
            actions.append(self._action(
                signal_kind="telemetry",
                component_name=row["component_name"],
                signal_name=row["signal_type"],
                severity=row["coverage_status"],
                action=action,
                reason_code=row["reason_code"],
            ))
        for row in snapshot["service_levels"]:
            action = {
                "critical": "declare_slo_incident_review",
                "warning": "investigate_slo_breach",
                "at_risk": "monitor_error_budget",
                "insufficient_data": "collect_more_observability_evidence",
                "met": "continue_monitoring",
            }[row["objective_status"]]
            actions.append(self._action(
                signal_kind="slo",
                component_name=row["service_name"],
                signal_name=row["sli_name"],
                severity=row["objective_status"],
                action=action,
                reason_code=row["reason_code"],
            ))
        actions.sort(
            key=lambda value: (
                value["signal_kind"], value["component_name"], value["signal_name"]
            )
        )
        review_count = sum(
            value["recommended_action"] != "continue_monitoring"
            for value in actions
        )
        tenant_id = snapshot["request"]["tenant_id"]
        source_fingerprint = snapshot["observability_snapshot_fingerprint"]
        fingerprint = stable_fingerprint({
            "policy_version": OBSERVABILITY_INCIDENT_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_observability_snapshot_fingerprint": source_fingerprint,
            "actions": actions,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": OBSERVABILITY_INCIDENT_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_observability_snapshot_fingerprint": source_fingerprint,
            "incident_review_plan_fingerprint": fingerprint,
            "action_count": len(actions),
            "review_required_count": review_count,
            "actions": actions,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported incident review plan status.")
        if payload.get("policy_version") != OBSERVABILITY_INCIDENT_POLICY_VERSION:
            raise ValueError("Unsupported incident review policy version.")
        tenant_id = required_slug(payload.get("tenant_id"), label="tenant_id")
        source_fingerprint = required_sha256_digest(
            payload.get("source_observability_snapshot_fingerprint"),
            label="source_observability_snapshot_fingerprint",
        )
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Incident review actions are required.")
        for action in actions:
            expected_action = self._expected_action(
                action.get("signal_kind"), action.get("severity")
            )
            if action.get("recommended_action") != expected_action:
                raise ValueError("Incident review action is inconsistent.")
            core = {
                key: action.get(key)
                for key in (
                    "signal_kind", "component_name", "signal_name", "severity",
                    "recommended_action", "reason_code",
                    "external_alert_dispatched", "automatic_remediation_performed",
                    "incident_declared_automatically", "manual_approval_required",
                    "production_routing_enabled",
                )
            }
            for field in (
                "external_alert_dispatched",
                "automatic_remediation_performed",
                "incident_declared_automatically",
                "production_routing_enabled",
            ):
                if core[field] is not False:
                    raise ValueError(f"Incident review cannot set {field}.")
            if core["manual_approval_required"] is not True:
                raise ValueError("Incident review manual approval is required.")
            if action.get("action_fingerprint") != stable_fingerprint(core):
                raise ValueError("Incident review action fingerprint mismatch.")
        if int(payload.get("action_count") or 0) != len(actions):
            raise ValueError("Incident review action count is inconsistent.")
        review_count = sum(
            value["recommended_action"] != "continue_monitoring"
            for value in actions
        )
        if int(payload.get("review_required_count") or 0) != review_count:
            raise ValueError("Incident review required count is inconsistent.")
        expected = stable_fingerprint({
            "policy_version": OBSERVABILITY_INCIDENT_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_observability_snapshot_fingerprint": source_fingerprint,
            "actions": actions,
        })
        if payload.get("incident_review_plan_fingerprint") != expected:
            raise ValueError("Incident review plan fingerprint mismatch.")
        _validate_safety(payload, "incident review plan")
        return payload

    def _action(
        self,
        *,
        signal_kind,
        component_name,
        signal_name,
        severity,
        action,
        reason_code,
    ):
        core = {
            "signal_kind": signal_kind,
            "component_name": component_name,
            "signal_name": signal_name,
            "severity": severity,
            "recommended_action": action,
            "reason_code": reason_code,
            "external_alert_dispatched": False,
            "automatic_remediation_performed": False,
            "incident_declared_automatically": False,
            "manual_approval_required": True,
            "production_routing_enabled": False,
        }
        return {**core, "action_fingerprint": stable_fingerprint(core)}

    def _expected_action(self, signal_kind, severity):
        return {
            ("telemetry", "critical"): "declare_observability_incident_review",
            ("telemetry", "warning"): "repair_telemetry_pipeline_review",
            ("telemetry", "healthy"): "continue_monitoring",
            ("slo", "critical"): "declare_slo_incident_review",
            ("slo", "warning"): "investigate_slo_breach",
            ("slo", "at_risk"): "monitor_error_budget",
            ("slo", "insufficient_data"): "collect_more_observability_evidence",
            ("slo", "met"): "continue_monitoring",
        }.get((signal_kind, severity))
