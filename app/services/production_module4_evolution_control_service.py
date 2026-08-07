from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.models.production_module4_evolution_control_contracts import (
    MODULE4_APPROVAL_SHADOW_POLICY_VERSION,
    MODULE4_DRIFT_POLICY_VERSION,
    MODULE4_RECOVERY_POLICY_VERSION,
    MODULE4_RECOMMENDATION_POLICY_VERSION,
    Module4DriftAnalysisRequest,
)
from app.services.production_module4_evolution_snapshot_service import (
    ProductionModule4EvolutionSnapshotService,
)


_FALSE_SAFETY_FIELDS = (
    "raw_identifiers_read",
    "raw_identifiers_stored",
    "raw_identifiers_returned",
    "audience_membership_read",
    "individual_behavior_inferred",
    "cohort_lifecycle_mutated",
    "automatic_evolution_performed",
    "automatic_approval_performed",
    "routing_enabled",
    "activation_or_export_performed",
    "downstream_export_enabled",
)


def _safety() -> dict[str, Any]:
    return {
        "raw_identifiers_read": False,
        "raw_identifiers_stored": False,
        "raw_identifiers_returned": False,
        "audience_membership_read": False,
        "individual_behavior_inferred": False,
        "cohort_lifecycle_mutated": False,
        "automatic_evolution_performed": False,
        "automatic_approval_performed": False,
        "manual_approval_required": True,
        "monitoring_required": True,
        "routing_enabled": False,
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
    if safety.get("manual_approval_required") is not True:
        raise ValueError(f"{label} manual approval must remain required.")
    if safety.get("monitoring_required") is not True:
        raise ValueError(f"{label} monitoring must remain required.")


class ProductionModule4DriftDetectionService:
    """Compare two immutable aggregate snapshots without reading membership."""

    def analyze(
        self,
        *,
        request: Module4DriftAnalysisRequest,
        baseline_snapshot: Mapping[str, Any],
        current_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        snapshot_service = ProductionModule4EvolutionSnapshotService()
        baseline = snapshot_service.validate_report(baseline_snapshot)
        current = snapshot_service.validate_report(current_snapshot)
        self._validate_sources(request, baseline, current)
        baseline_rows = self._by_id(baseline)
        current_rows = self._by_id(current)
        entries = [
            self._entry(cohort_id, baseline_rows.get(cohort_id), current_rows.get(cohort_id))
            for cohort_id in sorted(set(baseline_rows) | set(current_rows))
        ]
        fingerprint = stable_fingerprint({
            "policy_version": MODULE4_DRIFT_POLICY_VERSION,
            "request": request.to_record(),
            "entries": entries,
        })
        counts = self._counts(entries)
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE4_DRIFT_POLICY_VERSION,
            "request": request.to_record(),
            "drift_report_fingerprint": fingerprint,
            "cohort_count": len(entries),
            **counts,
            "drift_detected": counts["changed_count"] > 0,
            "drift_entries": entries,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 4.2 report status.")
        if payload.get("policy_version") != MODULE4_DRIFT_POLICY_VERSION:
            raise ValueError("Unsupported Module 4.2 policy version.")
        request = Module4DriftAnalysisRequest(**dict(payload.get("request") or {}))
        entries = payload.get("drift_entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Module 4.2 drift entries are required.")
        allowed_changes = {"new", "missing", "changed", "unchanged"}
        allowed_severities = {"none", "review", "high", "critical"}
        if any(
            value.get("change_type") not in allowed_changes
            or value.get("severity") not in allowed_severities
            or not str(value.get("export_cohort_id") or "").strip()
            or not value.get("reason_codes")
            for value in entries
        ):
            raise ValueError("Module 4.2 drift entry is invalid.")
        expected = stable_fingerprint({
            "policy_version": MODULE4_DRIFT_POLICY_VERSION,
            "request": request.to_record(),
            "entries": entries,
        })
        if payload.get("drift_report_fingerprint") != expected:
            raise ValueError("Module 4.2 drift fingerprint mismatch.")
        counts = self._counts(entries)
        if int(payload.get("cohort_count") or 0) != len(entries):
            raise ValueError("Module 4.2 cohort count is inconsistent.")
        for field, value in counts.items():
            if int(payload.get(field) or 0) != value:
                raise ValueError(f"Module 4.2 {field} is inconsistent.")
        if bool(payload.get("drift_detected")) != (counts["changed_count"] > 0):
            raise ValueError("Module 4.2 drift result is inconsistent.")
        _validate_safety(payload, "Module 4.2")
        return payload

    def _validate_sources(self, request, baseline, current) -> None:
        for label, report, fingerprint in (
            ("baseline", baseline, request.baseline_snapshot_fingerprint),
            ("current", current, request.current_snapshot_fingerprint),
        ):
            if report.get("snapshot_fingerprint") != fingerprint:
                raise ValueError(f"Module 4.2 {label} fingerprint mismatch.")
            if report.get("request", {}).get("tenant_id") != request.tenant_id:
                raise ValueError(f"Module 4.2 {label} tenant mismatch.")

    def _by_id(self, report):
        return {
            str(value["export_cohort_id"]): dict(value)
            for value in report.get("cohort_snapshots") or []
        }

    def _entry(self, cohort_id, baseline, current):
        if baseline is None:
            change, severity, reasons = "new", "review", ["new_cohort_observed"]
        elif current is None:
            change, severity, reasons = "missing", "critical", ["cohort_missing_from_current_snapshot"]
        else:
            quality_delta = round(
                float(current["quality_score"]) - float(baseline["quality_score"]), 8
            )
            reasons = []
            severity = "none"
            if str(current["monitoring_status"]).startswith("blocked_"):
                severity, reasons = "critical", ["current_policy_or_approval_block"]
            elif current["freshness_status"] != "fresh":
                severity, reasons = "high", ["current_source_not_fresh"]
            elif quality_delta < -0.15:
                severity, reasons = "high", ["quality_drop_exceeds_threshold"]
            elif any(
                baseline[field] != current[field]
                for field in (
                    "approval_status", "data_safety_status", "risk_decision",
                    "monitoring_status",
                )
            ):
                severity, reasons = "review", ["governance_state_changed"]
            change = "changed" if reasons else "unchanged"
            if not reasons:
                reasons = ["aggregate_snapshot_stable"]
            return {
                "export_cohort_id": cohort_id,
                "change_type": change,
                "severity": severity,
                "baseline_quality_score": baseline["quality_score"],
                "current_quality_score": current["quality_score"],
                "quality_delta": quality_delta,
                "baseline_monitoring_status": baseline["monitoring_status"],
                "current_monitoring_status": current["monitoring_status"],
                "reason_codes": reasons,
            }
        row = current or baseline
        return {
            "export_cohort_id": cohort_id,
            "change_type": change,
            "severity": severity,
            "baseline_quality_score": baseline.get("quality_score") if baseline else None,
            "current_quality_score": current.get("quality_score") if current else None,
            "quality_delta": None,
            "baseline_monitoring_status": baseline.get("monitoring_status") if baseline else None,
            "current_monitoring_status": current.get("monitoring_status") if current else None,
            "reason_codes": reasons,
        }

    def _counts(self, entries):
        return {
            "unchanged_count": sum(value["change_type"] == "unchanged" for value in entries),
            "changed_count": sum(value["change_type"] != "unchanged" for value in entries),
            "new_count": sum(value["change_type"] == "new" for value in entries),
            "missing_count": sum(value["change_type"] == "missing" for value in entries),
            "critical_count": sum(value["severity"] == "critical" for value in entries),
        }


class ProductionModule4EvolutionRecommendationService:
    """Convert drift into review recommendations, never mutations."""

    def recommend(self, *, drift_report: Mapping[str, Any]) -> dict[str, Any]:
        drift = ProductionModule4DriftDetectionService().validate_report(drift_report)
        recommendations = []
        action_by_state = {
            ("new", "review"): "onboarding_review",
            ("missing", "critical"): "rollback_review",
            ("changed", "critical"): "rollback_review",
            ("changed", "high"): "pause_and_review",
            ("changed", "review"): "investigate",
            ("unchanged", "none"): "maintain_monitoring",
        }
        for entry in drift["drift_entries"]:
            action = action_by_state.get(
                (entry["change_type"], entry["severity"]), "investigate"
            )
            identity = {
                "tenant_id": drift["request"]["tenant_id"],
                "drift_report_fingerprint": drift["drift_report_fingerprint"],
                "export_cohort_id": entry["export_cohort_id"],
                "recommended_action": action,
                "reason_codes": list(entry["reason_codes"]),
                "manual_approval_required": True,
                "lifecycle_mutated": False,
                "routing_enabled": False,
                "eligible_for_activation": False,
                "eligible_for_export": False,
            }
            recommendations.append({
                "recommendation_fingerprint": stable_fingerprint(identity),
                **identity,
            })
        fingerprint = stable_fingerprint({
            "policy_version": MODULE4_RECOMMENDATION_POLICY_VERSION,
            "drift_report_fingerprint": drift["drift_report_fingerprint"],
            "recommendations": recommendations,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE4_RECOMMENDATION_POLICY_VERSION,
            "tenant_id": drift["request"]["tenant_id"],
            "source_drift_report_fingerprint": drift["drift_report_fingerprint"],
            "recommendation_report_fingerprint": fingerprint,
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report):
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("policy_version") != MODULE4_RECOMMENDATION_POLICY_VERSION:
            raise ValueError("Unsupported Module 4.3 policy version.")
        values = payload.get("recommendations")
        if not isinstance(values, list) or not values:
            raise ValueError("Module 4.3 recommendations are required.")
        for value in values:
            identity = dict(value)
            fingerprint = identity.pop("recommendation_fingerprint", None)
            if stable_fingerprint(identity) != fingerprint:
                raise ValueError("Module 4.3 recommendation fingerprint mismatch.")
            if value.get("manual_approval_required") is not True or any(
                value.get(field) is not False
                for field in (
                    "lifecycle_mutated", "routing_enabled",
                    "eligible_for_activation", "eligible_for_export",
                )
            ):
                raise ValueError("Unsafe Module 4.3 recommendation state.")
        expected = stable_fingerprint({
            "policy_version": MODULE4_RECOMMENDATION_POLICY_VERSION,
            "drift_report_fingerprint": payload.get("source_drift_report_fingerprint"),
            "recommendations": values,
        })
        if payload.get("recommendation_report_fingerprint") != expected:
            raise ValueError("Module 4.3 report fingerprint mismatch.")
        if int(payload.get("recommendation_count") or 0) != len(values):
            raise ValueError("Module 4.3 recommendation count is inconsistent.")
        _validate_safety(payload, "Module 4.3")
        return payload


class ProductionModule4ApprovalShadowService:
    """Record manual review and non-routing shadow evidence."""

    _DECISIONS = {"approved_for_shadow_review", "rejected", "needs_review"}
    _OBSERVATIONS = {"aligned", "degraded", "error"}

    def evaluate(
        self,
        *,
        recommendation_report: Mapping[str, Any],
        manual_review_records: Sequence[Mapping[str, Any]],
        shadow_observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        recommendations = ProductionModule4EvolutionRecommendationService().validate_report(
            recommendation_report
        )
        by_fp = {
            value["recommendation_fingerprint"]: value
            for value in recommendations["recommendations"]
        }
        decisions = [dict(value) for value in manual_review_records]
        observations = [dict(value) for value in shadow_observations]
        assert_no_raw_identifier_fields([decisions, observations])
        if not decisions:
            raise ValueError("Module 4.4 requires manual review evidence.")
        decision_by_fp = {}
        for value in decisions:
            fingerprint = str(value.get("recommendation_fingerprint") or "")
            if fingerprint not in by_fp or fingerprint in decision_by_fp:
                raise ValueError("Module 4.4 manual review reference is invalid.")
            if value.get("decision") not in self._DECISIONS:
                raise ValueError("Module 4.4 manual review decision is invalid.")
            if not str(value.get("decision_reference") or "").strip():
                raise ValueError("Module 4.4 decision reference is required.")
            decision_by_fp[fingerprint] = value
        observation_by_fp = {}
        for value in observations:
            fingerprint = str(value.get("recommendation_fingerprint") or "")
            if fingerprint in observation_by_fp:
                raise ValueError("Module 4.4 shadow observation is duplicated.")
            decision = decision_by_fp.get(fingerprint)
            if not decision or decision.get("decision") != "approved_for_shadow_review":
                raise ValueError("Shadow evidence requires manual shadow-review approval.")
            if value.get("observation_status") not in self._OBSERVATIONS:
                raise ValueError("Module 4.4 shadow observation status is invalid.")
            if value.get("routing_enabled") is not False:
                raise ValueError("Module 4.4 shadow routing must remain disabled.")
            observation_by_fp[fingerprint] = value
        approved = {
            key for key, value in decision_by_fp.items()
            if value["decision"] == "approved_for_shadow_review"
        }
        passed = bool(
            approved
            and approved == set(observation_by_fp)
            and all(
                observation_by_fp[key]["observation_status"] == "aligned"
                for key in approved
            )
        )
        normalized_decisions = sorted(decisions, key=lambda value: value["recommendation_fingerprint"])
        normalized_observations = sorted(observations, key=lambda value: value["recommendation_fingerprint"])
        fingerprint = stable_fingerprint({
            "policy_version": MODULE4_APPROVAL_SHADOW_POLICY_VERSION,
            "recommendation_report_fingerprint": recommendations["recommendation_report_fingerprint"],
            "manual_review_records": normalized_decisions,
            "shadow_observations": normalized_observations,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE4_APPROVAL_SHADOW_POLICY_VERSION,
            "tenant_id": recommendations["tenant_id"],
            "source_recommendation_report_fingerprint": recommendations["recommendation_report_fingerprint"],
            "approval_shadow_report_fingerprint": fingerprint,
            "manual_review_count": len(decisions),
            "approved_for_shadow_review_count": len(approved),
            "shadow_observation_count": len(observations),
            "shadow_validation_passed": passed,
            "manual_review_records": normalized_decisions,
            "shadow_observations": normalized_observations,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report):
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("policy_version") != MODULE4_APPROVAL_SHADOW_POLICY_VERSION:
            raise ValueError("Unsupported Module 4.4 policy version.")
        decisions = payload.get("manual_review_records")
        observations = payload.get("shadow_observations")
        if not isinstance(decisions, list) or not isinstance(observations, list):
            raise ValueError("Module 4.4 review and shadow evidence are required.")
        decision_by_fp = {}
        for value in decisions:
            fingerprint = str(value.get("recommendation_fingerprint") or "")
            if (
                not fingerprint
                or fingerprint in decision_by_fp
                or value.get("decision") not in self._DECISIONS
                or not str(value.get("decision_reference") or "").strip()
            ):
                raise ValueError("Module 4.4 manual review record is invalid.")
            decision_by_fp[fingerprint] = value
        observation_by_fp = {}
        for value in observations:
            fingerprint = str(value.get("recommendation_fingerprint") or "")
            decision = decision_by_fp.get(fingerprint)
            if (
                not decision
                or fingerprint in observation_by_fp
                or decision.get("decision") != "approved_for_shadow_review"
                or value.get("observation_status") not in self._OBSERVATIONS
                or value.get("routing_enabled") is not False
            ):
                raise ValueError("Module 4.4 shadow observation is invalid.")
            observation_by_fp[fingerprint] = value
        expected = stable_fingerprint({
            "policy_version": MODULE4_APPROVAL_SHADOW_POLICY_VERSION,
            "recommendation_report_fingerprint": payload.get("source_recommendation_report_fingerprint"),
            "manual_review_records": decisions,
            "shadow_observations": observations,
        })
        if payload.get("approval_shadow_report_fingerprint") != expected:
            raise ValueError("Module 4.4 report fingerprint mismatch.")
        if int(payload.get("manual_review_count") or 0) != len(decisions):
            raise ValueError("Module 4.4 manual review count is inconsistent.")
        if int(payload.get("shadow_observation_count") or 0) != len(observations):
            raise ValueError("Module 4.4 shadow count is inconsistent.")
        approved = {
            key for key, value in decision_by_fp.items()
            if value["decision"] == "approved_for_shadow_review"
        }
        if int(payload.get("approved_for_shadow_review_count") or 0) != len(approved):
            raise ValueError("Module 4.4 approved count is inconsistent.")
        passed = bool(
            approved
            and approved == set(observation_by_fp)
            and all(
                observation_by_fp[key]["observation_status"] == "aligned"
                for key in approved
            )
        )
        if bool(payload.get("shadow_validation_passed")) != passed:
            raise ValueError("Module 4.4 shadow result is inconsistent.")
        _validate_safety(payload, "Module 4.4")
        return payload


class ProductionModule4RecoveryPlanningService:
    """Create non-executing rollback/recovery review plans."""

    def plan(
        self,
        *,
        recommendation_report: Mapping[str, Any],
        approval_shadow_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        recommendations = ProductionModule4EvolutionRecommendationService().validate_report(
            recommendation_report
        )
        approval = ProductionModule4ApprovalShadowService().validate_report(
            approval_shadow_report
        )
        if approval["source_recommendation_report_fingerprint"] != recommendations["recommendation_report_fingerprint"]:
            raise ValueError("Module 4.5 evidence lineage mismatch.")
        recommendation_by_fp = {
            value["recommendation_fingerprint"]: value
            for value in recommendations["recommendations"]
        }
        observation_by_fp = {
            value["recommendation_fingerprint"]: value
            for value in approval["shadow_observations"]
        }
        plans = []
        for decision in approval["manual_review_records"]:
            rec = recommendation_by_fp[decision["recommendation_fingerprint"]]
            observation = observation_by_fp.get(decision["recommendation_fingerprint"])
            if observation and observation["observation_status"] in {"degraded", "error"}:
                action = "restore_baseline_review"
            elif decision["decision"] == "rejected":
                action = "retain_current_state"
            elif decision["decision"] == "approved_for_shadow_review":
                action = "hold_baseline_recovery_ready"
            else:
                action = "await_manual_resolution"
            identity = {
                "tenant_id": recommendations["tenant_id"],
                "recommendation_fingerprint": decision["recommendation_fingerprint"],
                "export_cohort_id": rec["export_cohort_id"],
                "recovery_action": action,
                "manual_execution_required": True,
                "recovery_executed": False,
                "lifecycle_mutated": False,
                "routing_enabled": False,
                "activation_or_export_performed": False,
            }
            plans.append({"recovery_plan_fingerprint": stable_fingerprint(identity), **identity})
        plans.sort(key=lambda value: value["recovery_plan_fingerprint"])
        fingerprint = stable_fingerprint({
            "policy_version": MODULE4_RECOVERY_POLICY_VERSION,
            "approval_shadow_report_fingerprint": approval["approval_shadow_report_fingerprint"],
            "plans": plans,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": MODULE4_RECOVERY_POLICY_VERSION,
            "tenant_id": recommendations["tenant_id"],
            "source_approval_shadow_report_fingerprint": approval["approval_shadow_report_fingerprint"],
            "recovery_report_fingerprint": fingerprint,
            "reviewed_recommendation_count": len(approval["manual_review_records"]),
            "recovery_plan_count": len(plans),
            "recovery_coverage_complete": bool(plans) and len(plans) == len(approval["manual_review_records"]),
            "recovery_plans": plans,
            "safety": {**_safety(), "recovery_executed": False},
        }
        return self.validate_report(report)

    def validate_report(self, report):
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("policy_version") != MODULE4_RECOVERY_POLICY_VERSION:
            raise ValueError("Unsupported Module 4.5 policy version.")
        plans = payload.get("recovery_plans")
        if not isinstance(plans, list) or not plans:
            raise ValueError("Module 4.5 recovery plans are required.")
        for value in plans:
            identity = dict(value)
            fingerprint = identity.pop("recovery_plan_fingerprint", None)
            if stable_fingerprint(identity) != fingerprint:
                raise ValueError("Module 4.5 recovery plan fingerprint mismatch.")
            if value.get("manual_execution_required") is not True or any(
                value.get(field) is not False
                for field in (
                    "recovery_executed", "lifecycle_mutated", "routing_enabled",
                    "activation_or_export_performed",
                )
            ):
                raise ValueError("Unsafe Module 4.5 recovery state.")
        expected = stable_fingerprint({
            "policy_version": MODULE4_RECOVERY_POLICY_VERSION,
            "approval_shadow_report_fingerprint": payload.get("source_approval_shadow_report_fingerprint"),
            "plans": plans,
        })
        if payload.get("recovery_report_fingerprint") != expected:
            raise ValueError("Module 4.5 report fingerprint mismatch.")
        if int(payload.get("recovery_plan_count") or 0) != len(plans):
            raise ValueError("Module 4.5 plan count is inconsistent.")
        if int(payload.get("reviewed_recommendation_count") or 0) != len(plans):
            raise ValueError("Module 4.5 reviewed count is inconsistent.")
        if payload.get("recovery_coverage_complete") is not True:
            raise ValueError("Module 4.5 recovery coverage is incomplete.")
        _validate_safety(payload, "Module 4.5")
        if payload.get("safety", {}).get("recovery_executed") is not False:
            raise ValueError("Module 4.5 cannot execute recovery.")
        return payload
