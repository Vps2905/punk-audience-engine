from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionModule4StatusService:
    """Report Module 4.1-4.5 readiness without paths or secrets."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        evidence = self._read_json("MODULE4_EVOLUTION_SNAPSHOT_EVIDENCE_PATH")
        drift = self._read_json("MODULE4_DRIFT_EVIDENCE_PATH")
        recommendations = self._read_json("MODULE4_RECOMMENDATION_EVIDENCE_PATH")
        approval_shadow = self._read_json("MODULE4_APPROVAL_SHADOW_EVIDENCE_PATH")
        recovery = self._read_json("MODULE4_RECOVERY_EVIDENCE_PATH")
        snapshot_valid = self._valid_evidence(evidence, "source_cohort_count")
        drift_valid = bool(
            snapshot_valid
            and self._valid_evidence(drift, "cohort_count")
            and drift.get("drift_report_fingerprint")
            and drift.get("request", {}).get(
                "current_snapshot_fingerprint"
            )
            == evidence.get("snapshot_fingerprint")
        )
        recommendation_valid = bool(
            drift_valid
            and self._valid_evidence(recommendations, "recommendation_count")
            and recommendations.get("source_drift_report_fingerprint")
            == drift.get("drift_report_fingerprint")
        )
        approval_valid = bool(
            recommendation_valid
            and self._valid_evidence(approval_shadow, "manual_review_count")
            and approval_shadow.get("source_recommendation_report_fingerprint")
            == recommendations.get("recommendation_report_fingerprint")
            and approval_shadow.get("shadow_validation_passed") is True
        )
        recovery_valid = bool(
            approval_valid
            and self._valid_evidence(recovery, "recovery_plan_count")
            and recovery.get("source_approval_shadow_report_fingerprint")
            == approval_shadow.get("approval_shadow_report_fingerprint")
            and recovery.get("recovery_coverage_complete") is True
            and recovery.get("safety", {}).get("recovery_executed") is False
        )
        observation_enabled = _truthy(
            self._environment.get("MODULE4_EVOLUTION_SNAPSHOT_ENABLED")
        )
        drift_enabled = _truthy(
            self._environment.get("MODULE4_DRIFT_DETECTION_ENABLED")
        )
        recommendation_enabled = _truthy(
            self._environment.get("MODULE4_RECOMMENDATION_ENABLED")
        )
        shadow_enabled = _truthy(
            self._environment.get("MODULE4_SHADOW_VALIDATION_ENABLED")
        )
        mutation_enabled = _truthy(
            self._environment.get("MODULE4_AUTOMATIC_EVOLUTION_ENABLED")
        )
        routing_enabled = _truthy(
            self._environment.get("MODULE4_PRODUCTION_ROUTING_ENABLED")
        )
        if mutation_enabled or routing_enabled:
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif observation_enabled and not snapshot_valid:
            readiness = "unsafe_configuration_evolution_evidence_missing"
        elif drift_enabled and not drift_valid:
            readiness = "unsafe_configuration_drift_evidence_missing"
        elif recommendation_enabled and not recommendation_valid:
            readiness = "unsafe_configuration_recommendation_evidence_missing"
        elif shadow_enabled and not approval_valid:
            readiness = "unsafe_configuration_shadow_evidence_missing"
        elif recovery_valid:
            readiness = "module4_engineering_evidence_ready"
        elif approval_valid:
            readiness = "module4_4_engineering_evidence_ready"
        elif recommendation_valid:
            readiness = "module4_3_engineering_evidence_ready"
        elif drift_valid:
            readiness = "module4_2_engineering_evidence_ready"
        elif snapshot_valid:
            readiness = "module4_1_engineering_evidence_ready"
        else:
            readiness = "module4_1_evidence_pending"
        return {
            "module": "module_4_governed_audience_evolution",
            "status": readiness,
            "components": {
                "module_4_1_evolution_snapshot_foundation": snapshot_valid,
                "module_4_2_drift_detection": drift_valid,
                "module_4_3_evolution_recommendations": recommendation_valid,
                "module_4_4_approval_and_shadow_validation": approval_valid,
                "module_4_5_rollback_and_recovery": recovery_valid,
            },
            "module4_1_engineering_evidence_ready": snapshot_valid,
            "module4_2_engineering_evidence_ready": drift_valid,
            "module4_3_engineering_evidence_ready": recommendation_valid,
            "module4_4_engineering_evidence_ready": approval_valid,
            "module4_engineering_evidence_ready": recovery_valid,
            "evidence_presence": {
                "evolution_snapshot": evidence is not None,
                "drift_report": drift is not None,
                "recommendation_report": recommendations is not None,
                "approval_shadow_report": approval_shadow is not None,
                "recovery_report": recovery is not None,
            },
            "feature_flags": {
                "evolution_snapshot_enabled": observation_enabled,
                "drift_detection_enabled": drift_enabled,
                "recommendation_enabled": recommendation_enabled,
                "shadow_validation_enabled": shadow_enabled,
                "automatic_evolution_enabled": mutation_enabled,
                "production_routing_enabled": routing_enabled,
            },
            "safety": {
                "raw_identifiers_exposed": False,
                "audience_membership_read": False,
                "automatic_evolution_performed": False,
                "automatic_approval_performed": False,
                "manual_approval_required": True,
                "production_routing_enabled": False,
                "recovery_executed": False,
                "activation_or_export_performed": False,
            },
        }

    def _valid_evidence(
        self,
        evidence: Mapping[str, Any] | None,
        count_field: str,
    ) -> bool:
        safety = evidence.get("safety", {}) if evidence else {}
        return bool(
            evidence
            and evidence.get("status") == "engineering_preview_ready"
            and int(evidence.get(count_field) or 0) > 0
            and safety.get("raw_identifiers_returned") is False
            and safety.get("audience_membership_read") is False
            and safety.get("individual_behavior_inferred") is False
            and safety.get("cohort_lifecycle_mutated") is False
            and safety.get("automatic_evolution_performed") is False
            and safety.get("automatic_approval_performed") is False
            and safety.get("manual_approval_required") is True
            and safety.get("routing_enabled") is False
            and safety.get("activation_or_export_performed") is False
            and safety.get("downstream_export_enabled") is False
        )

    def _read_json(self, key: str) -> dict[str, Any] | None:
        raw = str(self._environment.get(key) or "").strip()
        if not raw or not Path(raw).is_file():
            return None
        try:
            value = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, Mapping) else None
