from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from app.services.production_module5_governance_service import (
    ProductionModule5AgentExecutionService,
    ProductionModule5CircuitBreakerService,
    ProductionModule5HumanShadowReviewService,
    ProductionModule5PolicyEvaluationService,
    ProductionModule5RecoveryCertificationService,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionModule5StatusService:
    """Report Module 5.1-5.5 engineering readiness without exposing evidence."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        execution = self._validated(
            "MODULE5_EXECUTION_EVIDENCE_PATH",
            ProductionModule5AgentExecutionService().validate_report,
        )
        policy = self._validated(
            "MODULE5_POLICY_EVIDENCE_PATH",
            ProductionModule5PolicyEvaluationService().validate_report,
        )
        circuit = self._validated(
            "MODULE5_CIRCUIT_BREAKER_EVIDENCE_PATH",
            ProductionModule5CircuitBreakerService().validate_report,
        )
        review = self._validated(
            "MODULE5_HUMAN_REVIEW_EVIDENCE_PATH",
            ProductionModule5HumanShadowReviewService().validate_report,
        )
        recovery = self._validated(
            "MODULE5_RECOVERY_EVIDENCE_PATH",
            ProductionModule5RecoveryCertificationService().validate_report,
        )

        execution_valid = execution is not None
        policy_valid = bool(
            execution_valid
            and policy
            and policy.get("tenant_id") == execution.get("request", {}).get("tenant_id")
            and policy.get("source_execution_report_fingerprint")
            == execution.get("execution_report_fingerprint")
        )
        circuit_valid = bool(
            execution_valid
            and circuit
            and circuit.get("request", {}).get("tenant_id")
            == execution.get("request", {}).get("tenant_id")
            and execution.get("execution_report_fingerprint")
            in circuit.get("request", {}).get("execution_report_fingerprints", [])
        )
        review_valid = bool(
            policy_valid
            and review
            and review.get("tenant_id") == policy.get("tenant_id")
            and review.get("source_policy_report_fingerprint")
            == policy.get("policy_report_fingerprint")
        )
        recovery_valid = bool(
            circuit_valid
            and review_valid
            and recovery
            and recovery.get("tenant_id") == review.get("tenant_id")
            and recovery.get("source_circuit_breaker_report_fingerprint")
            == circuit.get("circuit_breaker_report_fingerprint")
            and recovery.get("source_human_review_report_fingerprint")
            == review.get("human_review_report_fingerprint")
            and recovery.get("recovery_plan", {}).get(
                "engineering_evidence_complete"
            ) is True
            and recovery.get("recovery_plan", {}).get("production_certified")
            is False
        )

        execution_enabled = self._flag("MODULE5_GOVERNED_EXECUTION_ENABLED")
        policy_enabled = self._flag("MODULE5_POLICY_EVALUATION_ENABLED")
        circuit_enabled = self._flag("MODULE5_CIRCUIT_BREAKER_ENABLED")
        shadow_enabled = self._flag("MODULE5_SHADOW_REVIEW_ENABLED")
        mutation_enabled = self._flag("MODULE5_AUTONOMOUS_MUTATION_ENABLED")
        routing_enabled = self._flag("MODULE5_PRODUCTION_ROUTING_ENABLED")

        if mutation_enabled or routing_enabled:
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif execution_enabled and not execution_valid:
            readiness = "unsafe_configuration_execution_evidence_missing"
        elif policy_enabled and not policy_valid:
            readiness = "unsafe_configuration_policy_evidence_missing"
        elif circuit_enabled and not circuit_valid:
            readiness = "unsafe_configuration_circuit_breaker_evidence_missing"
        elif shadow_enabled and not review_valid:
            readiness = "unsafe_configuration_human_review_evidence_missing"
        elif recovery_valid:
            readiness = "module5_engineering_evidence_ready"
        elif review_valid and circuit_valid:
            readiness = "module5_4_engineering_evidence_ready"
        elif circuit_valid:
            readiness = "module5_3_engineering_evidence_ready"
        elif policy_valid:
            readiness = "module5_2_engineering_evidence_ready"
        elif execution_valid:
            readiness = "module5_1_engineering_evidence_ready"
        else:
            readiness = "module5_1_evidence_pending"

        return {
            "module": "module_5_governed_agent_coordination",
            "status": readiness,
            "components": {
                "module_5_1_minimized_execution_evidence": execution_valid,
                "module_5_2_supervisor_policy_evaluation": policy_valid,
                "module_5_3_circuit_breaker_governance": circuit_valid,
                "module_5_4_human_shadow_review": review_valid,
                "module_5_5_recovery_certification_planning": recovery_valid,
            },
            "module5_1_engineering_evidence_ready": execution_valid,
            "module5_2_engineering_evidence_ready": policy_valid,
            "module5_3_engineering_evidence_ready": circuit_valid,
            "module5_4_engineering_evidence_ready": review_valid,
            "module5_engineering_evidence_ready": recovery_valid,
            "live_production_certified": False,
            "evidence_presence": {
                "execution": execution is not None,
                "policy": policy is not None,
                "circuit_breaker": circuit is not None,
                "human_review": review is not None,
                "recovery": recovery is not None,
            },
            "feature_flags": {
                "governed_execution_enabled": execution_enabled,
                "policy_evaluation_enabled": policy_enabled,
                "circuit_breaker_enabled": circuit_enabled,
                "shadow_review_enabled": shadow_enabled,
                "autonomous_mutation_enabled": mutation_enabled,
                "production_routing_enabled": routing_enabled,
            },
            "safety": {
                "raw_identifiers_exposed": False,
                "prompt_content_exposed": False,
                "tool_arguments_exposed": False,
                "automatic_mutation_performed": False,
                "automatic_approval_performed": False,
                "manual_approval_required": True,
                "production_routing_enabled": False,
                "recovery_executed": False,
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
