from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.services.production_agent_security_certification_service import (
    ProductionAgentSecurityCertificationService,
)
from app.services.production_bounded_autonomy_certification_service import (
    ProductionBoundedAutonomyCertificationService,
)
from app.services.production_bounded_autonomy_functional_shadow_service import (
    ProductionBoundedAutonomyFunctionalShadowService,
)
from app.services.production_bounded_autonomy_service import (
    ProductionBoundedAutonomyService,
)
from app.services.production_bounded_autonomy_shadow_service import (
    ProductionBoundedAutonomyShadowComparisonService,
)
from app.services.production_module5_governance_service import (
    ProductionModule5AgentExecutionService,
    ProductionModule5CircuitBreakerService,
    ProductionModule5HumanShadowReviewService,
    ProductionModule5PolicyEvaluationService,
    ProductionModule5RecoveryCertificationService,
)
from app.services.production_module5_scale_recovery_certification_service import (
    ProductionModule5ScaleRecoveryCertificationService,
)
from app.services.production_security_hardening_service import (
    ProductionSecurityPostureService,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionModule5StatusService:
    """Report Module 5.1-5.12 engineering readiness without exposing evidence."""

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
        bounded_autonomy = self._validated(
            "MODULE5_BOUNDED_AUTONOMY_EVIDENCE_PATH",
            ProductionBoundedAutonomyService().validate_report,
        )
        dual_run_comparison = self._validated(
            "MODULE5_BOUNDED_AUTONOMY_COMPARISON_EVIDENCE_PATH",
            ProductionBoundedAutonomyShadowComparisonService(
                environment={}
            ).validate_report,
        )
        shadow_certification = self._validated(
            "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_EVIDENCE_PATH",
            ProductionBoundedAutonomyCertificationService(
                environment={}
            ).validate_report,
        )
        functional_shadow = self._validated(
            "MODULE5_BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_EVIDENCE_PATH",
            ProductionBoundedAutonomyFunctionalShadowService(
                environment={}
            ).validate_report,
        )
        security_posture = self._validated(
            "SECURITY_POSTURE_EVIDENCE_PATH",
            ProductionSecurityPostureService().validate_report,
        )
        agent_security = self._validated(
            "MODULE5_AGENT_SECURITY_CERTIFICATION_EVIDENCE_PATH",
            ProductionAgentSecurityCertificationService().validate_report,
        )
        scale_recovery = self._validated(
            "MODULE5_SCALE_RECOVERY_CERTIFICATION_EVIDENCE_PATH",
            ProductionModule5ScaleRecoveryCertificationService(
                environment={}
            ).validate_report,
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
            and recovery.get("recovery_plan", {}).get("engineering_evidence_complete")
            is True
            and recovery.get("recovery_plan", {}).get("production_certified") is False
        )

        execution_enabled = self._flag("MODULE5_GOVERNED_EXECUTION_ENABLED")
        policy_enabled = self._flag("MODULE5_POLICY_EVALUATION_ENABLED")
        circuit_enabled = self._flag("MODULE5_CIRCUIT_BREAKER_ENABLED")
        shadow_enabled = self._flag("MODULE5_SHADOW_REVIEW_ENABLED")
        mutation_enabled = self._flag("MODULE5_AUTONOMOUS_MUTATION_ENABLED")
        routing_enabled = self._flag("MODULE5_PRODUCTION_ROUTING_ENABLED")
        bounded_autonomy_enabled = self._flag("MODULE5_BOUNDED_AUTONOMY_SHADOW_ENABLED")
        bounded_autonomy_valid = bounded_autonomy is not None
        dual_run_enabled = self._flag("MODULE5_BOUNDED_AUTONOMY_DUAL_RUN_ENABLED")
        dual_run_valid = bool(
            bounded_autonomy_valid
            and dual_run_comparison
            and dual_run_comparison.get("tenant_id")
            == bounded_autonomy.get("goal", {}).get("tenant_id")
            and dual_run_comparison.get("lineage", {}).get(
                "bounded_autonomy_report_fingerprint"
            )
            == bounded_autonomy.get("bounded_autonomy_report_fingerprint")
        )
        certification_enabled = self._flag(
            "MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_ENABLED"
        )
        latest_comparison_fingerprint = (
            dual_run_comparison.get("shadow_comparison_report_fingerprint")
            if dual_run_comparison
            else None
        )
        certification_valid = bool(
            dual_run_valid
            and shadow_certification
            and shadow_certification.get("tenant_id")
            == dual_run_comparison.get("tenant_id")
            and latest_comparison_fingerprint
            in {
                sample.get("comparison_report_fingerprint")
                for sample in shadow_certification.get("samples", [])
            }
        )
        certification_passed = bool(
            certification_valid
            and shadow_certification.get("status") == "engineering_preview_ready"
            and shadow_certification.get("review", {}).get(
                "eligible_for_staging_review"
            )
            is True
            and shadow_certification.get("review", {}).get("live_cutover_authorized")
            is False
        )
        functional_shadow_enabled = self._flag(
            "MODULE5_BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_ENABLED"
        )
        certification_fingerprint = (
            shadow_certification.get("bounded_autonomy_certification_fingerprint")
            if shadow_certification
            else None
        )
        functional_shadow_valid = bool(
            certification_passed
            and functional_shadow
            and functional_shadow.get("request", {}).get("tenant_id")
            == shadow_certification.get("tenant_id")
            and functional_shadow.get("lineage", {}).get(
                "source_certification_report_fingerprint"
            )
            == certification_fingerprint
            and functional_shadow.get("status") == "engineering_preview_ready"
            and functional_shadow.get("review", {}).get("eligible_for_staging_review")
            is True
            and functional_shadow.get("review", {}).get("live_cutover_authorized")
            is False
        )
        agent_security_enabled = self._flag(
            "MODULE5_AGENT_SECURITY_CERTIFICATION_ENABLED"
        )
        production_effect_authorization_enabled = self._flag(
            "MODULE5_AGENT_PRODUCTION_EFFECT_AUTHORIZATION_ENABLED"
        )
        security_posture_valid = bool(
            functional_shadow
            and security_posture
            and security_posture.get("posture_status") == "pass"
            and security_posture.get("request", {}).get("tenant_id")
            == functional_shadow.get("request", {}).get("tenant_id")
        )
        agent_security_valid = bool(
            functional_shadow_valid
            and security_posture_valid
            and agent_security
            and agent_security.get("request", {}).get("tenant_id")
            == functional_shadow.get("request", {}).get("tenant_id")
            and agent_security.get("lineage", {}).get(
                "source_functional_shadow_report_fingerprint"
            )
            == functional_shadow.get("functional_shadow_report_fingerprint")
            and agent_security.get("lineage", {}).get(
                "source_security_posture_fingerprint"
            )
            == security_posture.get("security_posture_fingerprint")
            and agent_security.get("status") == "engineering_preview_ready"
            and agent_security.get("review", {}).get(
                "eligible_for_staging_review"
            )
            is True
            and agent_security.get("review", {}).get("live_cutover_authorized")
            is False
        )
        scale_recovery_enabled = self._flag(
            "MODULE5_SCALE_RECOVERY_CERTIFICATION_ENABLED"
        )
        scale_recovery_cutover_enabled = self._flag(
            "MODULE5_SCALE_RECOVERY_PRODUCTION_CUTOVER_ENABLED"
        )
        scale_recovery_valid = bool(
            agent_security_valid
            and scale_recovery
            and scale_recovery.get("request", {}).get("tenant_id")
            == agent_security.get("request", {}).get("tenant_id")
            and scale_recovery.get("lineage", {}).get(
                "source_functional_shadow_report_fingerprint"
            )
            == functional_shadow.get("functional_shadow_report_fingerprint")
            and scale_recovery.get("lineage", {}).get(
                "source_agent_security_certification_fingerprint"
            )
            == agent_security.get("agent_security_certification_fingerprint")
            and scale_recovery.get("status") == "engineering_preview_ready"
            and scale_recovery.get("policy", {}).get(
                "require_observed_work_units"
            )
            is True
            and scale_recovery.get("policy", {}).get(
                "require_historical_pipeline"
            )
            is True
            and scale_recovery.get("review", {}).get(
                "eligible_for_staging_review"
            )
            is True
            and scale_recovery.get("review", {}).get("live_cutover_authorized")
            is False
        )

        if (
            mutation_enabled
            or routing_enabled
            or production_effect_authorization_enabled
            or scale_recovery_cutover_enabled
        ):
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif scale_recovery_enabled and not scale_recovery_valid:
            readiness = "unsafe_configuration_scale_recovery_not_ready"
        elif agent_security_enabled and not agent_security_valid:
            readiness = "unsafe_configuration_agent_security_not_ready"
        elif functional_shadow_enabled and not functional_shadow_valid:
            readiness = "unsafe_configuration_functional_shadow_not_ready"
        elif dual_run_enabled and not dual_run_valid:
            readiness = "unsafe_configuration_dual_run_evidence_missing"
        elif certification_enabled and not certification_passed:
            readiness = "unsafe_configuration_shadow_certification_not_ready"
        elif bounded_autonomy_enabled and not bounded_autonomy_valid:
            readiness = "unsafe_configuration_bounded_autonomy_evidence_missing"
        elif execution_enabled and not execution_valid:
            readiness = "unsafe_configuration_execution_evidence_missing"
        elif policy_enabled and not policy_valid:
            readiness = "unsafe_configuration_policy_evidence_missing"
        elif circuit_enabled and not circuit_valid:
            readiness = "unsafe_configuration_circuit_breaker_evidence_missing"
        elif shadow_enabled and not review_valid:
            readiness = "unsafe_configuration_human_review_evidence_missing"
        elif (
            recovery_valid
            and bounded_autonomy_valid
            and dual_run_valid
            and certification_passed
            and functional_shadow_valid
            and agent_security_valid
            and scale_recovery_valid
        ):
            readiness = "module5_scale_latency_recovery_evidence_ready"
        elif (
            recovery_valid
            and bounded_autonomy_valid
            and dual_run_valid
            and certification_passed
            and functional_shadow_valid
            and agent_security_valid
        ):
            readiness = "module5_agent_security_authorization_evidence_ready"
        elif (
            recovery_valid
            and bounded_autonomy_valid
            and dual_run_valid
            and certification_passed
            and functional_shadow_valid
        ):
            readiness = "module5_functional_shadow_evidence_ready"
        elif (
            recovery_valid
            and bounded_autonomy_valid
            and dual_run_valid
            and certification_passed
        ):
            readiness = "module5_shadow_certification_evidence_ready"
        elif recovery_valid and bounded_autonomy_valid and dual_run_valid:
            readiness = "module5_bounded_autonomy_dual_run_evidence_ready"
        elif recovery_valid and bounded_autonomy_valid:
            readiness = "module5_bounded_autonomy_shadow_evidence_ready"
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

        components = {
            "module_5_1_minimized_execution_evidence": execution_valid,
            "module_5_2_supervisor_policy_evaluation": policy_valid,
            "module_5_3_circuit_breaker_governance": circuit_valid,
            "module_5_4_human_shadow_review": review_valid,
            "module_5_5_recovery_certification_planning": recovery_valid,
        }
        if bounded_autonomy_valid or bounded_autonomy_enabled:
            components["module_5_6_bounded_autonomy_shadow_kernel"] = (
                bounded_autonomy_valid
            )
        if dual_run_valid or dual_run_enabled:
            components["module_5_7_real_service_dual_run_comparison"] = dual_run_valid
        if certification_valid or certification_enabled:
            components["module_5_8_repeated_shadow_certification"] = (
                certification_passed
            )
        if functional_shadow_valid or functional_shadow_enabled:
            components["module_5_9_functional_shadow_execution"] = (
                functional_shadow_valid
            )
        if agent_security_valid or agent_security_enabled:
            components["module_5_10_agent_security_authorization"] = (
                agent_security_valid
            )
        if scale_recovery_valid or scale_recovery_enabled:
            components["module_5_11_scale_latency_recovery_certification"] = (
                scale_recovery_valid
            )
            components["module_5_12_real_historical_scale_adapter"] = (
                scale_recovery_valid
            )

        return {
            "module": "module_5_governed_agent_coordination",
            "status": readiness,
            "components": components,
            "module5_1_engineering_evidence_ready": execution_valid,
            "module5_2_engineering_evidence_ready": policy_valid,
            "module5_3_engineering_evidence_ready": circuit_valid,
            "module5_4_engineering_evidence_ready": review_valid,
            "module5_engineering_evidence_ready": recovery_valid,
            "module5_bounded_autonomy_shadow_evidence_ready": (bounded_autonomy_valid),
            "module5_bounded_autonomy_dual_run_evidence_ready": (dual_run_valid),
            "module5_bounded_autonomy_shadow_certified": certification_passed,
            "module5_bounded_autonomy_functional_shadow_ready": (
                functional_shadow_valid
            ),
            "module5_agent_security_authorization_ready": agent_security_valid,
            "module5_scale_latency_recovery_ready": scale_recovery_valid,
            "module5_real_historical_scale_adapter_ready": (
                scale_recovery_valid
            ),
            "live_production_certified": False,
            "evidence_presence": {
                "execution": execution is not None,
                "policy": policy is not None,
                "circuit_breaker": circuit is not None,
                "human_review": review is not None,
                "recovery": recovery is not None,
                "bounded_autonomy": bounded_autonomy is not None,
                "bounded_autonomy_dual_run": dual_run_comparison is not None,
                "bounded_autonomy_certification": (shadow_certification is not None),
                "bounded_autonomy_functional_shadow": (functional_shadow is not None),
                "security_posture": security_posture is not None,
                "agent_security_certification": agent_security is not None,
                "scale_recovery_certification": scale_recovery is not None,
            },
            "feature_flags": {
                "governed_execution_enabled": execution_enabled,
                "policy_evaluation_enabled": policy_enabled,
                "circuit_breaker_enabled": circuit_enabled,
                "shadow_review_enabled": shadow_enabled,
                "autonomous_mutation_enabled": mutation_enabled,
                "production_routing_enabled": routing_enabled,
                "bounded_autonomy_shadow_enabled": bounded_autonomy_enabled,
                "bounded_autonomy_dual_run_enabled": dual_run_enabled,
                "bounded_autonomy_certification_enabled": (certification_enabled),
                "bounded_autonomy_functional_shadow_enabled": (
                    functional_shadow_enabled
                ),
                "agent_security_certification_enabled": agent_security_enabled,
                "scale_recovery_certification_enabled": scale_recovery_enabled,
                "agent_production_effect_authorization_enabled": (
                    production_effect_authorization_enabled
                ),
                "scale_recovery_production_cutover_enabled": (
                    scale_recovery_cutover_enabled
                ),
            },
            "safety": {
                "raw_identifiers_exposed": False,
                "prompt_content_exposed": False,
                "tool_arguments_exposed": False,
                "automatic_mutation_performed": False,
                "automatic_approval_performed": False,
                "agent_approval_authority": False,
                "agent_delivery_authority": False,
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
