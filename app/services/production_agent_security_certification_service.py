from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.models.production_agent_security_contracts import (
    AGENT_SECURITY_CERTIFICATION_VERSION,
    AgentApprovalChain,
    AgentAuthorizationContext,
    AgentCapabilityAuthorizationRequest,
    AgentSecurityCertificationRequest,
    AgentSecurityPrincipal,
)
from app.models.production_bounded_autonomy_contracts import CapabilityDescriptor
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)
from app.services.production_agent_authorization_service import (
    ProductionAgentCapabilityAuthorizationService,
)
from app.services.production_bounded_autonomy_functional_shadow_service import (
    ProductionBoundedAutonomyFunctionalShadowService,
)
from app.services.production_security_hardening_service import (
    ProductionSecurityPostureService,
)


class ProductionAgentSecurityCertificationService:
    """Certify least privilege and duty separation using adversarial cases."""

    def __init__(
        self,
        *,
        authorization_service: (
            ProductionAgentCapabilityAuthorizationService | None
        ) = None,
    ) -> None:
        self.authorization_service = (
            authorization_service
            or ProductionAgentCapabilityAuthorizationService()
        )

    def certify(
        self,
        *,
        request: AgentSecurityCertificationRequest,
        functional_shadow_report: Mapping[str, Any],
        security_posture_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        functional = ProductionBoundedAutonomyFunctionalShadowService(
            environment={}
        ).validate_report(functional_shadow_report)
        posture = ProductionSecurityPostureService().validate_report(
            security_posture_report
        )
        self._validate_lineage(
            request=request,
            functional=functional,
            posture=posture,
        )
        scenarios = self._evaluate_scenarios(request)
        failed = [
            value["scenario"] for value in scenarios if value["passed"] is False
        ]
        control_families = {
            value["control_family"] for value in scenarios
        }
        required_families = {
            "allowlist",
            "expiry",
            "least_privilege",
            "mode_boundary",
            "principal_role",
            "separation_of_duties",
            "tenant_boundary",
        }
        positive = sum(
            value["expected_status"] == "allowed" for value in scenarios
        )
        negative = len(scenarios) - positive
        certification_gates = {
            "all_scenarios_passed": {
                "passed": not failed,
                "failed_scenario_count": len(failed),
            },
            "positive_authorization_coverage": {
                "passed": positive >= 3,
                "observed": positive,
                "minimum": 3,
            },
            "negative_authorization_coverage": {
                "passed": negative >= 8,
                "observed": negative,
                "minimum": 8,
            },
            "required_control_families": {
                "passed": required_families.issubset(control_families),
                "observed_count": len(control_families),
                "required_count": len(required_families),
            },
            "functional_authorization_enforced": {
                "passed": functional.get("safety", {}).get(
                    "capability_authorization_enforced"
                )
                is True,
            },
            "security_posture_passed": {
                "passed": posture.get("posture_status") == "pass",
            },
        }
        eligible = all(
            value["passed"] is True for value in certification_gates.values()
        )
        report = {
            "status": (
                "engineering_preview_ready"
                if eligible
                else "engineering_preview_blocked"
            ),
            "policy_version": AGENT_SECURITY_CERTIFICATION_VERSION,
            "request": request.to_record(),
            "lineage": {
                "source_functional_shadow_report_fingerprint": functional[
                    "functional_shadow_report_fingerprint"
                ],
                "source_security_posture_fingerprint": posture[
                    "security_posture_fingerprint"
                ],
            },
            "summary": {
                "scenario_count": len(scenarios),
                "passed_scenario_count": len(scenarios) - len(failed),
                "failed_scenario_count": len(failed),
                "positive_authorization_case_count": positive,
                "negative_authorization_case_count": negative,
                "control_family_count": len(control_families),
                "failed_scenario_codes": failed,
            },
            "scenarios": scenarios,
            "certification_gates": certification_gates,
            "review": {
                "eligible_for_staging_review": eligible,
                "live_cutover_authorized": False,
                "fresh_data_certified": False,
                "production_identity_provider_certified": False,
                "external_penetration_test_completed": False,
                "automatic_cutover_performed": False,
            },
            "safety": {
                "credentials_stored": False,
                "authentication_tokens_stored": False,
                "prompt_content_stored": False,
                "tool_arguments_stored": False,
                "raw_identifiers_returned": False,
                "cross_tenant_access_performed": False,
                "automatic_approval_performed": False,
                "production_effect_performed": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "manual_approval_required": True,
            },
        }
        report["agent_security_certification_fingerprint"] = stable_fingerprint(
            report
        )
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(report)))
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") not in {
            "engineering_preview_ready",
            "engineering_preview_blocked",
        }:
            raise ValueError("Unsupported agent security certification status.")
        if payload.get("policy_version") != AGENT_SECURITY_CERTIFICATION_VERSION:
            raise ValueError("Unsupported agent security certification policy.")
        request = AgentSecurityCertificationRequest(
            **dict(payload.get("request") or {})
        )
        lineage = payload.get("lineage")
        summary = payload.get("summary")
        scenarios = payload.get("scenarios")
        gates = payload.get("certification_gates")
        review = payload.get("review")
        safety = payload.get("safety")
        if not all(
            isinstance(value, Mapping)
            for value in (lineage, summary, gates, review, safety)
        ) or not isinstance(scenarios, list):
            raise TypeError("Agent security certification sections are required.")
        if lineage.get("source_functional_shadow_report_fingerprint") != (
            request.source_functional_shadow_report_fingerprint
        ) or lineage.get("source_security_posture_fingerprint") != (
            request.source_security_posture_fingerprint
        ):
            raise ValueError("Agent security certification lineage mismatch.")
        for value in lineage.values():
            required_sha256_digest(value, label="agent_security_lineage")
        scenario_codes = []
        failed = []
        positive = 0
        for value in scenarios:
            if not isinstance(value, Mapping):
                raise TypeError("Agent security scenario evidence is invalid.")
            scenario = required_slug(value.get("scenario"), label="scenario")
            required_slug(
                value.get("control_family"),
                label="control_family",
            )
            if scenario in scenario_codes:
                raise ValueError("Agent security scenarios must be unique.")
            scenario_codes.append(scenario)
            if value.get("expected_status") not in {"allowed", "denied"} or (
                value.get("observed_status") not in {"allowed", "denied"}
            ):
                raise ValueError("Agent security scenario status is invalid.")
            if value.get("passed") is not (
                value.get("expected_status") == value.get("observed_status")
            ):
                raise ValueError("Agent security scenario result is inconsistent.")
            if value.get("expected_status") == "allowed":
                positive += 1
            if value.get("passed") is False:
                failed.append(scenario)
            required_sha256_digest(
                value.get("authorization_decision_fingerprint"),
                label="authorization_decision_fingerprint",
            )
            scenario_payload = dict(value)
            supplied_scenario = scenario_payload.pop("scenario_fingerprint", None)
            if stable_fingerprint(scenario_payload) != supplied_scenario:
                raise ValueError("Agent security scenario fingerprint mismatch.")
        expected_summary = {
            "scenario_count": len(scenarios),
            "passed_scenario_count": len(scenarios) - len(failed),
            "failed_scenario_count": len(failed),
            "positive_authorization_case_count": positive,
            "negative_authorization_case_count": len(scenarios) - positive,
            "control_family_count": len(
                {value.get("control_family") for value in scenarios}
            ),
            "failed_scenario_codes": failed,
        }
        if dict(summary) != expected_summary:
            raise ValueError("Agent security certification summary mismatch.")
        observed_families = {
            value.get("control_family") for value in scenarios
        }
        required_families = {
            "allowlist",
            "expiry",
            "least_privilege",
            "mode_boundary",
            "principal_role",
            "separation_of_duties",
            "tenant_boundary",
        }
        expected_gates = {
            "all_scenarios_passed": {
                "passed": not failed,
                "failed_scenario_count": len(failed),
            },
            "positive_authorization_coverage": {
                "passed": positive >= 3,
                "observed": positive,
                "minimum": 3,
            },
            "negative_authorization_coverage": {
                "passed": len(scenarios) - positive >= 8,
                "observed": len(scenarios) - positive,
                "minimum": 8,
            },
            "required_control_families": {
                "passed": required_families.issubset(observed_families),
                "observed_count": len(observed_families),
                "required_count": len(required_families),
            },
            "functional_authorization_enforced": {"passed": True},
            "security_posture_passed": {"passed": True},
        }
        if dict(gates) != expected_gates:
            raise ValueError("Agent security certification gates are inconsistent.")
        eligible = all(value.get("passed") is True for value in gates.values())
        if review.get("eligible_for_staging_review") is not eligible:
            raise ValueError("Agent security staging-review result is inconsistent.")
        if payload.get("status") != (
            "engineering_preview_ready"
            if eligible
            else "engineering_preview_blocked"
        ):
            raise ValueError("Agent security certification status is inconsistent.")
        for field in (
            "live_cutover_authorized",
            "fresh_data_certified",
            "production_identity_provider_certified",
            "external_penetration_test_completed",
            "automatic_cutover_performed",
        ):
            if review.get(field) is not False:
                raise ValueError(f"Agent security review cannot set {field}.")
        for field in (
            "credentials_stored",
            "authentication_tokens_stored",
            "prompt_content_stored",
            "tool_arguments_stored",
            "raw_identifiers_returned",
            "cross_tenant_access_performed",
            "automatic_approval_performed",
            "production_effect_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe agent security field: {field}.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Agent security certification requires manual approval.")
        supplied = payload.pop("agent_security_certification_fingerprint", None)
        if stable_fingerprint(payload) != supplied:
            raise ValueError("Agent security certification fingerprint mismatch.")
        payload["agent_security_certification_fingerprint"] = (
            required_sha256_digest(
                supplied,
                label="agent_security_certification_fingerprint",
            )
        )
        return payload

    def _validate_lineage(
        self,
        *,
        request: AgentSecurityCertificationRequest,
        functional: Mapping[str, Any],
        posture: Mapping[str, Any],
    ) -> None:
        if functional.get("request", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Functional shadow tenant lineage mismatch.")
        if posture.get("request", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Security posture tenant lineage mismatch.")
        if functional.get("functional_shadow_report_fingerprint") != (
            request.source_functional_shadow_report_fingerprint
        ):
            raise ValueError("Functional shadow fingerprint lineage mismatch.")
        if posture.get("security_posture_fingerprint") != (
            request.source_security_posture_fingerprint
        ):
            raise ValueError("Security posture fingerprint lineage mismatch.")
        if functional.get("status") != "engineering_preview_ready" or (
            functional.get("review", {}).get("eligible_for_staging_review")
            is not True
        ):
            raise ValueError("Functional shadow is not ready for certification.")
        if functional.get("safety", {}).get(
            "capability_authorization_enforced"
        ) is not True:
            raise ValueError("Functional capability authorization is not enforced.")
        if posture.get("posture_status") != "pass":
            raise ValueError("Security posture must pass before agent certification.")

    def _evaluate_scenarios(
        self,
        request: AgentSecurityCertificationRequest,
    ) -> list[dict[str, Any]]:
        now = request.evaluation_epoch_seconds
        tenant = request.tenant_id
        read_capability = CapabilityDescriptor(
            capability_id="module2_semantic_retrieval",
            module_id=2,
            description="Governed read-only retrieval authorization probe.",
            requires=(),
            provides=("retrieval_evidence",),
            risk_class="read_only",
        )
        review_capability = CapabilityDescriptor(
            capability_id="module3_cohort_strategy",
            module_id=3,
            description="Governed review-only proposal authorization probe.",
            requires=(),
            provides=("audience_candidates",),
            risk_class="review_only",
        )
        delivery_capability = CapabilityDescriptor(
            capability_id="module5_approved_delivery_execution",
            module_id=5,
            description="Synthetic authorization-only delivery probe.",
            requires=(),
            provides=("delivery_receipt",),
            allowed_modes=("production",),
            risk_class="production_effect",
        )

        cases: list[tuple[str, str, str, AgentCapabilityAuthorizationRequest]] = []

        def principal(
            *,
            principal_id: str = "bounded-agent-certification",
            principal_type: str = "bounded_agent",
            tenant_id: str = tenant,
            scopes: tuple[str, ...] = ("audience:read", "audience:propose"),
            capabilities: tuple[str, ...] = (
                read_capability.capability_id,
                review_capability.capability_id,
            ),
            issued_at: int = now - 10,
            expires_at: int = now + 300,
        ) -> AgentSecurityPrincipal:
            return AgentSecurityPrincipal(
                tenant_id=tenant_id,
                principal_id=principal_id,
                principal_type=principal_type,
                authentication_method="workload_identity",
                scopes=scopes,
                allowed_capability_ids=capabilities,
                issued_at_epoch_seconds=issued_at,
                expires_at_epoch_seconds=expires_at,
            )

        def context(value: AgentSecurityPrincipal) -> AgentAuthorizationContext:
            return AgentAuthorizationContext(
                authorization_id=f"auth-{value.principal_id}",
                request_id="agent-security-certification-request",
                source_authentication_fingerprint=stable_fingerprint({
                    "tenant_id": value.tenant_id,
                    "principal_id": value.principal_id,
                    "authentication_method": value.authentication_method,
                    "certification_id": request.certification_id,
                }),
                principal=value,
            )

        def auth_request(
            *,
            capability: CapabilityDescriptor,
            principal_value: AgentSecurityPrincipal,
            action: str,
            execution_mode: str = "shadow",
            approval_chain: AgentApprovalChain | None = None,
        ) -> AgentCapabilityAuthorizationRequest:
            return AgentCapabilityAuthorizationRequest(
                tenant_id=tenant,
                request_id="agent-security-certification-request",
                capability_id=capability.capability_id,
                module_id=capability.module_id,
                risk_class=capability.risk_class,
                execution_mode=execution_mode,
                action=action,
                authorization_context=context(principal_value),
                approval_chain=approval_chain,
            )

        cases.extend([
            (
                "valid_bounded_agent_read",
                "least_privilege",
                "allowed",
                auth_request(
                    capability=read_capability,
                    principal_value=principal(),
                    action="read",
                ),
            ),
            (
                "valid_bounded_agent_proposal",
                "principal_role",
                "allowed",
                auth_request(
                    capability=review_capability,
                    principal_value=principal(),
                    action="propose",
                ),
            ),
            (
                "valid_human_reviewer_approval_authorization",
                "separation_of_duties",
                "allowed",
                auth_request(
                    capability=review_capability,
                    principal_value=principal(
                        principal_id="human-reviewer-certification",
                        principal_type="human_reviewer",
                        scopes=("audience:approve",),
                        capabilities=(review_capability.capability_id,),
                    ),
                    action="approve",
                    execution_mode="production",
                ),
            ),
            (
                "cross_tenant_principal",
                "tenant_boundary",
                "denied",
                auth_request(
                    capability=read_capability,
                    principal_value=principal(tenant_id="tenant_other"),
                    action="read",
                ),
            ),
            (
                "missing_required_scope",
                "least_privilege",
                "denied",
                auth_request(
                    capability=read_capability,
                    principal_value=principal(scopes=("audience:propose",)),
                    action="read",
                ),
            ),
            (
                "capability_not_allowlisted",
                "allowlist",
                "denied",
                auth_request(
                    capability=read_capability,
                    principal_value=principal(
                        capabilities=(review_capability.capability_id,)
                    ),
                    action="read",
                ),
            ),
            (
                "expired_agent_identity",
                "expiry",
                "denied",
                auth_request(
                    capability=read_capability,
                    principal_value=principal(
                        issued_at=now - 600,
                        expires_at=now - 1,
                    ),
                    action="read",
                ),
            ),
            (
                "agent_approval_attempt",
                "principal_role",
                "denied",
                auth_request(
                    capability=review_capability,
                    principal_value=principal(
                        scopes=("audience:approve",),
                    ),
                    action="approve",
                ),
            ),
        ])

        valid_chain = AgentApprovalChain(
            tenant_id=tenant,
            proposal_id="certification-proposal",
            proposer_principal_id="bounded-agent-certification",
            approver_principal_id="human-reviewer-certification",
            delivery_principal_id="delivery-service-certification",
            approval_evidence_fingerprint="d" * 64,
            approval_status="approved_for_delivery",
            expires_at_epoch_seconds=now + 120,
        )
        agent_delivery_principal = principal(
            scopes=("audience:deliver",),
            capabilities=(delivery_capability.capability_id,),
        )
        delivery_principal = principal(
            principal_id="delivery-service-certification",
            principal_type="delivery_service",
            scopes=("audience:deliver",),
            capabilities=(delivery_capability.capability_id,),
        )
        cases.extend([
            (
                "bounded_agent_delivery_attempt",
                "principal_role",
                "denied",
                auth_request(
                    capability=delivery_capability,
                    principal_value=agent_delivery_principal,
                    action="deliver",
                    execution_mode="production",
                    approval_chain=AgentApprovalChain(
                        **{
                            **valid_chain.to_record(),
                            "delivery_principal_id": (
                                agent_delivery_principal.principal_id
                            ),
                        }
                    ),
                ),
            ),
            (
                "shadow_production_effect_attempt",
                "mode_boundary",
                "denied",
                auth_request(
                    capability=delivery_capability,
                    principal_value=delivery_principal,
                    action="deliver",
                    execution_mode="shadow",
                    approval_chain=valid_chain,
                ),
            ),
            (
                "delivery_without_approval",
                "separation_of_duties",
                "denied",
                auth_request(
                    capability=delivery_capability,
                    principal_value=delivery_principal,
                    action="deliver",
                    execution_mode="production",
                ),
            ),
            (
                "self_approved_delivery",
                "separation_of_duties",
                "denied",
                auth_request(
                    capability=delivery_capability,
                    principal_value=delivery_principal,
                    action="deliver",
                    execution_mode="production",
                    approval_chain=AgentApprovalChain(
                        **{
                            **valid_chain.to_record(),
                            "approver_principal_id": (
                                valid_chain.proposer_principal_id
                            ),
                        }
                    ),
                ),
            ),
            (
                "valid_separated_delivery_authorization",
                "separation_of_duties",
                "allowed",
                auth_request(
                    capability=delivery_capability,
                    principal_value=delivery_principal,
                    action="deliver",
                    execution_mode="production",
                    approval_chain=valid_chain,
                ),
            ),
            (
                "human_reviewer_proposal_attempt",
                "principal_role",
                "denied",
                auth_request(
                    capability=review_capability,
                    principal_value=principal(
                        principal_id="human-reviewer-certification",
                        principal_type="human_reviewer",
                        scopes=("audience:propose",),
                        capabilities=(review_capability.capability_id,),
                    ),
                    action="propose",
                ),
            ),
        ])

        evidence = []
        for scenario, family, expected, authorization_request in cases:
            decision = self.authorization_service.authorize(
                authorization_request,
                evaluated_at_epoch_seconds=now,
            )
            value = {
                "scenario": scenario,
                "control_family": family,
                "expected_status": expected,
                "observed_status": decision["status"],
                "passed": decision["status"] == expected,
                "reason_codes": list(decision["reason_codes"]),
                "authorization_decision_fingerprint": decision[
                    "authorization_decision_fingerprint"
                ],
            }
            value["scenario_fingerprint"] = stable_fingerprint(value)
            evidence.append(value)
        return evidence
