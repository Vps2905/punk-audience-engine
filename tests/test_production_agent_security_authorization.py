from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_agent_security_contracts import (
    AgentApprovalChain,
    AgentAuthorizationContext,
    AgentCapabilityAuthorizationRequest,
    AgentSecurityPrincipal,
)
from app.models.production_bounded_autonomy_contracts import (
    AutonomyGoal,
    CapabilityDescriptor,
    CapabilityExecutionResult,
)
from app.models.production_module3_cohort_contracts import stable_fingerprint
from app.services.production_agent_authorization_service import (
    AgentCapabilityAuthorizationEnforcer,
    ProductionAgentCapabilityAuthorizationService,
)
from app.services.production_bounded_autonomy_service import CapabilityRegistry

TENANT = "tenant_a"
NOW = 1_800_000_000
READ_CAPABILITY = CapabilityDescriptor(
    capability_id="module2_semantic_retrieval",
    module_id=2,
    description="Read-only authorization test capability.",
    requires=(),
    provides=("retrieval_evidence",),
    risk_class="read_only",
)
DELIVERY_CAPABILITY = CapabilityDescriptor(
    capability_id="module5_approved_delivery_execution",
    module_id=5,
    description="Authorization-only delivery test capability.",
    requires=(),
    provides=("delivery_receipt",),
    allowed_modes=("production",),
    risk_class="production_effect",
)


def principal(
    *,
    tenant_id=TENANT,
    principal_id="bounded-agent-1",
    principal_type="bounded_agent",
    scopes=("audience:read", "audience:propose"),
    capabilities=(READ_CAPABILITY.capability_id,),
    issued_at=NOW - 60,
    expires_at=NOW + 300,
):
    return AgentSecurityPrincipal(
        tenant_id=tenant_id,
        principal_id=principal_id,
        principal_type=principal_type,
        authentication_method="workload_identity",
        scopes=tuple(scopes),
        allowed_capability_ids=tuple(capabilities),
        issued_at_epoch_seconds=issued_at,
        expires_at_epoch_seconds=expires_at,
    )


def context(principal_value=None):
    value = principal_value or principal()
    return AgentAuthorizationContext(
        authorization_id=f"authorization-{value.principal_id}",
        request_id="request-1",
        source_authentication_fingerprint=stable_fingerprint(
            {
                "tenant_id": value.tenant_id,
                "principal_id": value.principal_id,
                "method": value.authentication_method,
            }
        ),
        principal=value,
    )


def authorization_request(
    *,
    capability=READ_CAPABILITY,
    principal_value=None,
    tenant_id=TENANT,
    action="read",
    mode="shadow",
    approval_chain=None,
):
    return AgentCapabilityAuthorizationRequest(
        tenant_id=tenant_id,
        request_id="request-1",
        capability_id=capability.capability_id,
        module_id=capability.module_id,
        risk_class=capability.risk_class,
        execution_mode=mode,
        action=action,
        authorization_context=context(principal_value),
        approval_chain=approval_chain,
    )


def test_short_lived_allowlisted_read_is_authorized_without_storing_credentials():
    decision = ProductionAgentCapabilityAuthorizationService().authorize(
        authorization_request(),
        evaluated_at_epoch_seconds=NOW,
    )

    assert decision["status"] == "allowed"
    assert decision["reason_codes"] == [
        "least_privilege_authorization_satisfied"
    ]
    assert decision["safety"]["credential_material_stored"] is False
    assert decision["safety"]["production_effect_performed"] is False


@pytest.mark.parametrize(
    ("principal_value", "reason_code"),
    [
        (principal(tenant_id="tenant_other"), "cross_tenant_capability_denied"),
        (principal(scopes=("audience:propose",)), "required_scope_missing"),
        (
            principal(capabilities=(DELIVERY_CAPABILITY.capability_id,)),
            "capability_not_allowlisted",
        ),
        (
            principal(issued_at=NOW - 600, expires_at=NOW - 1),
            "principal_expired",
        ),
    ],
)
def test_tenant_scope_allowlist_and_expiry_fail_closed(
    principal_value,
    reason_code,
):
    decision = ProductionAgentCapabilityAuthorizationService().authorize(
        authorization_request(principal_value=principal_value),
        evaluated_at_epoch_seconds=NOW,
    )

    assert decision["status"] == "denied"
    assert reason_code in decision["reason_codes"]


def test_forged_read_action_cannot_disguise_a_production_effect():
    delivery = principal(
        principal_id="delivery-service-1",
        principal_type="delivery_service",
        scopes=("audience:read",),
        capabilities=(DELIVERY_CAPABILITY.capability_id,),
    )
    decision = ProductionAgentCapabilityAuthorizationService().authorize(
        authorization_request(
            capability=DELIVERY_CAPABILITY,
            principal_value=delivery,
            action="read",
            mode="production",
        ),
        evaluated_at_epoch_seconds=NOW,
    )

    assert decision["status"] == "denied"
    assert "capability_action_risk_mismatch" in decision["reason_codes"]


def test_agent_cannot_self_approve_or_deliver_even_when_scoped():
    agent = principal(
        scopes=("audience:deliver",),
        capabilities=(DELIVERY_CAPABILITY.capability_id,),
    )
    chain = AgentApprovalChain(
        tenant_id=TENANT,
        proposal_id="proposal-1",
        proposer_principal_id=agent.principal_id,
        approver_principal_id=agent.principal_id,
        delivery_principal_id=agent.principal_id,
        approval_evidence_fingerprint="d" * 64,
        approval_status="approved_for_delivery",
        expires_at_epoch_seconds=NOW + 120,
    )
    decision = ProductionAgentCapabilityAuthorizationService().authorize(
        authorization_request(
            capability=DELIVERY_CAPABILITY,
            principal_value=agent,
            action="deliver",
            mode="production",
            approval_chain=chain,
        ),
        evaluated_at_epoch_seconds=NOW,
    )

    assert decision["status"] == "denied"
    assert "agent_cannot_approve_or_deliver" in decision["reason_codes"]
    assert "separation_of_duties_violated" in decision["reason_codes"]


def test_human_reviewer_can_receive_approval_authorization_without_auto_approval():
    reviewer = principal(
        principal_id="human-reviewer-1",
        principal_type="human_reviewer",
        scopes=("audience:approve",),
        capabilities=("module3_cohort_strategy",),
    )
    review_capability = CapabilityDescriptor(
        capability_id="module3_cohort_strategy",
        module_id=3,
        description="Review-only approval authorization test capability.",
        requires=(),
        provides=("audience_candidates",),
        risk_class="review_only",
    )
    decision = ProductionAgentCapabilityAuthorizationService().authorize(
        authorization_request(
            capability=review_capability,
            principal_value=reviewer,
            action="approve",
            mode="production",
        ),
        evaluated_at_epoch_seconds=NOW,
    )

    assert decision["status"] == "allowed"
    assert decision["safety"]["automatic_approval_performed"] is False


def test_separated_delivery_chain_can_be_authorized_but_performs_no_effect():
    delivery = principal(
        principal_id="delivery-service-1",
        principal_type="delivery_service",
        scopes=("audience:deliver",),
        capabilities=(DELIVERY_CAPABILITY.capability_id,),
    )
    chain = AgentApprovalChain(
        tenant_id=TENANT,
        proposal_id="proposal-1",
        proposer_principal_id="bounded-agent-1",
        approver_principal_id="human-reviewer-1",
        delivery_principal_id=delivery.principal_id,
        approval_evidence_fingerprint="d" * 64,
        approval_status="approved_for_delivery",
        expires_at_epoch_seconds=NOW + 120,
    )
    decision = ProductionAgentCapabilityAuthorizationService().authorize(
        authorization_request(
            capability=DELIVERY_CAPABILITY,
            principal_value=delivery,
            action="deliver",
            mode="production",
            approval_chain=chain,
        ),
        evaluated_at_epoch_seconds=NOW,
    )

    assert decision["status"] == "allowed"
    assert decision["safety"]["production_effect_performed"] is False
    assert decision["safety"]["activation_or_export_performed"] is False


def test_tampered_allowed_decision_cannot_validate_after_refingerprinting():
    service = ProductionAgentCapabilityAuthorizationService()
    denied = service.authorize(
        authorization_request(
            principal_value=principal(scopes=("audience:propose",))
        ),
        evaluated_at_epoch_seconds=NOW,
    )
    tampered = deepcopy(denied)
    tampered["status"] = "allowed"
    tampered["reason_codes"] = ["least_privilege_authorization_satisfied"]
    tampered.pop("authorization_decision_fingerprint")
    tampered["authorization_decision_fingerprint"] = stable_fingerprint(tampered)

    with pytest.raises(ValueError, match="reason codes are inconsistent"):
        service.validate_decision(tampered)


def test_enforcer_denies_before_calling_real_handler():
    called = []
    registry = CapabilityRegistry((READ_CAPABILITY,))
    goal = AutonomyGoal(
        tenant_id=TENANT,
        request_id="request-1",
        goal_id="goal-1",
        objective="Review privacy-safe retrieval evidence.",
        requested_outcomes=("retrieval_evidence",),
        execution_mode="shadow",
    )

    def handler(goal_value, capability_value, invocation):
        del goal_value, capability_value, invocation
        called.append(True)
        return CapabilityExecutionResult(status="completed")

    enforcer = AgentCapabilityAuthorizationEnforcer(
        goal=goal,
        authorization_context=context(
            principal(scopes=("audience:propose",))
        ),
        registry=registry,
        evaluated_at_epoch_seconds=NOW,
    )
    wrapped = enforcer.wrap({READ_CAPABILITY.capability_id: handler})
    result = wrapped[READ_CAPABILITY.capability_id](goal, READ_CAPABILITY, {})

    assert result.status == "blocked"
    assert result.error_category == "authorization_denied"
    assert called == []
    assert enforcer.minimized_evidence()["denied_decision_count"] == 1


def test_agent_security_migration_is_immutable_tenant_scoped_and_no_effects():
    sql = Path(
        "migrations/0032_module5_agent_security_authorization_certification.sql"
    ).read_text(encoding="utf-8")

    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "current_setting('app.tenant_id', true)" in sql
    assert "BEFORE UPDATE OR DELETE" in sql
    assert "REVOKE ALL" in sql
    assert "production_effect_performed = FALSE" in sql
    assert "activation_or_export_performed = FALSE" in sql
    assert "production_identity_provider_certified = FALSE" in sql
