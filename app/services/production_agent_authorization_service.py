from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from app.models.production_agent_security_contracts import (
    AGENT_AUTHORIZATION_POLICY_VERSION,
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
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)
from app.services.production_bounded_autonomy_service import (
    CapabilityHandler,
    CapabilityRegistry,
)

_ACTION_SCOPES = {
    "read": "audience:read",
    "propose": "audience:propose",
    "approve": "audience:approve",
    "deliver": "audience:deliver",
}


def authorization_action_for(capability: CapabilityDescriptor) -> str:
    if capability.risk_class == "read_only":
        return "read"
    if capability.risk_class == "review_only":
        return "propose"
    return "deliver"


class ProductionAgentCapabilityAuthorizationService:
    """Authorize one capability with tenant, scope and duty separation."""

    def __init__(self, *, clock=time.time) -> None:
        self._clock = clock

    def authorize(
        self,
        request: AgentCapabilityAuthorizationRequest,
        *,
        evaluated_at_epoch_seconds: int | None = None,
    ) -> dict[str, Any]:
        now = int(
            evaluated_at_epoch_seconds
            if evaluated_at_epoch_seconds is not None
            else self._clock()
        )
        required_scope = _ACTION_SCOPES[request.action]
        ordered_reasons = self._authorization_reasons(
            request=request,
            evaluated_at_epoch_seconds=now,
        )
        decision = {
            "status": "allowed" if not ordered_reasons else "denied",
            "policy_version": AGENT_AUTHORIZATION_POLICY_VERSION,
            "request": request.to_record(),
            "evaluated_at_epoch_seconds": now,
            "required_scope": required_scope,
            "reason_codes": (
                ["least_privilege_authorization_satisfied"]
                if not ordered_reasons
                else list(ordered_reasons)
            ),
            "safety": {
                "credential_material_stored": False,
                "authentication_token_stored": False,
                "raw_identifiers_returned": False,
                "cross_tenant_access_performed": False,
                "automatic_approval_performed": False,
                "production_effect_performed": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
            },
        }
        decision["authorization_decision_fingerprint"] = stable_fingerprint(
            decision
        )
        return self.validate_decision(decision)

    def validate_decision(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(decision)))
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") not in {"allowed", "denied"}:
            raise ValueError("Unsupported agent authorization status.")
        if payload.get("policy_version") != AGENT_AUTHORIZATION_POLICY_VERSION:
            raise ValueError("Unsupported agent authorization policy version.")
        request = self._request_from_record(payload.get("request"))
        evaluated = int(payload.get("evaluated_at_epoch_seconds") or 0)
        if evaluated < 1:
            raise ValueError("Authorization evaluation time is required.")
        if payload.get("required_scope") != _ACTION_SCOPES[request.action]:
            raise ValueError("Authorization required scope is inconsistent.")
        reasons = payload.get("reason_codes")
        if not isinstance(reasons, list) or not reasons:
            raise ValueError("Authorization reason codes are required.")
        if len(reasons) != len(set(reasons)):
            raise ValueError("Authorization reason codes must be unique.")
        denied_reasons = self._authorization_reasons(
            request=request,
            evaluated_at_epoch_seconds=evaluated,
        )
        expected_reasons = (
            ["least_privilege_authorization_satisfied"]
            if not denied_reasons
            else list(denied_reasons)
        )
        if reasons != expected_reasons:
            raise ValueError("Authorization reason codes are inconsistent.")
        expected_allowed = not denied_reasons
        if (payload.get("status") == "allowed") is not expected_allowed:
            raise ValueError("Authorization status and reason codes conflict.")
        safety = payload.get("safety")
        if not isinstance(safety, Mapping):
            raise TypeError("Authorization safety evidence is required.")
        for field in (
            "credential_material_stored",
            "authentication_token_stored",
            "raw_identifiers_returned",
            "cross_tenant_access_performed",
            "automatic_approval_performed",
            "production_effect_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe authorization field: {field}.")
        supplied = payload.pop("authorization_decision_fingerprint", None)
        if stable_fingerprint(payload) != supplied:
            raise ValueError("Authorization decision fingerprint mismatch.")
        payload["authorization_decision_fingerprint"] = required_sha256_digest(
            supplied,
            label="authorization_decision_fingerprint",
        )
        return payload

    def _authorization_reasons(
        self,
        *,
        request: AgentCapabilityAuthorizationRequest,
        evaluated_at_epoch_seconds: int,
    ) -> tuple[str, ...]:
        context = request.authorization_context
        principal = context.principal
        reasons: list[str] = []
        if context.request_id != request.request_id:
            reasons.append("request_identity_mismatch")
        if principal.tenant_id != request.tenant_id:
            reasons.append("cross_tenant_capability_denied")
        if evaluated_at_epoch_seconds < principal.issued_at_epoch_seconds:
            reasons.append("principal_not_yet_valid")
        if evaluated_at_epoch_seconds >= principal.expires_at_epoch_seconds:
            reasons.append("principal_expired")
        if request.capability_id not in principal.allowed_capability_ids:
            reasons.append("capability_not_allowlisted")
        permitted_risk_actions = {
            "read_only": {"read"},
            "review_only": {"propose", "approve"},
            "production_effect": {"deliver"},
        }[request.risk_class]
        if request.action not in permitted_risk_actions:
            reasons.append("capability_action_risk_mismatch")
        required_scope = _ACTION_SCOPES[request.action]
        if required_scope not in principal.scopes:
            reasons.append("required_scope_missing")
        if request.execution_mode != "production" and (
            request.action in {"approve", "deliver"}
            or request.risk_class == "production_effect"
        ):
            reasons.append("nonproduction_effect_authority_denied")
        permitted_actions = {
            "bounded_agent": {"read", "propose"},
            "system_worker": {"read", "propose"},
            "human_reviewer": {"read", "approve"},
            "delivery_service": {"read", "deliver"},
        }[principal.principal_type]
        if request.action not in permitted_actions:
            reasons.append(f"{principal.principal_type}_action_denied")
        if principal.principal_type == "bounded_agent" and request.action in {
            "approve",
            "deliver",
        }:
            reasons.append("agent_cannot_approve_or_deliver")
        if principal.principal_type == "bounded_agent" and (
            request.risk_class == "production_effect"
        ):
            reasons.append("agent_production_effect_denied")
        if request.action == "deliver":
            reasons.extend(
                self._delivery_chain_reasons(
                    request=request,
                    evaluated_at_epoch_seconds=evaluated_at_epoch_seconds,
                )
            )
        return tuple(dict.fromkeys(sorted(reasons)))

    def _delivery_chain_reasons(
        self,
        *,
        request: AgentCapabilityAuthorizationRequest,
        evaluated_at_epoch_seconds: int,
    ) -> list[str]:
        chain = request.approval_chain
        principal = request.authorization_context.principal
        if chain is None:
            return ["approved_delivery_chain_required"]
        reasons = []
        if chain.tenant_id != request.tenant_id:
            reasons.append("approval_chain_tenant_mismatch")
        if chain.approval_status != "approved_for_delivery":
            reasons.append("delivery_approval_not_granted")
        if evaluated_at_epoch_seconds >= chain.expires_at_epoch_seconds:
            reasons.append("delivery_approval_expired")
        if chain.delivery_principal_id != principal.principal_id:
            reasons.append("delivery_principal_mismatch")
        actors = {
            chain.proposer_principal_id,
            chain.approver_principal_id,
            chain.delivery_principal_id,
        }
        if len(actors) != 3:
            reasons.append("separation_of_duties_violated")
        return reasons

    def _request_from_record(
        self,
        value: Any,
    ) -> AgentCapabilityAuthorizationRequest:
        if not isinstance(value, Mapping):
            raise TypeError("Authorization request evidence is required.")
        context_value = value.get("authorization_context")
        if not isinstance(context_value, Mapping):
            raise TypeError("Authorization context evidence is required.")
        principal_value = context_value.get("principal")
        if not isinstance(principal_value, Mapping):
            raise TypeError("Authorization principal evidence is required.")
        context = AgentAuthorizationContext(
            authorization_id=context_value.get("authorization_id"),
            request_id=context_value.get("request_id"),
            source_authentication_fingerprint=context_value.get(
                "source_authentication_fingerprint"
            ),
            principal=AgentSecurityPrincipal(
                **dict(principal_value),
            ),
        )
        chain_value = value.get("approval_chain")
        chain = (
            AgentApprovalChain(**dict(chain_value))
            if isinstance(chain_value, Mapping)
            else None
        )
        return AgentCapabilityAuthorizationRequest(
            tenant_id=value.get("tenant_id"),
            request_id=value.get("request_id"),
            capability_id=value.get("capability_id"),
            module_id=value.get("module_id"),
            risk_class=value.get("risk_class"),
            execution_mode=value.get("execution_mode"),
            action=value.get("action"),
            authorization_context=context,
            approval_chain=chain,
        )


class AgentCapabilityAuthorizationEnforcer:
    """Wrap real handlers and deny unauthorized invocations before execution."""

    def __init__(
        self,
        *,
        goal: AutonomyGoal,
        authorization_context: AgentAuthorizationContext,
        registry: CapabilityRegistry,
        service: ProductionAgentCapabilityAuthorizationService | None = None,
        evaluated_at_epoch_seconds: int | None = None,
    ) -> None:
        self.goal = goal
        self.authorization_context = authorization_context
        self.registry = registry
        self.service = service or ProductionAgentCapabilityAuthorizationService()
        self.evaluated_at_epoch_seconds = evaluated_at_epoch_seconds
        self.decisions: list[dict[str, Any]] = []

    def wrap(
        self,
        handlers: Mapping[str, CapabilityHandler],
    ) -> dict[str, CapabilityHandler]:
        wrapped = {}
        for capability_id, handler in handlers.items():
            capability = self.registry.get(capability_id)
            wrapped[capability_id] = self._wrapped_handler(capability, handler)
        return wrapped

    def minimized_evidence(self) -> dict[str, Any]:
        denied = sum(value["status"] == "denied" for value in self.decisions)
        return {
            "policy_version": AGENT_AUTHORIZATION_POLICY_VERSION,
            "principal_fingerprint": (
                self.authorization_context.principal.principal_fingerprint
            ),
            "authorization_context_fingerprint": (
                self.authorization_context.context_fingerprint
            ),
            "decision_count": len(self.decisions),
            "allowed_decision_count": len(self.decisions) - denied,
            "denied_decision_count": denied,
            "decision_fingerprints": [
                value["authorization_decision_fingerprint"]
                for value in self.decisions
            ],
            "credential_material_stored": False,
            "authentication_token_stored": False,
        }

    def _wrapped_handler(
        self,
        capability: CapabilityDescriptor,
        handler: CapabilityHandler,
    ) -> CapabilityHandler:
        def authorized(
            goal: AutonomyGoal,
            runtime_capability: CapabilityDescriptor,
            invocation: Mapping[str, Any],
        ) -> CapabilityExecutionResult:
            if runtime_capability.capability_id != capability.capability_id:
                return CapabilityExecutionResult(
                    status="blocked",
                    reason_codes=("authorization_descriptor_mismatch",),
                    error_category="authorization_denied",
                )
            request = AgentCapabilityAuthorizationRequest(
                tenant_id=goal.tenant_id,
                request_id=goal.request_id,
                capability_id=capability.capability_id,
                module_id=capability.module_id,
                risk_class=capability.risk_class,
                execution_mode=goal.execution_mode,
                action=authorization_action_for(capability),
                authorization_context=self.authorization_context,
            )
            decision = self.service.authorize(
                request,
                evaluated_at_epoch_seconds=self.evaluated_at_epoch_seconds,
            )
            self.decisions.append(decision)
            if decision["status"] != "allowed":
                return CapabilityExecutionResult(
                    status="blocked",
                    reason_codes=("capability_authorization_denied",),
                    metrics={"authorization_allowed": False},
                    error_category="authorization_denied",
                )
            return handler(goal, runtime_capability, invocation)

        return authorized
