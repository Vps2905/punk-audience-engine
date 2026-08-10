from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_bounded_autonomy_contracts import (
    ExecutionMode,
    RiskClass,
    required_metadata_token,
)
from app.models.production_module3_cohort_contracts import (
    required_slug,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)

AGENT_AUTHORIZATION_POLICY_VERSION = "module5_agent_authorization_v1"
AGENT_SECURITY_CERTIFICATION_VERSION = "module5_agent_security_certification_v1"

PrincipalType = Literal[
    "bounded_agent",
    "system_worker",
    "human_reviewer",
    "delivery_service",
]
AuthenticationMethod = Literal[
    "short_lived_token",
    "signed_service_identity",
    "workload_identity",
]
AuthorizationAction = Literal["read", "propose", "approve", "deliver"]


def _required_identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
    )
    if not text or len(text) > 128 or any(char not in allowed for char in text):
        raise ValueError(f"{label} must be a safe opaque identifier.")
    return text


def _unique_tokens(
    values: tuple[str, ...],
    *,
    label: str,
    maximum: int,
) -> tuple[str, ...]:
    normalized = tuple(
        required_metadata_token(value, label=label) for value in values
    )
    if len(normalized) > maximum:
        raise ValueError(f"{label} is bounded to {maximum} values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} values must be unique.")
    return normalized


@dataclass(frozen=True)
class AgentSecurityPrincipal:
    tenant_id: str
    principal_id: str
    principal_type: PrincipalType
    authentication_method: AuthenticationMethod
    scopes: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    issued_at_epoch_seconds: int
    expires_at_epoch_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "principal_id",
            _required_identifier(self.principal_id, label="principal_id"),
        )
        if self.principal_type not in {
            "bounded_agent",
            "system_worker",
            "human_reviewer",
            "delivery_service",
        }:
            raise ValueError("Unsupported agent principal_type.")
        if self.authentication_method not in {
            "short_lived_token",
            "signed_service_identity",
            "workload_identity",
        }:
            raise ValueError("Unsupported agent authentication_method.")
        scopes = _unique_tokens(
            tuple(self.scopes),
            label="authorization_scope",
            maximum=32,
        )
        if not scopes:
            raise ValueError("At least one authorization scope is required.")
        object.__setattr__(self, "scopes", scopes)
        capabilities = _unique_tokens(
            tuple(self.allowed_capability_ids),
            label="allowed_capability_id",
            maximum=64,
        )
        if not capabilities:
            raise ValueError("At least one allowed capability is required.")
        object.__setattr__(self, "allowed_capability_ids", capabilities)
        issued = int(self.issued_at_epoch_seconds)
        expires = int(self.expires_at_epoch_seconds)
        if issued < 1 or expires <= issued:
            raise ValueError("Principal lifetime is invalid.")
        if expires - issued > 3600:
            raise ValueError("Agent principals must expire within one hour.")
        object.__setattr__(self, "issued_at_epoch_seconds", issued)
        object.__setattr__(self, "expires_at_epoch_seconds", expires)

    @property
    def principal_fingerprint(self) -> str:
        return stable_fingerprint(self.to_record())

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["scopes"] = list(self.scopes)
        value["allowed_capability_ids"] = list(self.allowed_capability_ids)
        return value


@dataclass(frozen=True)
class AgentApprovalChain:
    tenant_id: str
    proposal_id: str
    proposer_principal_id: str
    approver_principal_id: str
    delivery_principal_id: str
    approval_evidence_fingerprint: str
    approval_status: Literal["approved_for_delivery", "pending", "rejected"]
    expires_at_epoch_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        for field in (
            "proposal_id",
            "proposer_principal_id",
            "approver_principal_id",
            "delivery_principal_id",
        ):
            object.__setattr__(
                self,
                field,
                _required_identifier(getattr(self, field), label=field),
            )
        object.__setattr__(
            self,
            "approval_evidence_fingerprint",
            required_sha256_digest(
                self.approval_evidence_fingerprint,
                label="approval_evidence_fingerprint",
            ),
        )
        if self.approval_status not in {
            "approved_for_delivery",
            "pending",
            "rejected",
        }:
            raise ValueError("Unsupported approval_status.")
        expires = int(self.expires_at_epoch_seconds)
        if expires < 1:
            raise ValueError("Approval expiry must be positive.")
        object.__setattr__(self, "expires_at_epoch_seconds", expires)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentAuthorizationContext:
    authorization_id: str
    request_id: str
    source_authentication_fingerprint: str
    principal: AgentSecurityPrincipal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorization_id",
            _required_identifier(
                self.authorization_id,
                label="authorization_id",
            ),
        )
        object.__setattr__(
            self,
            "request_id",
            _required_identifier(self.request_id, label="request_id"),
        )
        object.__setattr__(
            self,
            "source_authentication_fingerprint",
            required_sha256_digest(
                self.source_authentication_fingerprint,
                label="source_authentication_fingerprint",
            ),
        )
        if not isinstance(self.principal, AgentSecurityPrincipal):
            raise TypeError("AgentAuthorizationContext requires a principal.")

    @property
    def context_fingerprint(self) -> str:
        return stable_fingerprint(self.to_record())

    def to_record(self) -> dict[str, Any]:
        return {
            "authorization_id": self.authorization_id,
            "request_id": self.request_id,
            "source_authentication_fingerprint": (
                self.source_authentication_fingerprint
            ),
            "principal": self.principal.to_record(),
        }


@dataclass(frozen=True)
class AgentCapabilityAuthorizationRequest:
    tenant_id: str
    request_id: str
    capability_id: str
    module_id: Literal[1, 2, 3, 4, 5]
    risk_class: RiskClass
    execution_mode: ExecutionMode
    action: AuthorizationAction
    authorization_context: AgentAuthorizationContext
    approval_chain: AgentApprovalChain | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "request_id",
            _required_identifier(self.request_id, label="request_id"),
        )
        object.__setattr__(
            self,
            "capability_id",
            required_metadata_token(
                self.capability_id,
                label="capability_id",
            ),
        )
        if self.module_id not in {1, 2, 3, 4, 5}:
            raise ValueError("module_id must be between 1 and 5.")
        if self.risk_class not in {
            "read_only",
            "review_only",
            "production_effect",
        }:
            raise ValueError("Unsupported authorization risk_class.")
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "shadow",
            "production",
        }:
            raise ValueError("Unsupported authorization execution_mode.")
        if self.action not in {"read", "propose", "approve", "deliver"}:
            raise ValueError("Unsupported authorization action.")
        if not isinstance(self.authorization_context, AgentAuthorizationContext):
            raise TypeError("Authorization request requires an agent context.")
        if self.approval_chain is not None and not isinstance(
            self.approval_chain,
            AgentApprovalChain,
        ):
            raise TypeError("approval_chain must use AgentApprovalChain.")

    def to_record(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "module_id": self.module_id,
            "risk_class": self.risk_class,
            "execution_mode": self.execution_mode,
            "action": self.action,
            "authorization_context": self.authorization_context.to_record(),
            "approval_chain": (
                self.approval_chain.to_record()
                if self.approval_chain is not None
                else None
            ),
        }


@dataclass(frozen=True)
class AgentSecurityCertificationRequest:
    tenant_id: str
    certification_id: str
    evaluation_epoch_seconds: int
    source_functional_shadow_report_fingerprint: str
    source_security_posture_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "certification_id",
            _required_identifier(
                self.certification_id,
                label="certification_id",
            ),
        )
        evaluated = int(self.evaluation_epoch_seconds)
        if evaluated < 1:
            raise ValueError("evaluation_epoch_seconds must be positive.")
        object.__setattr__(self, "evaluation_epoch_seconds", evaluated)
        for field in (
            "source_functional_shadow_report_fingerprint",
            "source_security_posture_fingerprint",
        ):
            object.__setattr__(
                self,
                field,
                required_sha256_digest(getattr(self, field), label=field),
            )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
