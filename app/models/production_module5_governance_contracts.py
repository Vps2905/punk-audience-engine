from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.models.production_module3_cohort_contracts import (
    required_slug,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


MODULE5_EXECUTION_POLICY_VERSION = "module5_agent_execution_evidence_v1"
MODULE5_POLICY_POLICY_VERSION = "module5_supervisor_policy_evaluation_v1"
MODULE5_CIRCUIT_BREAKER_POLICY_VERSION = "module5_circuit_breaker_policy_v1"
MODULE5_HUMAN_REVIEW_POLICY_VERSION = "module5_human_shadow_review_v1"
MODULE5_RECOVERY_POLICY_VERSION = "module5_recovery_certification_v1"


def _required_identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
    if not text or len(text) > 128 or any(character not in allowed for character in text):
        raise ValueError(f"{label} must be a safe opaque identifier.")
    return text


@dataclass(frozen=True)
class Module5AgentExecutionRequest:
    tenant_id: str
    request_id: str
    run_id: str
    execution_mode: Literal[
        "historical_preview",
        "offline_evaluation",
        "production",
    ]

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
            "run_id",
            _required_identifier(self.run_id, label="run_id"),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported Module 5 execution_mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Module5CircuitBreakerRequest:
    tenant_id: str
    execution_report_fingerprints: tuple[str, ...]
    failure_rate_threshold: float = 0.5
    minimum_sample_size: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        fingerprints = tuple(
            required_sha256_digest(value, label="execution_report_fingerprint")
            for value in self.execution_report_fingerprints
        )
        if not fingerprints:
            raise ValueError("Module 5 circuit-breaker evidence is required.")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("Module 5 execution fingerprints must be unique.")
        object.__setattr__(self, "execution_report_fingerprints", fingerprints)
        threshold = float(self.failure_rate_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("failure_rate_threshold must be in (0, 1].")
        object.__setattr__(self, "failure_rate_threshold", threshold)
        if not 1 <= int(self.minimum_sample_size) <= 10000:
            raise ValueError("minimum_sample_size must be between 1 and 10000.")
        object.__setattr__(self, "minimum_sample_size", int(self.minimum_sample_size))

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["execution_report_fingerprints"] = list(
            self.execution_report_fingerprints
        )
        return value
