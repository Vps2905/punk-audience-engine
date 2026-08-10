from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from app.models.production_module3_cohort_contracts import required_slug


BOUNDED_AUTONOMY_PLAN_VERSION = "module5_bounded_autonomy_plan_v1"
BOUNDED_AUTONOMY_EVIDENCE_VERSION = "module5_bounded_autonomy_evidence_v1"

ExecutionMode = Literal[
    "historical_preview",
    "offline_evaluation",
    "shadow",
    "production",
]
RiskClass = Literal["read_only", "review_only", "production_effect"]
CapabilityStatus = Literal["completed", "failed", "blocked", "skipped"]


def required_metadata_token(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_:-.")
    if not text or len(text) > 128 or any(char not in allowed for char in text):
        raise ValueError(f"{label} must be a safe metadata token.")
    return text


def _unique_tokens(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(
        required_metadata_token(value, label=label)
        for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} values must be unique.")
    return normalized


def _required_identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
    if not text or len(text) > 128 or any(char not in allowed for char in text):
        raise ValueError(f"{label} must be a safe opaque identifier.")
    return text


@dataclass(frozen=True)
class AutonomyBudget:
    max_tasks: int = 16
    max_total_attempts: int = 24
    max_replans: int = 1
    max_parallel_tasks: int = 4
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_tasks) <= 64:
            raise ValueError("max_tasks must be between 1 and 64.")
        if not 1 <= int(self.max_total_attempts) <= 256:
            raise ValueError("max_total_attempts must be between 1 and 256.")
        if not 0 <= int(self.max_replans) <= 5:
            raise ValueError("max_replans must be between 0 and 5.")
        if not 1 <= int(self.max_parallel_tasks) <= 16:
            raise ValueError("max_parallel_tasks must be between 1 and 16.")
        if not 1 <= int(self.timeout_seconds) <= 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600.")

    def to_record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AutonomyGoal:
    tenant_id: str
    request_id: str
    goal_id: str
    objective: str
    requested_outcomes: tuple[str, ...]
    execution_mode: ExecutionMode = "offline_evaluation"
    constraints: tuple[str, ...] = ()
    budget: AutonomyBudget = field(default_factory=AutonomyBudget)

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
            "goal_id",
            _required_identifier(self.goal_id, label="goal_id"),
        )
        objective = str(self.objective or "").strip()
        if not objective or len(objective) > 4000:
            raise ValueError("objective must contain between 1 and 4000 characters.")
        object.__setattr__(self, "objective", objective)
        outcomes = _unique_tokens(
            tuple(self.requested_outcomes),
            label="requested_outcome",
        )
        if not outcomes:
            raise ValueError("At least one requested outcome is required.")
        object.__setattr__(self, "requested_outcomes", outcomes)
        object.__setattr__(
            self,
            "constraints",
            _unique_tokens(tuple(self.constraints), label="goal_constraint"),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "shadow",
            "production",
        }:
            raise ValueError("Unsupported execution_mode.")

    @property
    def objective_sha256(self) -> str:
        return hashlib.sha256(self.objective.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, Any]:
        """Return privacy-minimized goal metadata without prompt content."""
        return {
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "goal_id": self.goal_id,
            "objective_sha256": self.objective_sha256,
            "requested_outcomes": list(self.requested_outcomes),
            "execution_mode": self.execution_mode,
            "constraints": list(self.constraints),
            "budget": self.budget.to_record(),
        }


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    module_id: Literal[1, 2, 3, 4, 5]
    description: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    allowed_modes: tuple[ExecutionMode, ...] = (
        "historical_preview",
        "offline_evaluation",
        "shadow",
        "production",
    )
    risk_class: RiskClass = "read_only"
    idempotent: bool = True
    max_attempts: int = 1
    retryable_error_categories: tuple[str, ...] = ()
    fallback_capability_ids: tuple[str, ...] = ()
    priority: int = 100
    mandatory_preflight: bool = False
    postcondition: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_id",
            required_metadata_token(self.capability_id, label="capability_id"),
        )
        if self.module_id not in {1, 2, 3, 4, 5}:
            raise ValueError("module_id must be between 1 and 5.")
        description = str(self.description or "").strip()
        if not description or len(description) > 500:
            raise ValueError("description must contain between 1 and 500 characters.")
        object.__setattr__(self, "description", description)
        object.__setattr__(
            self,
            "requires",
            _unique_tokens(tuple(self.requires), label="required_outcome"),
        )
        provides = _unique_tokens(tuple(self.provides), label="provided_outcome")
        if not provides:
            raise ValueError("A capability must provide at least one outcome.")
        object.__setattr__(self, "provides", provides)
        modes = tuple(dict.fromkeys(self.allowed_modes))
        if not modes or any(
            mode not in {
                "historical_preview",
                "offline_evaluation",
                "shadow",
                "production",
            }
            for mode in modes
        ):
            raise ValueError("allowed_modes contains an unsupported execution mode.")
        object.__setattr__(self, "allowed_modes", modes)
        if self.risk_class not in {
            "read_only",
            "review_only",
            "production_effect",
        }:
            raise ValueError("Unsupported risk_class.")
        if not 1 <= int(self.max_attempts) <= 10:
            raise ValueError("max_attempts must be between 1 and 10.")
        object.__setattr__(self, "max_attempts", int(self.max_attempts))
        object.__setattr__(
            self,
            "retryable_error_categories",
            _unique_tokens(
                tuple(self.retryable_error_categories),
                label="retryable_error_category",
            ),
        )
        object.__setattr__(
            self,
            "fallback_capability_ids",
            _unique_tokens(
                tuple(self.fallback_capability_ids),
                label="fallback_capability_id",
            ),
        )
        if not 0 <= int(self.priority) <= 10000:
            raise ValueError("priority must be between 0 and 10000.")
        object.__setattr__(self, "priority", int(self.priority))
        if self.mandatory_preflight and self.postcondition:
            raise ValueError("A capability cannot be both preflight and postcondition.")

    def to_record(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "module_id": self.module_id,
            "requires": list(self.requires),
            "provides": list(self.provides),
            "allowed_modes": list(self.allowed_modes),
            "risk_class": self.risk_class,
            "idempotent": self.idempotent,
            "max_attempts": self.max_attempts,
            "retryable_error_categories": list(
                self.retryable_error_categories
            ),
            "fallback_capability_ids": list(self.fallback_capability_ids),
            "priority": self.priority,
            "mandatory_preflight": self.mandatory_preflight,
            "postcondition": self.postcondition,
        }


@dataclass(frozen=True)
class PlanTask:
    task_id: str
    capability_id: str
    module_id: int
    depends_on: tuple[str, ...]
    required_outcomes: tuple[str, ...]
    risk_class: RiskClass
    max_attempts: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_id",
            required_metadata_token(self.task_id, label="task_id"),
        )
        object.__setattr__(
            self,
            "capability_id",
            required_metadata_token(self.capability_id, label="capability_id"),
        )
        object.__setattr__(
            self,
            "depends_on",
            _unique_tokens(tuple(self.depends_on), label="task_dependency"),
        )
        object.__setattr__(
            self,
            "required_outcomes",
            _unique_tokens(
                tuple(self.required_outcomes),
                label="required_outcome",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["depends_on"] = list(self.depends_on)
        value["required_outcomes"] = list(self.required_outcomes)
        return value


@dataclass(frozen=True)
class BoundedExecutionPlan:
    plan_id: str
    plan_version: str
    goal: Mapping[str, Any]
    tasks: tuple[PlanTask, ...]
    requested_outcomes: tuple[str, ...]
    selected_capability_ids: tuple[str, ...]
    parallel_waves: tuple[tuple[str, ...], ...]
    excluded_capability_ids: tuple[str, ...]
    plan_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "goal": dict(self.goal),
            "tasks": [task.to_record() for task in self.tasks],
            "requested_outcomes": list(self.requested_outcomes),
            "selected_capability_ids": list(self.selected_capability_ids),
            "parallel_waves": [list(wave) for wave in self.parallel_waves],
            "excluded_capability_ids": list(self.excluded_capability_ids),
            "plan_fingerprint": self.plan_fingerprint,
        }


@dataclass(frozen=True)
class CapabilityExecutionResult:
    status: CapabilityStatus
    provided_outcomes: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, int | float | bool] = field(default_factory=dict)
    artifact_references: tuple[str, ...] = ()
    output_values: Mapping[str, Any] = field(default_factory=dict, repr=False)
    error_category: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "blocked", "skipped"}:
            raise ValueError("Unsupported capability result status.")
        object.__setattr__(
            self,
            "provided_outcomes",
            _unique_tokens(
                tuple(self.provided_outcomes),
                label="provided_outcome",
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_tokens(tuple(self.reason_codes), label="reason_code"),
        )
        object.__setattr__(
            self,
            "artifact_references",
            tuple(
                _required_identifier(value, label="artifact_reference")
                for value in self.artifact_references
            ),
        )
        if len(self.metrics) > 32:
            raise ValueError("Capability metrics are bounded to 32 values.")
        clean_metrics: dict[str, int | float | bool] = {}
        for key, value in self.metrics.items():
            token = required_metadata_token(key, label="metric_name")
            if not isinstance(value, (int, float, bool)):
                raise ValueError("Capability metrics must be numeric or boolean.")
            clean_metrics[token] = value
        object.__setattr__(self, "metrics", clean_metrics)
        if self.error_category is not None:
            object.__setattr__(
                self,
                "error_category",
                required_metadata_token(
                    self.error_category,
                    label="error_category",
                ),
            )

    def to_evidence(self, *, attempts: int) -> dict[str, Any]:
        return {
            "status": self.status,
            "provided_outcomes": list(self.provided_outcomes),
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
            "artifact_references": list(self.artifact_references),
            "error_category": self.error_category,
            "attempts": int(attempts),
        }
