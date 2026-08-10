from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from app.models.production_bounded_autonomy_contracts import (
    required_metadata_token,
)
from app.models.production_module3_cohort_contracts import required_slug

BOUNDED_AUTONOMY_SHADOW_COMPARISON_VERSION = (
    "module5_bounded_autonomy_shadow_comparison_v1"
)


def _opaque_identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
    )
    if not text or len(text) > 160 or any(char not in allowed for char in text):
        raise ValueError(f"{label} must be a safe opaque identifier.")
    return text


def _bounded_count(value: Any, *, label: str) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not 0 <= number <= 1_000_000_000:
        raise ValueError(f"{label} is outside the supported range.")
    return number


@dataclass(frozen=True)
class CanonicalOrchestrationObservation:
    """Privacy-minimized facts derived from one real orchestrator result."""

    tenant_id: str
    request_id: str
    run_id: str
    pipeline_status: str
    approval_status: str
    filter_mode: str
    decision_route: str
    decision_stage: str
    freshness_status: str
    source_evaluated: bool
    source_row_count: int
    vector_count: int
    ranked_match_count: int
    selected_cohort_count: int
    prepared_candidate_count: int
    terminal: bool
    approval_required: bool
    downstream_export_enabled: bool
    raw_identifiers_returned: bool
    activation_or_export_performed: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            required_slug(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "request_id",
            _opaque_identifier(self.request_id, label="request_id"),
        )
        object.__setattr__(
            self,
            "run_id",
            _opaque_identifier(self.run_id, label="run_id"),
        )
        for field_name in (
            "pipeline_status",
            "approval_status",
            "filter_mode",
            "decision_route",
            "decision_stage",
            "freshness_status",
        ):
            object.__setattr__(
                self,
                field_name,
                required_metadata_token(
                    getattr(self, field_name),
                    label=field_name,
                ),
            )
        for field_name in (
            "source_row_count",
            "vector_count",
            "ranked_match_count",
            "selected_cohort_count",
            "prepared_candidate_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_count(getattr(self, field_name), label=field_name),
            )
        reasons = tuple(
            required_metadata_token(value, label="reason_code")
            for value in self.reason_codes
        )
        if len(reasons) > 32 or len(set(reasons)) != len(reasons):
            raise ValueError("reason_codes must be unique and bounded.")
        object.__setattr__(self, "reason_codes", reasons)

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        return value


@dataclass(frozen=True)
class AutonomousShadowDecision:
    """Ephemeral independent decision emitted by the bounded adapters."""

    route: str
    stage: str
    freshness_status: str
    source_evaluated: bool
    source_row_count: int
    vector_count: int
    ranked_match_count: int
    selected_cohort_count: int
    prepared_candidate_count: int
    terminal: bool
    approval_required: bool
    downstream_export_enabled: bool = False
    raw_identifiers_returned: bool = False
    activation_or_export_performed: bool = False
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("route", "stage", "freshness_status"):
            object.__setattr__(
                self,
                field_name,
                required_metadata_token(
                    getattr(self, field_name),
                    label=field_name,
                ),
            )
        for field_name in (
            "source_row_count",
            "vector_count",
            "ranked_match_count",
            "selected_cohort_count",
            "prepared_candidate_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_count(getattr(self, field_name), label=field_name),
            )
        reasons = tuple(
            required_metadata_token(value, label="reason_code")
            for value in self.reason_codes
        )
        if len(reasons) > 32 or len(set(reasons)) != len(reasons):
            raise ValueError("reason_codes must be unique and bounded.")
        object.__setattr__(self, "reason_codes", reasons)

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        return value


@dataclass(frozen=True)
class ShadowDivergencePolicy:
    max_source_row_delta: int = 0
    max_vector_count_delta: int = 0
    max_ranked_match_delta: int = 0
    max_selected_cohort_delta: int = 0
    max_prepared_candidate_delta: int = 0
    require_route_match: bool = True
    require_stage_match: bool = True
    require_terminal_match: bool = True
    require_freshness_match: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "max_source_row_delta",
            "max_vector_count_delta",
            "max_ranked_match_delta",
            "max_selected_cohort_delta",
            "max_prepared_candidate_delta",
        ):
            value = _bounded_count(getattr(self, field_name), label=field_name)
            object.__setattr__(self, field_name, value)

    def to_record(self) -> Mapping[str, Any]:
        return asdict(self)
