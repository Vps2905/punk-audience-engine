from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.models.production_bounded_autonomy_contracts import (
    required_metadata_token,
)
from app.models.production_module3_cohort_contracts import (
    required_slug,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)

BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_VERSION = (
    "module5_functional_shadow_execution_v2_authorized"
)


def _required_identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
    if not text or len(text) > 128 or any(char not in allowed for char in text):
        raise ValueError(f"{label} must be a safe opaque identifier.")
    return text


@dataclass(frozen=True)
class FunctionalShadowPolicy:
    """Strict comparison and resource policy for one functional shadow run."""

    max_safe_feature_rows: int = 5000
    max_vector_count_delta: int = 0
    max_ranked_match_delta: int = 0
    max_selected_cohort_delta: int = 0
    max_total_latency_ms: float = 15000.0
    max_stage_latency_ms: float = 10000.0
    max_peak_python_bytes: int = 1_073_741_824
    require_route_match: bool = True
    require_stage_match: bool = True
    require_freshness_match: bool = True
    require_terminal_match: bool = True
    require_approval_match: bool = True

    def __post_init__(self) -> None:
        for label, value, maximum in (
            ("max_safe_feature_rows", self.max_safe_feature_rows, 100000),
            ("max_vector_count_delta", self.max_vector_count_delta, 100000),
            ("max_ranked_match_delta", self.max_ranked_match_delta, 100000),
            ("max_selected_cohort_delta", self.max_selected_cohort_delta, 100000),
            ("max_peak_python_bytes", self.max_peak_python_bytes, 16_000_000_000),
        ):
            parsed = int(value)
            if parsed < 0 or parsed > maximum:
                raise ValueError(f"{label} must be between 0 and {maximum}.")
            object.__setattr__(self, label, parsed)
        if self.max_safe_feature_rows < 1:
            raise ValueError("max_safe_feature_rows must be positive.")
        for label in ("max_total_latency_ms", "max_stage_latency_ms"):
            value = float(getattr(self, label))
            if value <= 0.0 or value > 3_600_000.0:
                raise ValueError(f"{label} must be between 0 and 3600000 ms.")
            object.__setattr__(self, label, value)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FunctionalShadowRunRequest:
    tenant_id: str
    request_id: str
    functional_run_id: str
    feature_set_id: str
    feature_set_version: int
    purpose: str
    source_certification_report_fingerprint: str
    execution_mode: Literal[
        "historical_preview",
        "offline_evaluation",
        "shadow",
    ] = "shadow"

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
            "functional_run_id",
            _required_identifier(
                self.functional_run_id,
                label="functional_run_id",
            ),
        )
        feature_set_id = str(self.feature_set_id or "").strip()
        if not feature_set_id or len(feature_set_id) > 256:
            raise ValueError("feature_set_id is required and must be bounded.")
        object.__setattr__(self, "feature_set_id", feature_set_id)
        if int(self.feature_set_version) < 1:
            raise ValueError("feature_set_version must be >= 1.")
        object.__setattr__(
            self,
            "feature_set_version",
            int(self.feature_set_version),
        )
        object.__setattr__(
            self,
            "purpose",
            required_slug(self.purpose, label="purpose"),
        )
        object.__setattr__(
            self,
            "source_certification_report_fingerprint",
            required_sha256_digest(
                self.source_certification_report_fingerprint,
                label="source_certification_report_fingerprint",
            ),
        )
        if self.execution_mode not in {
            "historical_preview",
            "offline_evaluation",
            "shadow",
        }:
            raise ValueError("Functional shadow cannot use production mode.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FunctionalStageRecord:
    stage_id: str
    module_id: Literal[1, 2, 3, 4, 5]
    status: Literal["completed", "failed", "blocked", "skipped"]
    latency_ms: float
    peak_python_bytes: int
    result_fingerprint: str | None = None
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_id",
            required_metadata_token(self.stage_id, label="stage_id"),
        )
        if self.module_id not in {1, 2, 3, 4, 5}:
            raise ValueError("module_id must be between 1 and 5.")
        if self.status not in {"completed", "failed", "blocked", "skipped"}:
            raise ValueError("Unsupported functional stage status.")
        latency = float(self.latency_ms)
        if latency < 0.0 or latency > 3_600_000.0:
            raise ValueError("latency_ms must be bounded.")
        object.__setattr__(self, "latency_ms", round(latency, 6))
        peak = int(self.peak_python_bytes)
        if peak < 0 or peak > 16_000_000_000:
            raise ValueError("peak_python_bytes must be bounded.")
        object.__setattr__(self, "peak_python_bytes", peak)
        if self.result_fingerprint is not None:
            object.__setattr__(
                self,
                "result_fingerprint",
                required_sha256_digest(
                    self.result_fingerprint,
                    label="result_fingerprint",
                ),
            )
        reasons = tuple(
            required_metadata_token(value, label="reason_code")
            for value in self.reason_codes
        )
        if len(reasons) > 16 or len(set(reasons)) != len(reasons):
            raise ValueError("reason_codes must be unique and bounded.")
        object.__setattr__(self, "reason_codes", reasons)
        if len(self.metrics) > 32:
            raise ValueError("Functional stage metrics are bounded to 32 values.")
        metrics: dict[str, int | float | bool] = {}
        for key, value in self.metrics.items():
            token = required_metadata_token(key, label="metric_name")
            if not isinstance(value, (int, float, bool)):
                raise TypeError("Functional stage metrics must be numeric or boolean.")
            metrics[token] = value
        object.__setattr__(self, "metrics", metrics)

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        value["metrics"] = dict(self.metrics)
        return value
