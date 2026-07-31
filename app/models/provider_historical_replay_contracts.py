from __future__ import annotations

import re
from dataclasses import dataclass


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class HistoricalReplayConfig:
    """Fail-closed contract for the legacy Postgres privacy replay."""

    tenant_id: str
    provider_id: str = "historical_echo"
    dataset_id: str = "maid_extractions"
    schema_name: str = "public"
    table_name: str = "maid_extractions"
    min_cohort_size: int = 1000
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    statement_timeout_ms: int = 900_000
    data_use_mode: str = "offline_evaluation"
    privacy_policy_version: str = "historical-replay-policy-v2"

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "provider_id", "dataset_id"):
            value = str(getattr(self, field_name) or "").strip()
            if not _SLUG.fullmatch(value):
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)

        for field_name in ("schema_name", "table_name"):
            value = str(getattr(self, field_name) or "").strip()
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)

        if self.min_cohort_size < 1000:
            raise ValueError(
                "Historical replay requires min_cohort_size >= 1000"
            )
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if not 1_000 <= self.statement_timeout_ms <= 3_600_000:
            raise ValueError(
                "statement_timeout_ms must be between 1000 and 3600000"
            )
        if self.data_use_mode not in {
            "historical_preview",
            "offline_evaluation",
        }:
            raise ValueError(
                "Historical replay is restricted to non-activatable modes"
            )
        policy = str(self.privacy_policy_version or "").strip()
        if not _SLUG.fullmatch(policy):
            raise ValueError("privacy_policy_version is invalid")
        object.__setattr__(self, "privacy_policy_version", policy)
