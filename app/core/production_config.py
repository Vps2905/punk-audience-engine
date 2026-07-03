from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ProductionConfig:
    app_env: str
    production_mode: bool
    synthetic_engine: str
    synthetic_epsilon: float
    k_min: int
    allow_synthetic_fallback: bool
    max_epsilon_per_run: float
    privacy_ledger_path: Path
    audit_log_path: Path
    approval_dir: Path


def load_production_config() -> ProductionConfig:
    app_env = os.getenv("APP_ENV", "local").strip().lower()
    production_mode = _bool_env("PRODUCTION_MODE", app_env == "production")

    return ProductionConfig(
        app_env=app_env,
        production_mode=production_mode,
        synthetic_engine=os.getenv("SYNTHETIC_ENGINE", "dpgc").strip().lower(),
        synthetic_epsilon=float(os.getenv("SYNTHETIC_EPSILON", "1.0")),
        k_min=int(os.getenv("K_ANONYMITY_MIN", "1000")),
        allow_synthetic_fallback=_bool_env("ALLOW_SYNTHETIC_FALLBACK", False),
        max_epsilon_per_run=float(os.getenv("MAX_EPSILON_PER_RUN", "10.0")),
        privacy_ledger_path=Path(os.getenv("PRIVACY_LEDGER_PATH", "data/privacy_ledger/ledger.jsonl")),
        audit_log_path=Path(os.getenv("AUDIT_LOG_PATH", "data/audit/audit_log.jsonl")),
        approval_dir=Path(os.getenv("APPROVAL_DIR", "data/approvals")),
    )
