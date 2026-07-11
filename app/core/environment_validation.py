from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class EnvironmentValidationResult:
    ok: bool
    mode: str
    missing_required: List[str]
    warnings: List[str]
    env_presence: Dict[str, bool]


def is_production_mode() -> bool:
    return _truthy(os.getenv("PRODUCTION_MODE")) or os.getenv("APP_ENV", "").strip().lower() == "production"


def validate_environment() -> EnvironmentValidationResult:
    production = is_production_mode()
    mode = "production" if production else os.getenv("APP_ENV", "local")

    required_in_production = [
        "AUDIENCE_API_KEY",
        "ECHO_DATABASE_URL",
        "HASH_SECRET",
        "AUDIENCE_HASH_SALT",
        "VECTOR_BACKEND",
    ]

    optional_but_recommended = [
        "OPENROUTER_API_KEY",
    ]

    missing_required: List[str] = []
    warnings: List[str] = []

    for key in required_in_production:
        if production and not os.getenv(key):
            missing_required.append(key)

    for key in optional_but_recommended:
        if production and not os.getenv(key):
            warnings.append(f"{key} is not configured.")

    if production and _truthy(os.getenv("ALLOW_LOCAL_FILE_STORAGE")):
        warnings.append("ALLOW_LOCAL_FILE_STORAGE is enabled in production.")

    if production and _truthy(os.getenv("ALLOW_DEMO_ROUTES")):
        warnings.append("ALLOW_DEMO_ROUTES is enabled in production.")

    vector_backend = os.getenv("VECTOR_BACKEND", "").strip().lower()
    allowed_production_vector_backends = {"postgres_array", "postgres", "pg_array", "pgvector"}

    if production and vector_backend and vector_backend not in allowed_production_vector_backends:
        missing_required.append("VECTOR_BACKEND(postgres_array|pgvector)")

    env_presence = {
        "AUDIENCE_API_KEY": bool(os.getenv("AUDIENCE_API_KEY")),
        "ECHO_DATABASE_URL": bool(os.getenv("ECHO_DATABASE_URL")),
        "HASH_SECRET": bool(os.getenv("HASH_SECRET")),
        "AUDIENCE_HASH_SALT": bool(os.getenv("AUDIENCE_HASH_SALT")),
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "PRODUCTION_MODE": bool(os.getenv("PRODUCTION_MODE")),
        "APP_ENV": bool(os.getenv("APP_ENV")),
        "ALLOW_LOCAL_FILE_STORAGE": bool(os.getenv("ALLOW_LOCAL_FILE_STORAGE")),
        "ALLOW_DEMO_ROUTES": bool(os.getenv("ALLOW_DEMO_ROUTES")),
        "VECTOR_BACKEND": bool(os.getenv("VECTOR_BACKEND")),
    }

    return EnvironmentValidationResult(
        ok=not missing_required,
        mode=mode,
        missing_required=missing_required,
        warnings=warnings,
        env_presence=env_presence,
    )


def assert_environment_ready() -> None:
    result = validate_environment()

    if not result.ok:
        missing = ", ".join(result.missing_required)
        raise RuntimeError(f"Production environment is not ready. Missing: {missing}")
