from __future__ import annotations

import os

from fastapi import HTTPException, status


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_production_mode() -> bool:
    app_env = os.getenv("APP_ENV", "local").strip().lower()
    return _bool_env("PRODUCTION_MODE", app_env == "production")


def local_file_storage_allowed() -> bool:
    """
    Local file storage is acceptable for local/dev/test,
    but blocked in production unless explicitly enabled.
    """
    if not is_production_mode():
        return True

    return _bool_env("ALLOW_LOCAL_FILE_STORAGE", False)


def demo_routes_allowed() -> bool:
    """
    Demo/review routes are acceptable locally,
    but blocked in production unless explicitly enabled.
    """
    if not is_production_mode():
        return True

    return _bool_env("ALLOW_DEMO_ROUTES", False)


def require_legacy_local_routes_allowed() -> None:
    """Hide local-file APIs completely from the production route surface."""

    if local_file_storage_allowed():
        return

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Legacy local route is disabled in production.",
    )


def require_demo_routes_allowed() -> None:
    """Hide all adaptive/demo execution routes in production."""

    if demo_routes_allowed():
        return

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Demo route is disabled in production.",
    )


def require_local_file_storage_allowed(feature_name: str) -> None:
    if local_file_storage_allowed():
        return

    raise RuntimeError(
        f"Production guardrail blocked local file storage for {feature_name}. "
        "Use a production DB/object store/vector backend, or set "
        "ALLOW_LOCAL_FILE_STORAGE=true only for an explicitly approved internal environment."
    )
