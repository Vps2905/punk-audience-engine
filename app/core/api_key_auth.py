from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException, status


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_dotenv_value(key: str) -> Optional[str]:
    env_path = Path(os.getenv("AUDIENCE_API_KEY_CONFIG_FILE", ".env"))

    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)

        if name.strip() == key:
            return value.strip().strip('"').strip("'")

    return None


def _get_expected_api_key() -> Optional[str]:
    return os.getenv("AUDIENCE_API_KEY") or _read_dotenv_value("AUDIENCE_API_KEY")


def _clean_header_value(value: Optional[str]) -> Optional[str]:
    """
    Normalize FastAPI-injected header values.

    When called by FastAPI, values are strings or None.
    When called directly in unit tests, default Header(...) objects can appear.
    Treat non-string values as missing.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return None

    value = value.strip()
    return value or None


def _is_production_mode() -> bool:
    return _truthy(os.getenv("PRODUCTION_MODE") or _read_dotenv_value("PRODUCTION_MODE"))


def _auth_required() -> bool:
    explicit = os.getenv("REQUIRE_AUDIENCE_API_KEY") or _read_dotenv_value("REQUIRE_AUDIENCE_API_KEY")

    if explicit is not None:
        return _truthy(explicit)

    return _is_production_mode()


def require_audience_api_key(
    x_audience_api_key: Optional[str] = Header(default=None, alias="X-Audience-API-Key"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> bool:
    """
    Protect Audience Intelligence endpoints.

    Accepted auth:
    - X-Audience-API-Key: <key>
    - X-API-Key: <key>
    - Authorization: Bearer <key>

    Fail-closed in production if API key is missing from env/.env.
    """

    if not _auth_required():
        return True

    expected = _get_expected_api_key()

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audience API key is required but not configured.",
        )

    provided = _clean_header_value(x_audience_api_key) or _clean_header_value(x_api_key)
    authorization_value = _clean_header_value(authorization)

    if not provided and authorization_value:
        prefix = "Bearer "
        if authorization_value.startswith(prefix):
            provided = authorization_value[len(prefix):].strip()

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Audience API key.",
        )

    if not hmac.compare_digest(str(provided), str(expected)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Audience API key.",
        )

    return True
