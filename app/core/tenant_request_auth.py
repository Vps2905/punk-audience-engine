from __future__ import annotations

import hashlib
import hmac
import os
import re

from fastapi import Header, HTTPException, status

TENANT_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def tenant_signature(
    tenant_id: str,
    *,
    secret: str,
) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        tenant_id.lower().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def require_verified_audience_tenant(
    x_audience_tenant_id: str = Header(
        ...,
        alias="X-Audience-Tenant-Id",
        min_length=1,
        max_length=128,
    ),
    x_audience_tenant_signature: str | None = Header(
        default=None,
        alias="X-Audience-Tenant-Signature",
    ),
) -> str:
    tenant_id = x_audience_tenant_id.strip()
    if not TENANT_PATTERN.fullmatch(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tenant identifier.",
        )

    production = _truthy(os.getenv("PRODUCTION_MODE")) or (
        os.getenv("APP_ENV", "").strip().lower() == "production"
    )
    signature_required = production or _truthy(
        os.getenv("REQUIRE_PUNK_AI_TENANT_SIGNATURE")
    )
    if not signature_required:
        return tenant_id.lower()

    secret = os.getenv("PUNK_AI_TENANT_AUTH_SECRET")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Punk AI tenant authentication is not configured.",
        )
    provided_signature = str(x_audience_tenant_signature or "").strip()
    if not provided_signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Punk AI tenant signature.",
        )
    expected_signature = tenant_signature(tenant_id, secret=secret)
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Punk AI tenant signature.",
        )
    return tenant_id.lower()
