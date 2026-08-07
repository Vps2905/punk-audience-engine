from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, Response, status

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class AudienceRequestContext:
    """Privacy-safe identity propagated across an authenticated API request."""

    tenant_id: str
    request_id: str

    def to_safe_dict(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
        }


def require_authenticated_audience_request(
    request: Request,
    response: Response,
    _authenticated: bool = Depends(require_audience_api_key),
    tenant_id: str = Depends(require_verified_audience_tenant),
    x_request_id: Optional[str] = Header(
        default=None,
        alias="X-Request-Id",
        max_length=128,
    ),
) -> AudienceRequestContext:
    """
    Enforce the shared API-key and verified-tenant boundary.

    The resulting context is stored on ``request.state`` so downstream route,
    audit, and service layers can propagate identity without re-reading headers.
    A caller-supplied request ID is accepted only when it is log-safe; otherwise
    a server-generated ID is used.
    """

    request_id = str(x_request_id or "").strip()
    if request_id and not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request identifier.",
        )
    if not request_id:
        request_id = f"req_{uuid4().hex}"

    context = AudienceRequestContext(
        tenant_id=tenant_id,
        request_id=request_id,
    )
    request.state.audience_request_context = context
    response.headers["X-Request-Id"] = request_id
    return context


def get_audience_request_context(request: Request) -> AudienceRequestContext:
    context = getattr(request.state, "audience_request_context", None)
    if not isinstance(context, AudienceRequestContext):
        raise RuntimeError("Authenticated audience request context is missing.")
    return context
