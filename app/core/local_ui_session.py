from __future__ import annotations

import hashlib
import ipaddress
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Mapping
from uuid import uuid4

from fastapi import Header, HTTPException, Request, Response, status

from app.core.audience_request_context import AudienceRequestContext
from app.core.tenant_request_auth import TENANT_PATTERN


LOCAL_UI_COOKIE_NAME = "punk_audience_local_ui"
LOCAL_UI_SESSION_SECONDS = 8 * 60 * 60
LOCAL_UI_HEADER = "X-Punk-Local-UI"


@dataclass(frozen=True)
class _LocalUISession:
    tenant_id: str
    expires_at: float


_session_lock = threading.Lock()
_sessions: dict[str, _LocalUISession] = {}


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def local_ui_enabled(
    environment: Mapping[str, str] | None = None,
) -> bool:
    selected = environment if environment is not None else os.environ
    production = _truthy(selected.get("PRODUCTION_MODE")) or (
        str(selected.get("APP_ENV") or "").strip().lower()
        == "production"
    )
    return bool(
        not production
        and _truthy(selected.get("AUDIENCE_LOCAL_UI_ENABLED"))
    )


def _loopback_request(request: Request) -> bool:
    host = str(request.client.host if request.client else "").strip()
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _require_local_request(request: Request) -> None:
    if not local_ui_enabled() or not _loopback_request(request):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found.",
        )


def require_local_ui_request(request: Request) -> None:
    """Allow explicitly enabled loopback-only workspace resources."""
    _require_local_request(request)


def _tenant_id() -> str:
    tenant_id = str(
        os.getenv("AUDIENCE_LOCAL_UI_TENANT_ID") or "punk_internal"
    ).strip().lower()
    if not TENANT_PATTERN.fullmatch(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local audience workspace tenant is invalid.",
        )
    return tenant_id


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _remove_expired_sessions(now: float) -> None:
    expired = [
        digest
        for digest, session in _sessions.items()
        if session.expires_at <= now
    ]
    for digest in expired:
        _sessions.pop(digest, None)


def issue_local_ui_session(
    request: Request,
    response: Response,
) -> str:
    _require_local_request(request)
    tenant_id = _tenant_id()
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _session_lock:
        _remove_expired_sessions(now)
        _sessions[_token_digest(token)] = _LocalUISession(
            tenant_id=tenant_id,
            expires_at=now + LOCAL_UI_SESSION_SECONDS,
        )
    response.set_cookie(
        key=LOCAL_UI_COOKIE_NAME,
        value=token,
        max_age=LOCAL_UI_SESSION_SECONDS,
        httponly=True,
        secure=False,
        samesite="strict",
        path="/",
    )
    return tenant_id


def require_local_ui_session(
    request: Request,
    x_punk_local_ui: str | None = Header(
        default=None,
        alias=LOCAL_UI_HEADER,
    ),
) -> AudienceRequestContext:
    _require_local_request(request)
    if str(x_punk_local_ui or "").strip() != "1":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local audience workspace request was rejected.",
        )
    token = str(request.cookies.get(LOCAL_UI_COOKIE_NAME) or "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Local audience workspace session is missing.",
        )

    now = time.time()
    with _session_lock:
        _remove_expired_sessions(now)
        session = _sessions.get(_token_digest(token))
    if session is None or session.expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Local audience workspace session has expired.",
        )

    return AudienceRequestContext(
        tenant_id=session.tenant_id,
        request_id=f"local_ui_{uuid4().hex}",
    )
