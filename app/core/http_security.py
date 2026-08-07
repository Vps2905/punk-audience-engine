from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from app.core.production_guardrails import is_production_mode


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def production_api_docs_enabled() -> bool:
    if not is_production_mode():
        return True
    return _truthy(os.getenv("EXPOSE_API_DOCS"), default=False)


def configured_allowed_hosts() -> tuple[str, ...]:
    values = []
    for value in str(os.getenv("PRODUCTION_ALLOWED_HOSTS") or "").split(","):
        host = value.strip().lower().rstrip(".")
        if host and host not in values:
            values.append(host)
    return tuple(values)


def maximum_request_body_bytes() -> int:
    raw = str(os.getenv("MAX_REQUEST_BODY_BYTES") or "10485760").strip()
    try:
        value = int(raw)
    except ValueError:
        return 10_485_760
    return min(max(value, 1024), 104_857_600)


class ProductionHTTPSecurityMiddleware:
    """Apply safe HTTP response headers and bounded production request checks."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        if self._host_blocked(scope):
            await self._reject(send, 400, b"Invalid host header.")
            return
        if self._content_length_exceeded(scope):
            await self._reject(send, 413, b"Request body too large.")
            return
        messages, exceeded = await self._bounded_request_messages(receive)
        if exceeded:
            await self._reject(send, 413, b"Request body too large.")
            return

        async def bounded_receive():
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def security_send(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                self._replace(headers, b"x-content-type-options", b"nosniff")
                self._replace(headers, b"x-frame-options", b"DENY")
                self._replace(headers, b"referrer-policy", b"no-referrer")
                self._replace(
                    headers,
                    b"permissions-policy",
                    b"camera=(), microphone=(), geolocation=()",
                )
                self._replace(
                    headers,
                    b"cross-origin-resource-policy",
                    b"same-origin",
                )
                path = str(scope.get("path") or "")
                if path.startswith("/api/"):
                    self._replace(headers, b"cache-control", b"no-store")
                    self._replace(
                        headers,
                        b"content-security-policy",
                        b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
                    )
                if is_production_mode():
                    self._replace(
                        headers,
                        b"strict-transport-security",
                        b"max-age=31536000; includeSubDomains",
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, bounded_receive, security_send)

    def _host_blocked(self, scope: dict[str, Any]) -> bool:
        if not is_production_mode():
            return False
        allowed = configured_allowed_hosts()
        if not allowed:
            return False
        host = self._header(scope.get("headers") or [], b"host")
        if host.startswith("[") and "]" in host:
            hostname = host[1:host.index("]")]
        else:
            hostname = host.split(":", 1)[0]
        hostname = hostname.lower().rstrip(".")
        return not hostname or hostname not in allowed

    def _content_length_exceeded(self, scope: dict[str, Any]) -> bool:
        raw = self._header(scope.get("headers") or [], b"content-length")
        if not raw:
            return False
        try:
            return int(raw) > maximum_request_body_bytes()
        except ValueError:
            return True

    async def _bounded_request_messages(self, receive) -> tuple[list[dict], bool]:
        messages: list[dict] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.disconnect":
                return messages, False
            if message.get("type") != "http.request":
                continue
            total += len(message.get("body") or b"")
            if total > maximum_request_body_bytes():
                return messages, True
            if not message.get("more_body", False):
                return messages, False

    def _header(self, headers: Iterable[tuple[bytes, bytes]], name: bytes) -> str:
        for key, value in headers:
            if key.lower() == name:
                return value.decode("latin-1").strip()
        return ""

    def _replace(
        self,
        headers: list[tuple[bytes, bytes]],
        name: bytes,
        value: bytes,
    ) -> None:
        headers[:] = [(key, item) for key, item in headers if key.lower() != name]
        headers.append((name, value))

    async def _reject(self, send, status_code: int, body: bytes) -> None:
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-content-type-options", b"nosniff"),
                (b"x-frame-options", b"DENY"),
                (b"referrer-policy", b"no-referrer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
