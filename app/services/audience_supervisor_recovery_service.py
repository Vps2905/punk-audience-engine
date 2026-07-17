from __future__ import annotations

import errno
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AudienceSupervisorRecoveryDecision:
    retryable: bool
    error_category: str
    safe_error_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudienceSupervisorRecoveryService:
    # Conservative retry classification for graph-level orchestrator failures.
    # Only known transient transport, timeout, throttling, and server failures
    # are retried. Error messages are never returned because provider messages
    # may contain prompts, credentials, URLs, or other sensitive context.

    _TRANSIENT_ERRNOS = {
        errno.EAGAIN,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    }

    def classify(
        self,
        exc: BaseException,
    ) -> dict[str, Any]:
        safe_type = type(exc).__name__

        if isinstance(exc, TimeoutError):
            return AudienceSupervisorRecoveryDecision(
                retryable=True,
                error_category="timeout",
                safe_error_type=safe_type,
            ).to_dict()

        if isinstance(
            exc,
            (
                ConnectionError,
                BrokenPipeError,
                ConnectionResetError,
            ),
        ):
            return AudienceSupervisorRecoveryDecision(
                retryable=True,
                error_category="connection",
                safe_error_type=safe_type,
            ).to_dict()

        if isinstance(exc, OSError):
            error_number = getattr(exc, "errno", None)
            if error_number in self._TRANSIENT_ERRNOS:
                return AudienceSupervisorRecoveryDecision(
                    retryable=True,
                    error_category="transport",
                    safe_error_type=safe_type,
                ).to_dict()

        status_code = self._status_code(exc)
        if status_code == 429:
            return AudienceSupervisorRecoveryDecision(
                retryable=True,
                error_category="rate_limited",
                safe_error_type=safe_type,
            ).to_dict()

        if status_code is not None and 500 <= status_code <= 599:
            return AudienceSupervisorRecoveryDecision(
                retryable=True,
                error_category="upstream_server",
                safe_error_type=safe_type,
            ).to_dict()

        return AudienceSupervisorRecoveryDecision(
            retryable=False,
            error_category="non_transient",
            safe_error_type=safe_type,
        ).to_dict()

    def max_attempts(self) -> int:
        return self._env_int(
            "AUTONOMOUS_SUPERVISOR_MAX_ORCHESTRATOR_ATTEMPTS",
            default=2,
            minimum=1,
            maximum=3,
        )

    def retry_backoff_seconds(self) -> float:
        return self._env_float(
            "AUTONOMOUS_SUPERVISOR_RETRY_BACKOFF_SECONDS",
            default=0.25,
            minimum=0.0,
            maximum=5.0,
        )

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        candidates = [
            getattr(exc, "status_code", None),
            getattr(exc, "status", None),
        ]

        response = getattr(exc, "response", None)
        if response is not None:
            candidates.extend(
                [
                    getattr(response, "status_code", None),
                    getattr(response, "status", None),
                ]
            )

        for value in candidates:
            try:
                code = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= code <= 599:
                return code

        return None

    @staticmethod
    def _env_int(
        name: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw = os.getenv(name)
        try:
            value = int(str(raw).strip()) if raw is not None else default
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(value, maximum))

    @staticmethod
    def _env_float(
        name: str,
        *,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = os.getenv(name)
        try:
            value = (
                float(str(raw).strip())
                if raw is not None
                else default
            )
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(value, maximum))
