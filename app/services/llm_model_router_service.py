from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


ResponseValidator = Callable[[str], Any]


class LLMResponseValidationError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = str(category or "validation_error")


class LLMModelRouterError(RuntimeError):
    def __init__(self, message: str, telemetry: dict[str, Any]) -> None:
        super().__init__(message)
        self.telemetry = telemetry


@dataclass(frozen=True)
class LLMModelTarget:
    provider: str
    model: str


class LLMModelRouterService:
    """
    Provider-agnostic LLM model router for production intent resolution.

    The router:
    - attempts the configured primary model first;
    - fails over through LLM_INTENT_MODEL_CHAIN;
    - retries only transient provider/network failures;
    - validates every response before accepting it;
    - returns sanitized run telemetry without secrets or prompt content.
    """

    TRANSIENT_CATEGORIES = {
        "timeout",
        "network_error",
        "rate_limited",
        "provider_server_error",
    }

    def route(
        self,
        *,
        messages: list[dict[str, str]],
        primary_provider: str,
        primary_model: str,
        primary_api_key: str,
        primary_base_url: str,
        timeout_seconds: int,
        max_tokens: int,
        validator: ResponseValidator,
    ) -> dict[str, Any]:
        started = time.monotonic()
        targets = self._build_targets(
            primary_provider=primary_provider,
            primary_model=primary_model,
        )
        max_attempts_per_model = self._env_int(
            "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
            default=2,
            minimum=1,
            maximum=3,
        )
        backoff_seconds = self._env_float(
            "LLM_ROUTER_RETRY_BACKOFF_SECONDS",
            default=0.5,
            minimum=0.0,
            maximum=10.0,
        )

        attempts: list[dict[str, Any]] = []
        first_failure_category: str | None = None

        for target_index, target in enumerate(targets):
            runtime = self._resolve_runtime(
                target=target,
                primary_provider=primary_provider,
                primary_api_key=primary_api_key,
                primary_base_url=primary_base_url,
            )

            if not runtime["api_key"]:
                failure = self._attempt_record(
                    target=target,
                    attempt=1,
                    status="failed",
                    latency_ms=0,
                    category="missing_api_key",
                    http_status=None,
                    error_message=(
                        f"No API key configured for provider "
                        f"{target.provider}."
                    ),
                    retryable=False,
                )
                attempts.append(failure)
                first_failure_category = (
                    first_failure_category
                    or failure["error_category"]
                )
                continue

            if not runtime["base_url"]:
                failure = self._attempt_record(
                    target=target,
                    attempt=1,
                    status="failed",
                    latency_ms=0,
                    category="missing_base_url",
                    http_status=None,
                    error_message=(
                        f"No base URL configured for provider "
                        f"{target.provider}."
                    ),
                    retryable=False,
                )
                attempts.append(failure)
                first_failure_category = (
                    first_failure_category
                    or failure["error_category"]
                )
                continue

            for model_attempt in range(
                1,
                max_attempts_per_model + 1,
            ):
                attempt_started = time.monotonic()

                try:
                    content = self._invoke_target(
                        target=target,
                        messages=messages,
                        api_key=runtime["api_key"],
                        base_url=runtime["base_url"],
                        timeout_seconds=timeout_seconds,
                        max_tokens=max_tokens,
                    )
                    try:
                        validated = validator(content)
                    except LLMResponseValidationError:
                        raise
                    except json.JSONDecodeError as exc:
                        raise LLMResponseValidationError(
                            "invalid_json",
                            "Model output was not valid JSON.",
                        ) from exc
                    except ValueError as exc:
                        raise LLMResponseValidationError(
                            "invalid_json",
                            "Model output failed JSON validation.",
                        ) from exc
                    latency_ms = self._elapsed_ms(
                        attempt_started
                    )
                    attempts.append(
                        self._attempt_record(
                            target=target,
                            attempt=model_attempt,
                            status="succeeded",
                            latency_ms=latency_ms,
                            category=None,
                            http_status=200,
                            error_message=None,
                            retryable=False,
                        )
                    )

                    telemetry = self._telemetry(
                        success=True,
                        attempts=attempts,
                        started=started,
                        selected_target=target,
                        primary_target=targets[0],
                        first_failure_category=(
                            first_failure_category
                        ),
                    )
                    return {
                        "content": content,
                        "validated": validated,
                        "telemetry": telemetry,
                    }

                except LLMResponseValidationError as exc:
                    category = exc.category
                    retryable = False
                    http_status = None
                    error_message = str(exc)

                except urllib.error.HTTPError as exc:
                    category = self._classify_http_status(
                        exc.code
                    )
                    retryable = (
                        category in self.TRANSIENT_CATEGORIES
                    )
                    http_status = int(exc.code)
                    error_message = self._telemetry_error_message(
                        category=category,
                        http_status=http_status,
                    )
                    try:
                        exc.close()
                    except Exception:
                        pass

                except (TimeoutError, urllib.error.URLError) as exc:
                    reason = getattr(exc, "reason", None)
                    if isinstance(reason, TimeoutError):
                        category = "timeout"
                    else:
                        category = "network_error"
                    retryable = True
                    http_status = None
                    error_message = self._telemetry_error_message(
                        category=category,
                        http_status=None,
                    )

                except json.JSONDecodeError:
                    category = "invalid_provider_envelope"
                    retryable = False
                    http_status = None
                    error_message = self._telemetry_error_message(
                        category=category,
                        http_status=None,
                    )

                except Exception:
                    category = "provider_response_error"
                    retryable = False
                    http_status = None
                    error_message = self._telemetry_error_message(
                        category=category,
                        http_status=None,
                    )

                latency_ms = self._elapsed_ms(attempt_started)
                attempts.append(
                    self._attempt_record(
                        target=target,
                        attempt=model_attempt,
                        status="failed",
                        latency_ms=latency_ms,
                        category=category,
                        http_status=http_status,
                        error_message=error_message,
                        retryable=retryable,
                    )
                )
                first_failure_category = (
                    first_failure_category or category
                )

                should_retry_same_model = (
                    retryable
                    and model_attempt < max_attempts_per_model
                )
                if not should_retry_same_model:
                    break

                if backoff_seconds:
                    time.sleep(
                        backoff_seconds * model_attempt
                    )

        telemetry = self._telemetry(
            success=False,
            attempts=attempts,
            started=started,
            selected_target=None,
            primary_target=(targets[0] if targets else None),
            first_failure_category=first_failure_category,
        )
        raise LLMModelRouterError(
            "All configured LLM models failed validation or provider execution.",
            telemetry,
        )

    def _build_targets(
        self,
        *,
        primary_provider: str,
        primary_model: str,
    ) -> list[LLMModelTarget]:
        provider = str(primary_provider or "").strip().lower()
        model = str(primary_model or "").strip()

        targets: list[LLMModelTarget] = []
        if provider and model:
            targets.append(
                LLMModelTarget(
                    provider=provider,
                    model=model,
                )
            )

        raw_chain = os.getenv(
            "LLM_INTENT_MODEL_CHAIN",
            "",
        )
        max_models = self._env_int(
            "LLM_ROUTER_MAX_MODELS",
            default=5,
            minimum=1,
            maximum=10,
        )

        for raw_item in raw_chain.split(","):
            item = raw_item.strip()
            if not item:
                continue

            if "::" in item:
                item_provider, item_model = item.split(
                    "::",
                    1,
                )
                target_provider = (
                    item_provider.strip().lower()
                )
                target_model = item_model.strip()
            else:
                target_provider = provider
                target_model = item

            if not target_provider or not target_model:
                continue

            target = LLMModelTarget(
                provider=target_provider,
                model=target_model,
            )
            if target not in targets:
                targets.append(target)

            if len(targets) >= max_models:
                break

        if not targets:
            raise LLMModelRouterError(
                "No valid LLM model targets were configured.",
                {
                    "llm_attempted": False,
                    "llm_used": False,
                    "success": False,
                    "attempt_count": 0,
                    "attempts": [],
                    "fallback_used": True,
                    "fallback_reason": "missing_model_configuration",
                },
            )

        return targets

    def _resolve_runtime(
        self,
        *,
        target: LLMModelTarget,
        primary_provider: str,
        primary_api_key: str,
        primary_base_url: str,
    ) -> dict[str, str]:
        provider = target.provider
        primary_provider_norm = str(
            primary_provider or ""
        ).strip().lower()

        if provider == primary_provider_norm:
            api_key = str(primary_api_key or "").strip()
            base_url = str(primary_base_url or "").strip()
        else:
            api_key = ""
            base_url = ""

        if provider == "openrouter":
            api_key = (
                api_key
                or os.getenv("OPENROUTER_API_KEY", "")
                or os.getenv("LLM_INTENT_API_KEY", "")
            ).strip()
            base_url = (
                base_url
                or os.getenv("OPENROUTER_BASE_URL", "")
                or "https://openrouter.ai/api/v1/chat/completions"
            ).strip()
        elif provider == "openai":
            api_key = (
                api_key
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("LLM_INTENT_API_KEY", "")
            ).strip()
            base_url = (
                base_url
                or os.getenv("OPENAI_BASE_URL", "")
                or "https://api.openai.com/v1/chat/completions"
            ).strip()
        else:
            api_key = (
                api_key
                or os.getenv("LLM_INTENT_API_KEY", "")
            ).strip()
            base_url = (
                base_url
                or os.getenv("LLM_INTENT_BASE_URL", "")
            ).strip()

        return {
            "api_key": api_key,
            "base_url": base_url,
        }

    def _invoke_target(
        self,
        *,
        target: LLMModelTarget,
        messages: list[dict[str, str]],
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": target.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(max_tokens),
            "response_format": {
                "type": "json_object",
            },
        }

        if target.provider == "openrouter":
            reasoning_effort = os.getenv(
                "LLM_INTENT_REASONING_EFFORT",
                "none",
            ).strip().lower()

            if reasoning_effort:
                payload["reasoning"] = {
                    "effort": reasoning_effort,
                    "exclude": True,
                }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if target.provider == "openrouter":
            headers["X-Title"] = (
                "Punk AI Audience Intelligence Intent Resolver"
            )

        request = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=int(timeout_seconds),
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        content = self._extract_message_content(
            message.get("content")
        )

        if not content:
            finish_reason = choice.get("finish_reason")
            native_finish_reason = choice.get(
                "native_finish_reason"
            )
            reasoning_present = bool(message.get("reasoning"))
            tool_calls_present = bool(message.get("tool_calls"))

            raise LLMResponseValidationError(
                "empty_response",
                (
                    "LLM response contained no usable message content. "
                    f"finish_reason={finish_reason!r}; "
                    f"native_finish_reason={native_finish_reason!r}; "
                    f"reasoning_present={reasoning_present}; "
                    f"tool_calls_present={tool_calls_present}."
                ),
            )

        return content

    @staticmethod
    def _extract_message_content(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()

        if not isinstance(value, list):
            return ""

        parts: list[str] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            text = item.get("text")

            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue

            nested_content = item.get("content")

            if (
                isinstance(nested_content, str)
                and nested_content.strip()
            ):
                parts.append(nested_content.strip())

        return "\n".join(parts).strip()

    def _classify_http_status(self, status: int) -> str:
        if status == 402:
            return "credit_or_quota"
        if status == 408:
            return "timeout"
        if status == 429:
            return "rate_limited"
        if 500 <= status <= 599:
            return "provider_server_error"
        if status in {401, 403}:
            return "authentication_error"
        if status == 400:
            return "provider_request_rejected"
        return "provider_http_error"

    def _attempt_record(
        self,
        *,
        target: LLMModelTarget,
        attempt: int,
        status: str,
        latency_ms: int,
        category: str | None,
        http_status: int | None,
        error_message: str | None,
        retryable: bool,
    ) -> dict[str, Any]:
        return {
            "provider": target.provider,
            "model": target.model,
            "attempt": int(attempt),
            "status": status,
            "latency_ms": int(latency_ms),
            "error_category": category,
            "http_status": http_status,
            "error_message": (
                str(error_message)[:500]
                if error_message
                else None
            ),
            "retryable": bool(retryable),
        }

    def _telemetry(
        self,
        *,
        success: bool,
        attempts: list[dict[str, Any]],
        started: float,
        selected_target: LLMModelTarget | None,
        primary_target: LLMModelTarget | None,
        first_failure_category: str | None,
    ) -> dict[str, Any]:
        fallback_used = bool(
            not success
            or (
                selected_target is not None
                and primary_target is not None
                and selected_target != primary_target
            )
        )

        return {
            "llm_attempted": bool(attempts),
            "llm_used": bool(success),
            "success": bool(success),
            "provider_used": (
                selected_target.provider
                if selected_target
                else None
            ),
            "model_used": (
                selected_target.model
                if selected_target
                else None
            ),
            "attempt_count": len(attempts),
            "total_latency_ms": self._elapsed_ms(started),
            "fallback_used": fallback_used,
            "fallback_reason": (
                first_failure_category
                if fallback_used
                else None
            ),
            "attempts": attempts,
        }

    def _telemetry_error_message(
        self,
        *,
        category: str | None,
        http_status: int | None,
    ) -> str:
        safe_category = str(category or "unknown_error")
        labels = {
            "missing_api_key": "Provider API key is not configured.",
            "missing_base_url": "Provider base URL is not configured.",
            "credit_or_quota": "Provider credit or quota is unavailable.",
            "timeout": "Provider request timed out.",
            "rate_limited": "Provider rate limit was reached.",
            "provider_server_error": "Provider returned a server error.",
            "authentication_error": "Provider authentication failed.",
            "provider_request_rejected": "Provider rejected the request.",
            "provider_http_error": "Provider returned an HTTP error.",
            "network_error": "Provider network request failed.",
            "invalid_provider_envelope": "Provider returned an invalid response envelope.",
            "provider_response_error": "Provider response could not be processed.",
            "invalid_json": "Model output was not valid JSON.",
            "low_confidence": "Model output confidence was below the required threshold.",
            "empty_response": "Model returned an empty response.",
        }
        message = labels.get(
            safe_category,
            "Model response failed validation or execution.",
        )
        if http_status is not None:
            return f"{message} HTTP {int(http_status)}."
        return message

    def _env_int(
        self,
        name: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw = os.getenv(name)
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            value = int(default)
        return min(max(value, minimum), maximum)

    def _env_float(
        self,
        name: str,
        *,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = os.getenv(name)
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        return min(max(value, minimum), maximum)

    def _elapsed_ms(self, started: float) -> int:
        return int((time.monotonic() - started) * 1000)
