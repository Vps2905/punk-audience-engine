import json
import urllib.error

import pytest

from app.services.llm_model_router_service import (
    LLMModelRouterError,
    LLMModelRouterService,
    LLMResponseValidationError,
)


def _validator(content: str):
    value = json.loads(content)
    confidence = float(value.get("confidence_score") or 0)
    if confidence < 0.65:
        raise LLMResponseValidationError(
            "low_confidence",
            "confidence below threshold",
        )
    return value


def test_primary_credit_failure_fails_over_to_secondary(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_INTENT_MODEL_CHAIN",
        "secondary-model",
    )
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "1",
    )

    service = LLMModelRouterService()

    def fake_invoke(**kwargs):
        target = kwargs["target"]
        if target.model == "primary-model":
            raise urllib.error.HTTPError(
                url="https://example.invalid",
                code=402,
                msg="credits",
                hdrs=None,
                fp=None,
            )
        return json.dumps(
            {
                "confidence_score": 0.91,
                "locations": ["montreal"],
            }
        )

    monkeypatch.setattr(
        service,
        "_invoke_target",
        fake_invoke,
    )

    result = service.route(
        messages=[{"role": "user", "content": "hello"}],
        primary_provider="openrouter",
        primary_model="primary-model",
        primary_api_key="key",
        primary_base_url="https://example.invalid",
        timeout_seconds=10,
        max_tokens=300,
        validator=_validator,
    )

    telemetry = result["telemetry"]
    assert telemetry["llm_used"] is True
    assert telemetry["provider_used"] == "openrouter"
    assert telemetry["model_used"] == "secondary-model"
    assert telemetry["fallback_used"] is True
    assert telemetry["fallback_reason"] == "credit_or_quota"
    assert telemetry["attempt_count"] == 2


def test_invalid_primary_response_fails_over(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_INTENT_MODEL_CHAIN",
        "secondary-model",
    )
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "1",
    )

    service = LLMModelRouterService()

    def fake_invoke(**kwargs):
        if kwargs["target"].model == "primary-model":
            return "not-json"
        return json.dumps({"confidence_score": 0.9})

    monkeypatch.setattr(
        service,
        "_invoke_target",
        fake_invoke,
    )

    result = service.route(
        messages=[{"role": "user", "content": "hello"}],
        primary_provider="openrouter",
        primary_model="primary-model",
        primary_api_key="key",
        primary_base_url="https://example.invalid",
        timeout_seconds=10,
        max_tokens=300,
        validator=_validator,
    )

    assert result["telemetry"]["model_used"] == "secondary-model"
    assert result["telemetry"]["attempt_count"] == 2


def test_all_models_fail_returns_sanitized_telemetry(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_INTENT_MODEL_CHAIN",
        "secondary-model",
    )
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "1",
    )

    service = LLMModelRouterService()

    def fake_invoke(**kwargs):
        raise urllib.error.HTTPError(
            url="https://example.invalid",
            code=402,
            msg="credits",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        service,
        "_invoke_target",
        fake_invoke,
    )

    with pytest.raises(LLMModelRouterError) as caught:
        service.route(
            messages=[{"role": "user", "content": "secret prompt"}],
            primary_provider="openrouter",
            primary_model="primary-model",
            primary_api_key="secret-key",
            primary_base_url="https://example.invalid",
            timeout_seconds=10,
            max_tokens=300,
            validator=_validator,
        )

    telemetry = caught.value.telemetry
    assert telemetry["llm_used"] is False
    assert telemetry["attempt_count"] == 2
    assert telemetry["fallback_used"] is True
    assert "secret-key" not in str(telemetry)
    assert "secret prompt" not in str(telemetry)


def test_low_confidence_primary_uses_next_model(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_INTENT_MODEL_CHAIN",
        "secondary-model",
    )
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "1",
    )

    service = LLMModelRouterService()

    def fake_invoke(**kwargs):
        confidence = (
            0.4
            if kwargs["target"].model == "primary-model"
            else 0.92
        )
        return json.dumps({"confidence_score": confidence})

    monkeypatch.setattr(
        service,
        "_invoke_target",
        fake_invoke,
    )

    result = service.route(
        messages=[{"role": "user", "content": "hello"}],
        primary_provider="openrouter",
        primary_model="primary-model",
        primary_api_key="key",
        primary_base_url="https://example.invalid",
        timeout_seconds=10,
        max_tokens=300,
        validator=_validator,
    )

    assert result["telemetry"]["model_used"] == "secondary-model"
    assert (
        result["telemetry"]["attempts"][0]["error_category"]
        == "low_confidence"
    )


def test_provider_error_body_is_never_persisted_in_telemetry(
    monkeypatch,
):
    import io

    monkeypatch.setenv("LLM_INTENT_MODEL_CHAIN", "")
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "1",
    )

    secret_key = "sk-sensitive-key"
    secret_prompt = "private audience request"
    service = LLMModelRouterService()

    def fake_invoke(**kwargs):
        body = (
            f'{{"error":"{secret_key} {secret_prompt}"}}'
        ).encode("utf-8")
        raise urllib.error.HTTPError(
            url="https://example.invalid",
            code=402,
            msg="quota",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr(
        service,
        "_invoke_target",
        fake_invoke,
    )

    with pytest.raises(LLMModelRouterError) as caught:
        service.route(
            messages=[
                {"role": "user", "content": secret_prompt}
            ],
            primary_provider="openrouter",
            primary_model="primary-model",
            primary_api_key=secret_key,
            primary_base_url="https://example.invalid",
            timeout_seconds=10,
            max_tokens=300,
            validator=_validator,
        )

    telemetry_text = str(caught.value.telemetry)
    assert secret_key not in telemetry_text
    assert secret_prompt not in telemetry_text
    assert "Provider credit or quota is unavailable." in telemetry_text


def test_invalid_router_numeric_env_uses_safe_defaults(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_INTENT_MODEL_CHAIN",
        "secondary-model",
    )
    monkeypatch.setenv("LLM_ROUTER_MAX_MODELS", "invalid")
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "invalid",
    )
    monkeypatch.setenv(
        "LLM_ROUTER_RETRY_BACKOFF_SECONDS",
        "not-a-number",
    )

    service = LLMModelRouterService()

    def fake_invoke(**kwargs):
        if kwargs["target"].model == "primary-model":
            raise urllib.error.HTTPError(
                url="https://example.invalid",
                code=402,
                msg="quota",
                hdrs=None,
                fp=None,
            )
        return json.dumps({"confidence_score": 0.9})

    monkeypatch.setattr(
        service,
        "_invoke_target",
        fake_invoke,
    )

    result = service.route(
        messages=[{"role": "user", "content": "hello"}],
        primary_provider="openrouter",
        primary_model="primary-model",
        primary_api_key="key",
        primary_base_url="https://example.invalid",
        timeout_seconds=10,
        max_tokens=300,
        validator=_validator,
    )

    assert result["telemetry"]["model_used"] == "secondary-model"
    assert result["telemetry"]["attempt_count"] == 2


def test_router_numeric_env_is_clamped_to_safe_limits(
    monkeypatch,
):
    service = LLMModelRouterService()

    monkeypatch.setenv("ROUTER_TEST_INT", "999")
    monkeypatch.setenv("ROUTER_TEST_FLOAT", "inf")

    assert service._env_int(
        "ROUTER_TEST_INT",
        default=2,
        minimum=1,
        maximum=3,
    ) == 3
    assert service._env_float(
        "ROUTER_TEST_FLOAT",
        default=0.5,
        minimum=0.0,
        maximum=10.0,
    ) == 0.5
