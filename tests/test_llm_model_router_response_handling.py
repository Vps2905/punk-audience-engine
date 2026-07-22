import json
import urllib.request

import pytest

from app.services.llm_model_router_service import (
    LLMModelRouterError,
    LLMModelRouterService,
    LLMModelTarget,
    LLMResponseValidationError,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def invoke(service: LLMModelRouterService) -> str:
    return service._invoke_target(
        target=LLMModelTarget(
            provider="openrouter",
            model="test/model",
        ),
        messages=[
            {
                "role": "user",
                "content": "Return JSON.",
            }
        ],
        api_key="test-key",
        base_url="https://example.invalid/chat",
        timeout_seconds=1,
        max_tokens=100,
    )


def test_extracts_plain_string_content(monkeypatch):
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"confidence_score": 0.9}'
                },
            }
        ]
    }

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    assert invoke(LLMModelRouterService()) == (
        '{"confidence_score": 0.9}'
    )


def test_extracts_openai_text_content_blocks(monkeypatch):
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"confidence_score": 0.91}',
                        }
                    ]
                },
            }
        ]
    }

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    assert invoke(LLMModelRouterService()) == (
        '{"confidence_score": 0.91}'
    )


def test_empty_content_records_safe_provider_metadata(monkeypatch):
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "native_finish_reason": "MAX_TOKENS",
                "message": {
                    "content": "",
                    "reasoning": "hidden reasoning",
                },
            }
        ]
    }

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(LLMResponseValidationError) as exc_info:
        invoke(LLMModelRouterService())

    assert exc_info.value.category == "empty_response"
    assert "finish_reason='length'" in str(exc_info.value)
    assert "reasoning_present=True" in str(exc_info.value)


def test_validator_json_error_is_classified_as_invalid_json(
    monkeypatch,
):
    monkeypatch.setenv("LLM_INTENT_MODEL_CHAIN", "")
    monkeypatch.setenv(
        "LLM_ROUTER_MAX_ATTEMPTS_PER_MODEL",
        "1",
    )

    service = LLMModelRouterService()

    monkeypatch.setattr(
        service,
        "_invoke_target",
        lambda **kwargs: "not valid json",
    )

    with pytest.raises(LLMModelRouterError) as exc_info:
        service.route(
            messages=[
                {
                    "role": "user",
                    "content": "Return JSON.",
                }
            ],
            primary_provider="openrouter",
            primary_model="test/model",
            primary_api_key="test-key",
            primary_base_url=(
                "https://example.invalid/chat"
            ),
            timeout_seconds=1,
            max_tokens=100,
            validator=json.loads,
        )

    attempts = exc_info.value.telemetry["attempts"]

    assert len(attempts) == 1
    assert attempts[0]["error_category"] == "invalid_json"
    assert attempts[0]["error_message"] == (
        "Model output was not valid JSON."
    )
