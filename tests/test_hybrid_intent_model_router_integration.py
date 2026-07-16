import json

import pandas as pd

from app.agents import hybrid_semantic_intent_agent as agent_module
from app.agents.hybrid_semantic_intent_agent import (
    HybridSemanticIntentAgent,
)
from app.services.llm_model_router_service import (
    LLMModelRouterError,
)


def _safe_cohorts():
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "cafe",
                "created_day_part": "evening",
                "quality_score": 0.8,
                "total_maid_volume": 2000,
                "privacy_status": "passed",
            }
        ]
    )


def _valid_intent():
    return {
        "business_intent": "cafe",
        "audience_goal": "reach cafe visitors",
        "locations": ["montreal"],
        "requested_categories": ["cafe"],
        "matched_available_poi_types": ["cafe"],
        "poi_terms": ["espresso", "cafe"],
        "dayparts": ["evening"],
        "quality_intent": "high",
        "fallback_tolerance": "strict_exact_first",
        "data_gap_likely": False,
        "confidence_score": 0.95,
        "reasoning_summary": "Montreal cafe evening intent.",
    }


def test_hybrid_agent_exposes_router_telemetry(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_INTENT_MODEL", "primary-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "key")

    class FakeRouter:
        def route(self, **kwargs):
            return {
                "content": json.dumps(_valid_intent()),
                "validated": _valid_intent(),
                "telemetry": {
                    "llm_attempted": True,
                    "llm_used": True,
                    "provider_used": "openrouter",
                    "model_used": "secondary-model",
                    "attempt_count": 2,
                    "total_latency_ms": 120,
                    "fallback_used": True,
                    "fallback_reason": "credit_or_quota",
                    "attempts": [
                        {
                            "provider": "openrouter",
                            "model": "primary-model",
                            "attempt": 1,
                            "status": "failed",
                            "latency_ms": 20,
                            "error_category": "credit_or_quota",
                            "http_status": 402,
                            "error_message": "credits",
                            "retryable": False,
                        },
                        {
                            "provider": "openrouter",
                            "model": "secondary-model",
                            "attempt": 1,
                            "status": "succeeded",
                            "latency_ms": 100,
                            "error_category": None,
                            "http_status": 200,
                            "error_message": None,
                            "retryable": False,
                        },
                    ],
                },
            }

    monkeypatch.setattr(
        agent_module,
        "LLMModelRouterService",
        FakeRouter,
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Reach professionals grabbing espresso near "
            "Montreal in the evening."
        ),
        safe_cohorts=_safe_cohorts(),
    )

    assert result["llm_used"] is True
    assert result["resolver_mode"] == "llm_rag_primary"
    assert result["llm_provider"] == "openrouter"
    assert result["llm_model"] == "secondary-model"
    assert result["llm_attempt_count"] == 2
    assert result["llm_fallback_used"] is True
    assert result["llm_fallback_reason"] == "credit_or_quota"


def test_hybrid_agent_falls_back_after_all_models_fail(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_INTENT_MODEL", "primary-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "key")

    class FakeRouter:
        def route(self, **kwargs):
            telemetry = {
                "llm_attempted": True,
                "llm_used": False,
                "provider_used": None,
                "model_used": None,
                "attempt_count": 2,
                "total_latency_ms": 50,
                "fallback_used": True,
                "fallback_reason": "credit_or_quota",
                "attempts": [
                    {
                        "provider": "openrouter",
                        "model": "primary-model",
                        "attempt": 1,
                        "status": "failed",
                        "latency_ms": 25,
                        "error_category": "credit_or_quota",
                        "http_status": 402,
                        "error_message": "credits",
                        "retryable": False,
                    },
                    {
                        "provider": "openrouter",
                        "model": "secondary-model",
                        "attempt": 1,
                        "status": "failed",
                        "latency_ms": 25,
                        "error_category": "credit_or_quota",
                        "http_status": 402,
                        "error_message": "credits",
                        "retryable": False,
                    },
                ],
            }
            raise LLMModelRouterError(
                "all failed",
                telemetry,
            )

    monkeypatch.setattr(
        agent_module,
        "LLMModelRouterService",
        FakeRouter,
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find people taking a caffeine break after "
            "office near Montreal."
        ),
        safe_cohorts=_safe_cohorts(),
    )

    assert result["llm_used"] is False
    assert (
        result["resolver_mode"]
        == "deterministic_fallback_llm_failed"
    )
    assert result["llm_attempted"] is True
    assert result["llm_attempt_count"] == 2
    assert result["llm_fallback_used"] is True
    assert result["canonical_categories"] == ["cafe"]
    assert result["dayparts"] == ["evening"]
