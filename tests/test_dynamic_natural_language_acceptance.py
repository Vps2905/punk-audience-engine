import pandas as pd

from app.agents.hybrid_semantic_intent_agent import (
    HybridSemanticIntentAgent,
)


def safe_cohorts():
    return pd.DataFrame(
        {
            "location_name": [
                "montreal",
                "vancouver",
                "new york",
                "new york",
                "oslo",
            ],
            "primary_poi_type": [
                "cafe",
                "shopping_mall",
                "shopping_mall",
                "restaurant",
                "cafe",
            ],
            "created_day_part": [
                "evening",
                "evening",
                "afternoon",
                "afternoon",
                "evening",
            ],
            "quality_score": [0.8] * 5,
            "total_maid_volume": [1000] * 5,
            "privacy_status": ["passed"] * 5,
        }
    )


def assert_dynamic_resolver(result):
    assert result["resolver_mode"] in {
        "llm_rag_primary",
        "local_semantic_primary_llm_disabled",
        "local_semantic_fallback_llm_failed",
    }


def test_understands_after_leaving_work_as_evening(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "I want professionals who grab espresso and a snack "
            "after leaving work in Montreal."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["montreal"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_understands_after_finishing_work_as_evening(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find people who go shopping after finishing work "
            "in Vancouver."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["vancouver"]
    assert "retail" in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_preserves_multiple_requested_categories(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find shopping mall and restaurant visitors "
            "in the afternoon near New York."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["new york"]
    assert set(result["canonical_categories"]) >= {
        "retail",
        "restaurant",
    }
    assert "afternoon" in result["dayparts"]


def test_location_is_derived_from_safe_rag_context(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Reach people stopping for coffee on their "
            "way home from work near Oslo."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["oslo"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_semantic_resolver_does_not_invent_daypart(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_LLM_INTENT",
        "false",
    )
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find coffee shop visitors near Montreal."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["montreal"]
    assert "cafe" in result["canonical_categories"]
    assert result["dayparts"] == []



def test_understands_completed_professional_duties_as_evening(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_LLM_INTENT",
        "false",
    )
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Reach espresso customers once their "
            "professional duties are over in Montreal."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["montreal"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_understands_end_of_work_shift_as_evening(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_LLM_INTENT",
        "false",
    )
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find Montreal coffee customers once "
            "their daily work shift has ended."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert_dynamic_resolver(result)
    assert result["locations"] == ["montreal"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_local_semantic_fallback_when_llm_router_fails(
    monkeypatch,
):
    from app.services.llm_model_router_service import (
        LLMModelRouterError,
    )

    monkeypatch.setenv(
        "ENABLE_LLM_INTENT",
        "true",
    )
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )

    def fail_router(*args, **kwargs):
        raise LLMModelRouterError(
            "provider unavailable",
            {
                "llm_attempted": True,
                "llm_used": False,
                "attempt_count": 1,
                "total_latency_ms": 1,
                "fallback_used": True,
                "fallback_reason": "provider_failed",
                "attempts": [],
            },
        )

    agent = HybridSemanticIntentAgent()

    monkeypatch.setattr(
        agent,
        "_call_llm",
        fail_router,
    )

    result = agent.resolve(
        prompt=(
            "Reach people stopping for coffee on "
            "their way home from work near Oslo."
        ),
        safe_cohorts=safe_cohorts(),
    )

    assert result["resolver_mode"] == (
        "local_semantic_fallback_llm_failed"
    )
    assert result["local_semantic_used"] is True
    assert result["locations"] == ["oslo"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]
