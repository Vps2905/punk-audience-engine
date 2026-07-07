import pandas as pd

from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent
from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent


def test_semantic_fallback_caffeine_after_office_is_cafe_not_office():
    result = SemanticPromptIntelligenceAgent().analyze_prompt(
        "Find people taking a caffeine break after office near San Francisco."
    )

    assert "cafe" in result["canonical_categories"]
    assert "office" not in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_hybrid_fallback_caffeine_after_office_is_cafe_not_office(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")

    safe = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal"],
            "primary_poi_type": ["coworking_space", "cafe"],
            "created_day_part": ["evening", "evening"],
        }
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt="Find people taking a caffeine break after office near San Francisco.",
        safe_cohorts=safe,
    )

    assert result["llm_used"] is False
    assert result["resolver_mode"] == "deterministic_fallback_llm_disabled"
    assert "cafe" in result["canonical_categories"]
    assert "office" not in result["canonical_categories"]
    assert "evening" in result["dayparts"]


def test_llm_payload_uses_limited_max_tokens(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_MODEL", "test-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_INTENT_MAX_TOKENS", "500")

    config = HybridSemanticIntentAgent()._load_config()

    assert config.max_tokens == 500
