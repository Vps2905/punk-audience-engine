import os

import pandas as pd

from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent


def test_hybrid_intent_falls_back_when_llm_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")

    safe = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal"],
            "primary_poi_type": ["cafe", "coworking_space"],
            "created_day_part": ["evening", "morning"],
        }
    )

    result = HybridSemanticIntentAgent().resolve(
        prompt="I need people who grab espresso after work near San Francisco.",
        safe_cohorts=safe,
        output_dir=tmp_path,
    )

    assert result["llm_used"] is False
    assert result["resolver_mode"] == "deterministic_fallback_llm_disabled"
    assert (tmp_path / "prompt_intent.json").exists()


def test_hybrid_intent_uses_llm_json_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_MODEL", "test-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "test-key")

    def fake_llm(messages, config):
        return """
        {
          "business_intent": "cafe",
          "audience_goal": "visitor_footfall_audience",
          "locations": ["san francisco"],
          "requested_categories": ["cafe"],
          "matched_available_poi_types": ["cafe"],
          "poi_terms": ["cafe"],
          "dayparts": ["evening"],
          "quality_intent": "balanced",
          "fallback_tolerance": "strict_exact_first",
          "data_gap_likely": false,
          "confidence_score": 0.91,
          "reasoning_summary": "espresso after work maps to cafe evening intent"
        }
        """

    safe = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal"],
            "primary_poi_type": ["cafe", "coworking_space"],
            "created_day_part": ["evening", "morning"],
        }
    )

    result = HybridSemanticIntentAgent(llm_client=fake_llm).resolve(
        prompt="People taking a caffeine break after work near San Francisco.",
        safe_cohorts=safe,
    )

    assert result["llm_used"] is True
    assert result["resolver_mode"] == "llm_rag_primary"
    assert result["locations"] == ["san francisco"]
    assert "cafe" in result["poi_terms"]
    assert "evening" in result["dayparts"]
    assert result["confidence_score"] >= 0.9


def test_hybrid_rag_context_excludes_unsafe_columns(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")

    safe = pd.DataFrame(
        {
            "session_id": ["secret-session"],
            "raw_lat": [12.3],
            "raw_lng": [45.6],
            "location_name": ["san francisco"],
            "primary_poi_type": ["cafe"],
            "created_day_part": ["evening"],
        }
    )

    agent = HybridSemanticIntentAgent()
    context = agent._build_safe_rag_context(safe)

    text = str(context).lower()

    assert "secret-session" not in text
    assert "raw_lat" not in text
    assert "raw_lng" not in text
    assert "san francisco" in text
    assert "cafe" in text
