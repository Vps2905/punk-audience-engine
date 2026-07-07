import pandas as pd

from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent


def test_llm_intent_normalizes_business_intent_and_marks_combo_gap(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_MODEL", "test-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "test-key")

    def fake_llm(messages, config):
        return """
        {
          "business_intent": "Find people taking a caffeine break after office near San Francisco.",
          "audience_goal": "visitor_footfall_audience",
          "locations": ["san francisco"],
          "requested_categories": ["cafe"],
          "matched_available_poi_types": ["cafe"],
          "poi_terms": ["cafe", "caffeine break", "coffee"],
          "dayparts": ["evening"],
          "quality_intent": "balanced",
          "fallback_tolerance": "strict_exact_first",
          "data_gap_likely": false,
          "confidence_score": 0.9,
          "reasoning_summary": "caffeine break maps to cafe and after office maps to evening"
        }
        """

    safe = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal"],
            "primary_poi_type": ["coworking_space", "cafe"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.8, 0.6],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    result = HybridSemanticIntentAgent(llm_client=fake_llm).resolve(
        prompt="Find people taking a caffeine break after office near San Francisco.",
        safe_cohorts=safe,
    )

    assert result["llm_used"] is True
    assert result["business_intent"] == "cafe"
    assert result["data_gap_likely"] is True
    assert result["exact_safe_combo_available"] is False


def test_llm_intent_marks_combo_available_when_exact_match_exists(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_MODEL", "test-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "test-key")

    def fake_llm(messages, config):
        return """
        {
          "business_intent": "cafe",
          "audience_goal": "visitor_footfall_audience",
          "locations": ["montreal"],
          "requested_categories": ["cafe"],
          "matched_available_poi_types": ["cafe"],
          "poi_terms": ["cafe", "coffee"],
          "dayparts": ["evening"],
          "quality_intent": "balanced",
          "fallback_tolerance": "strict_exact_first",
          "data_gap_likely": false,
          "confidence_score": 0.92,
          "reasoning_summary": "coffee intent maps to cafe evening"
        }
        """

    safe = pd.DataFrame(
        {
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["cafe", "coworking_space"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.6, 0.8],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    result = HybridSemanticIntentAgent(llm_client=fake_llm).resolve(
        prompt="Find coffee visitors after work near Montreal.",
        safe_cohorts=safe,
    )

    assert result["llm_used"] is True
    assert result["business_intent"] == "cafe"
    assert result["data_gap_likely"] is False
    assert result["exact_safe_combo_available"] is True
