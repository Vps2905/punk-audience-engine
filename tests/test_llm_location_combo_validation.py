import pandas as pd

from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent


def test_llm_cannot_replace_montreal_with_westmount_when_user_said_montreal(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_MODEL", "test-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "test-key")

    def fake_llm(messages, config):
        return """
        {
          "business_intent": "cafe",
          "audience_goal": "visitor_footfall_audience",
          "locations": ["westmount, montreal"],
          "requested_categories": ["cafe"],
          "matched_available_poi_types": ["cafe"],
          "poi_terms": ["cafe", "coffee", "caffeine break"],
          "dayparts": ["evening"],
          "quality_intent": "balanced",
          "fallback_tolerance": "strict_exact_first",
          "data_gap_likely": true,
          "confidence_score": 0.9,
          "reasoning_summary": "coffee break maps to cafe evening"
        }
        """

    safe = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal downtown", "westmount, montreal"],
            "primary_poi_type": ["cafe", "cafe", "coworking_space"],
            "created_day_part": ["evening", "evening", "evening"],
            "quality_score": [0.5, 0.4, 0.8],
            "total_maid_volume": [10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    result = HybridSemanticIntentAgent(llm_client=fake_llm).resolve(
        prompt="Find people taking a caffeine break after office near Montreal.",
        safe_cohorts=safe,
    )

    assert result["llm_used"] is True
    assert result["locations"] == ["montreal"]
    assert result["business_intent"] == "cafe"
    assert result["exact_safe_combo_available"] is True
    assert result["data_gap_likely"] is True  # LLM said true, but exact combo is available


def test_exact_combo_validation_uses_all_safe_combinations(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv("LLM_INTENT_MODEL", "test-model")
    monkeypatch.setenv("LLM_INTENT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_INTENT_MAX_CONTEXT_ITEMS", "1")

    def fake_llm(messages, config):
        return """
        {
          "business_intent": "cafe",
          "audience_goal": "visitor_footfall_audience",
          "locations": ["montreal"],
          "requested_categories": ["cafe"],
          "matched_available_poi_types": ["cafe"],
          "poi_terms": ["cafe"],
          "dayparts": ["evening"],
          "quality_intent": "balanced",
          "fallback_tolerance": "strict_exact_first",
          "data_gap_likely": false,
          "confidence_score": 0.91,
          "reasoning_summary": "cafe evening in Montreal"
        }
        """

    rows = []
    for i in range(100):
        rows.append(
            {
                "location_name": f"other city {i}",
                "primary_poi_type": "restaurant",
                "created_day_part": "morning",
                "quality_score": 0.1,
                "total_maid_volume": 10000,
                "privacy_status": "passed",
            }
        )

    rows.append(
        {
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
            "quality_score": 0.8,
            "total_maid_volume": 10000,
            "privacy_status": "passed",
        }
    )

    safe = pd.DataFrame(rows)

    result = HybridSemanticIntentAgent(llm_client=fake_llm).resolve(
        prompt="Find people taking a caffeine break after office near Montreal.",
        safe_cohorts=safe,
    )

    assert result["exact_safe_combo_available"] is True
    assert result["data_gap_likely"] is False
    assert result["rag_context_summary"]["safe_available_combination_count"] == 101
