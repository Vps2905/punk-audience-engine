from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_orchestrator_merges_v2_ai_intent_when_old_prompt_filter_misses():
    agent = AudienceIntelligenceOrchestratorAgent()

    old_report = {
        "filter_mode": "all",
        "locations_detected": [],
        "poi_terms_detected": [],
        "dayparts_detected": [],
    }

    v2_result = {
        "prompt_intent": {
            "llm_used": True,
            "resolver_mode": "llm_rag_primary",
            "confidence_score": 0.91,
            "data_gap_likely": False,
            "locations": ["san francisco"],
            "matched_available_poi_types": ["cafe"],
            "dayparts": ["evening"],
        }
    }

    merged = agent._merge_v2_intent_into_prompt_filter_report(
        prompt_filter_report=old_report,
        v2_result=v2_result,
    )

    assert merged["locations_detected"] == ["san_francisco"]
    assert merged["poi_terms_detected"] == ["cafe"]
    assert merged["dayparts_detected"] == ["evening"]
    assert merged["filter_mode"] == "location+poi+daypart"
    assert merged["ai_intent_merge"]["llm_used"] is True
