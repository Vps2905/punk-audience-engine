import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_hybrid_intent_recovers_casino_when_llm_returns_unknown():
    agent = AudienceIntelligenceOrchestratorAgent()

    safe_cohorts = pd.DataFrame(
        [
            {
                "location_name": "quebec",
                "primary_poi_type": "casino",
                "created_day_part": "afternoon",
                "quality_score": 0.58,
                "privacy_status": "passed",
            },
            {
                "location_name": "quebec",
                "primary_poi_type": "casino",
                "created_day_part": "evening",
                "quality_score": 0.47,
                "privacy_status": "passed",
            },
        ]
    )

    prompt = (
        "I’m promoting a casino entertainment offer in Quebec. "
        "Find people who visit casinos, gaming venues, or tourist entertainment places "
        "during afternoon or evening hours."
    )

    prompt_filter_report = {
        "business_intent": "unknown_business_intent",
        "locations": ["quebec"],
        "dayparts": ["afternoon", "evening"],
    }

    intent = agent._build_hybrid_retrieval_intent(
        prompt=prompt,
        prompt_filter_report=prompt_filter_report,
        privacy_cohorts=safe_cohorts,
    )

    assert intent["business_intent"] == "casino"
    assert "casino" in intent["poi_terms"]
    assert "casino" in intent["requested_categories"]
    assert intent["locations"] == ["quebec"]
    assert set(intent["dayparts"]) == {"afternoon", "evening"}
