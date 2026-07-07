import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_new_york_does_not_auto_expand_to_times_square():
    df = pd.DataFrame(
        {
            "location_name": ["times square, new york", "new york", "montreal"],
            "primary_poi_type": ["cafe", "cafe", "cafe"],
            "created_day_part": ["evening", "evening", "evening"],
        }
    )

    locations = AudienceIntelligenceOrchestratorAgent()._extract_location_terms(
        "I need people who grab espresso and snacks after work near New York.",
        df,
    )

    assert locations == ["new york"]


def test_times_square_keeps_specific_location_only():
    df = pd.DataFrame(
        {
            "location_name": ["times square, new york", "new york", "montreal"],
            "primary_poi_type": ["cafe", "cafe", "cafe"],
            "created_day_part": ["evening", "evening", "evening"],
        }
    )

    locations = AudienceIntelligenceOrchestratorAgent()._extract_location_terms(
        "Find cafe visitors near Times Square, New York.",
        df,
    )

    assert locations == ["times square, new york"]
