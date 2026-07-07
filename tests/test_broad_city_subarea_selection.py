import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_montreal_prompt_matches_montreal_downtown_cafe_evening():
    df = pd.DataFrame(
        {
            "location_name": ["montreal downtown", "san francisco", "montreal"],
            "primary_poi_type": ["cafe", "coworking_space", "restaurant"],
            "created_day_part": ["evening", "evening", "morning"],
            "quality_score": [0.8, 0.7, 0.6],
            "total_maid_volume": [10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Find people taking a caffeine break after office near Montreal.",
        cohorts=df,
    )

    assert not selected.empty
    assert len(selected) == 1
    assert selected.iloc[0]["location_name"] == "montreal downtown"
    assert selected.iloc[0]["primary_poi_type"] == "cafe"
    assert selected.iloc[0]["created_day_part"] == "evening"
    assert report["filter_mode"] == "location+poi+daypart"


def test_requested_subarea_does_not_fallback_to_broader_city():
    df = pd.DataFrame(
        {
            "location_name": ["montreal"],
            "primary_poi_type": ["cafe"],
            "created_day_part": ["evening"],
            "quality_score": [0.8],
            "total_maid_volume": [10000],
            "privacy_status": ["passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Find people taking a caffeine break after office near Westmount Montreal.",
        cohorts=df,
    )

    assert selected.empty
    assert report["filter_mode"] == "location_category_gap_no_export"


def test_san_francisco_cafe_gap_still_blocks_coworking_fallback():
    df = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal downtown"],
            "primary_poi_type": ["coworking_space", "cafe"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.9, 0.8],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Find people taking a caffeine break after office near San Francisco.",
        cohorts=df,
    )

    assert selected.empty
    assert report["filter_mode"] == "location_category_gap_no_export"
