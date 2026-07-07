import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_caffeine_break_after_office_maps_to_cafe_not_office():
    agent = AudienceIntelligenceOrchestratorAgent()

    terms = agent._extract_poi_terms(
        "Find people taking a caffeine break after office near San Francisco."
    )

    assert "cafe" in terms
    assert "coffee" in terms
    assert "coworking_space" not in terms
    assert "corporate_office" not in terms


def test_caffeine_break_sf_blocks_export_when_no_sf_cafe_exists():
    df = pd.DataFrame(
        {
            "location_name": ["san francisco", "san francisco", "montreal"],
            "primary_poi_type": ["coworking_space", "corporate_office", "cafe"],
            "created_day_part": ["evening", "evening", "evening"],
            "quality_score": [0.8, 0.7, 0.6],
            "total_maid_volume": [10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Find people taking a caffeine break after office near San Francisco.",
        cohorts=df,
    )

    assert report["filter_mode"] == "location_category_gap_no_export"
    assert selected.empty
    assert "cafe" in report["poi_terms_detected"]
    assert "san francisco" in report["locations_detected"]
    assert "evening" in report["dayparts_detected"]


def test_flexible_workspaces_still_maps_to_coworking():
    agent = AudienceIntelligenceOrchestratorAgent()

    terms = agent._extract_poi_terms(
        "Find young professionals around flexible workspaces after office hours."
    )

    assert "coworking_space" in terms
    assert "corporate_office" in terms
