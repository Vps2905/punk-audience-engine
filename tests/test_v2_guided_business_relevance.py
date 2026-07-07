import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent
from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent


def test_after_office_coworking_prompt_detects_office_and_evening():
    result = SemanticPromptIntelligenceAgent().analyze_prompt(
        "Find the after-office crowd around coworking hubs and business centers."
    )

    assert "office" in result["canonical_categories"]
    assert "evening" in result["dayparts"]
    assert result["confidence_score"] >= 0.65


def test_v1_prompt_selection_understands_coworking_business_prompt():
    df = pd.DataFrame(
        {
            "location_name": ["san francisco", "san francisco", "montreal", "quebec"],
            "primary_poi_type": ["coworking_space", "coworking_space", "gas_station", "casino"],
            "created_day_part": ["evening", "afternoon", "afternoon", "evening"],
            "quality_score": [0.8, 0.7, 0.9, 0.9],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Find the after-office crowd around coworking hubs and business centers.",
        cohorts=df,
    )

    assert report["poi_terms_detected"]
    assert "evening" in report["dayparts_detected"]
    assert set(selected["primary_poi_type"]) == {"coworking_space"}


def test_v2_guided_selection_blocks_unrelated_high_quality_rows(tmp_path):
    ranked = pd.DataFrame(
        {
            "location_name": ["san francisco", "san francisco", "montreal", "quebec"],
            "primary_poi_type": ["coworking_space", "coworking_space", "gas_station", "casino"],
            "created_day_part": ["evening", "afternoon", "afternoon", "evening"],
            "quality_score": [0.6, 0.5, 0.9, 0.9],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
            "final_match_score": [0.72, 0.61, 0.29, 0.31],
            "category_match_score": [0.9, 0.9, 0.0, 0.0],
            "match_type": ["adjacent_category", "broader_daypart", "data_gap_candidate", "data_gap_candidate"],
        }
    )

    ranked_path = tmp_path / "ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_from_v2_ranked(
        privacy_cohorts=ranked[
            [
                "location_name",
                "primary_poi_type",
                "created_day_part",
                "quality_score",
                "total_maid_volume",
                "privacy_status",
            ]
        ],
        v2_result={"ranked_matches_path": str(ranked_path)},
        max_rows=25,
    )

    assert report["enabled"] is True
    assert set(selected["primary_poi_type"]) == {"coworking_space"}
    assert "gas_station" not in set(selected["primary_poi_type"])
    assert "casino" not in set(selected["primary_poi_type"])
