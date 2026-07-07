import pandas as pd

from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent


def test_requested_city_with_no_safe_data_returns_zero_selected_and_zero_export_candidates():
    safe = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal"],
            "primary_poi_type": ["coworking_space", "yoga_studio"],
            "created_day_part": ["evening", "afternoon"],
            "quality_score": [0.9, 0.8],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "sports_recovery_physiotherapy",
        "locations": [],
        "requested_categories": ["gym", "wellness"],
        "matched_available_poi_types": [],
        "canonical_categories": [],
        "poi_terms": ["gyms", "yoga studios", "sports stores", "physiotherapy centers", "wellness"],
        "dayparts": [],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="I’m launching a sports recovery and physiotherapy service in Austin. Find people who visit gyms, yoga studios, sports stores, physiotherapy centers, or wellness places after workouts.",
        cohorts=safe,
        intent=intent,
    )

    assert selected.empty
    assert report["locations_detected"] == ["austin"]
    assert report["filter_mode"] == "location_category_gap_no_export"
