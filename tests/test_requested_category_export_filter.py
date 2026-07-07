import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_coffee_request_filters_out_restaurant_and_shawarma_exports():
    agent = AudienceIntelligenceOrchestratorAgent()

    df = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal", "montreal", "montreal"],
            "primary_poi_type": ["cafe", "restaurant", "shawarma_restaurant", "middle_eastern_restaurant"],
            "created_day_part": ["evening", "evening", "evening", "evening"],
            "quality_score": [0.4, 0.8, 0.9, 0.7],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
            "final_match_score": [0.80, 0.77, 0.76, 0.75],
            "category_match_score": [0.95, 0.90, 0.89, 0.88],
            "match_type": ["exact_match", "adjacent_category", "adjacent_category", "adjacent_category"],
        }
    )

    selected, report = agent._filter_to_requested_export_category(
        df,
        {
            "poi_terms_detected": ["cafe", "coffee"],
            "locations_detected": ["montreal"],
            "dayparts_detected": ["evening"],
        },
    )

    assert report["enabled"] is True
    assert set(selected["primary_poi_type"]) == {"cafe"}
    assert "restaurant" not in set(selected["primary_poi_type"])
    assert "shawarma_restaurant" not in set(selected["primary_poi_type"])


def test_restaurant_request_still_allows_restaurant_family_exports():
    agent = AudienceIntelligenceOrchestratorAgent()

    df = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal", "montreal", "montreal"],
            "primary_poi_type": ["cafe", "restaurant", "shawarma_restaurant", "middle_eastern_restaurant"],
            "created_day_part": ["evening", "evening", "evening", "evening"],
        }
    )

    selected, report = agent._filter_to_requested_export_category(
        df,
        {
            "poi_terms_detected": ["restaurant", "food"],
            "locations_detected": ["montreal"],
            "dayparts_detected": ["evening"],
        },
    )

    assert report["enabled"] is True
    assert set(selected["primary_poi_type"]) == {
        "restaurant",
        "shawarma_restaurant",
        "middle_eastern_restaurant",
    }
    assert "cafe" not in set(selected["primary_poi_type"])
