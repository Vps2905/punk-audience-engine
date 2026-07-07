import pandas as pd

from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent


def _safe_city_data():
    return pd.DataFrame(
        {
            "location_name": [
                "san francisco",
                "montreal",
                "montreal",
                "montreal",
            ],
            "primary_poi_type": [
                "coworking_space",
                "yoga_studio",
                "clothing_store",
                "shopping_mall",
            ],
            "created_day_part": [
                "evening",
                "afternoon",
                "evening",
                "evening",
            ],
            "quality_score": [0.9, 0.8, 0.7, 0.6],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
        }
    )


def test_london_prompt_blocks_cross_city_export_when_llm_misses_location():
    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "premium_lifestyle_retail",
        "locations": [],
        "requested_categories": ["retail", "store"],
        "matched_available_poi_types": [],
        "canonical_categories": [],
        "poi_terms": ["boutiques", "shopping malls", "clothing stores", "fashion stores", "home decor"],
        "dayparts": ["weekend", "evening"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="I manage a premium lifestyle and fashion store in London. Build an audience of people who visit boutiques, shopping malls, clothing stores, fashion stores, or home decor shops during weekend evenings.",
        cohorts=_safe_city_data(),
        intent=intent,
    )

    assert selected.empty
    assert report["locations_detected"] == ["london"]
    assert report["filter_mode"] == "location_category_gap_no_export"


def test_austin_prompt_blocks_cross_city_export_when_llm_misses_location():
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
        cohorts=_safe_city_data(),
        intent=intent,
    )

    assert selected.empty
    assert report["locations_detected"] == ["austin"]
    assert report["filter_mode"] == "location_category_gap_no_export"


def test_dubai_prompt_blocks_cross_city_export_when_llm_misses_location():
    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "luxury_travel",
        "locations": [],
        "requested_categories": ["travel", "leisure"],
        "matched_available_poi_types": [],
        "canonical_categories": [],
        "poi_terms": ["hotels", "travel agencies", "luggage stores", "airports", "tourist attractions"],
        "dayparts": ["evening"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="I run a luxury travel package business in Dubai. Find people who visit hotels, travel agencies, luggage stores, airports, tourist attractions, or premium leisure places before evening travel.",
        cohorts=_safe_city_data(),
        intent=intent,
    )

    assert selected.empty
    assert report["locations_detected"] == ["dubai"]
    assert report["filter_mode"] == "location_category_gap_no_export"
