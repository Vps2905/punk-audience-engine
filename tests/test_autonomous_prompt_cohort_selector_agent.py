import pandas as pd

from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent


def test_dynamic_selector_matches_broad_city_to_subarea_without_city_rule():
    safe = pd.DataFrame(
        {
            "location_name": ["montreal downtown", "san francisco"],
            "primary_poi_type": ["cafe", "coworking_space"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.8, 0.9],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "cafe",
        "locations": ["montreal"],
        "requested_categories": ["cafe"],
        "matched_available_poi_types": ["cafe"],
        "canonical_categories": ["cafe"],
        "poi_terms": ["coffee", "caffeine break"],
        "dayparts": ["evening"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="Find people taking a caffeine break after office near Montreal.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert selected.iloc[0]["location_name"] == "montreal downtown"
    assert selected.iloc[0]["primary_poi_type"] == "cafe"
    assert report["selector_mode"] == "autonomous_hybrid_rag_embedding_selector"


def test_dynamic_selector_blocks_cross_category_fallback():
    safe = pd.DataFrame(
        {
            "location_name": ["san francisco", "montreal downtown"],
            "primary_poi_type": ["coworking_space", "cafe"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.9, 0.8],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "cafe",
        "locations": ["san francisco"],
        "requested_categories": ["cafe"],
        "matched_available_poi_types": ["cafe"],
        "canonical_categories": ["cafe"],
        "poi_terms": ["coffee", "caffeine break"],
        "dayparts": ["evening"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="Find people taking a caffeine break after office near San Francisco.",
        cohorts=safe,
        intent=intent,
    )

    assert selected.empty
    assert report["filter_mode"] == "location_category_gap_no_export"


def test_dynamic_selector_is_not_tied_to_known_cities_or_known_categories():
    safe = pd.DataFrame(
        {
            "location_name": ["river city central", "mountain district"],
            "primary_poi_type": ["artisan_tea_house", "robotics_lab"],
            "created_day_part": ["night", "morning"],
            "quality_score": [0.7, 0.9],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "artisan_tea_house",
        "locations": ["river city"],
        "requested_categories": ["artisan_tea_house"],
        "matched_available_poi_types": ["artisan_tea_house"],
        "canonical_categories": ["artisan_tea_house"],
        "poi_terms": ["tea house"],
        "dayparts": ["night"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="Find premium tea visitors at night near River City.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert selected.iloc[0]["location_name"] == "river city central"
    assert selected.iloc[0]["primary_poi_type"] == "artisan_tea_house"
