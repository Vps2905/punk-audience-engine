import pandas as pd

from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent


def test_prompt_text_rescues_specific_available_poi_and_blocks_generic_store():
    safe = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal", "montreal"],
            "primary_poi_type": ["store", "pet_store", "pet_store"],
            "created_day_part": ["morning", "evening", "night"],
            "quality_score": [0.9, 0.4, 0.3],
            "total_maid_volume": [10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    # Simulates LLM returning broad terms. Selector should dynamically rescue
    # pet_store from available safe metadata using the prompt text.
    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "pet_care",
        "locations": ["montreal"],
        "requested_categories": ["retail", "healthcare", "beauty"],
        "matched_available_poi_types": [],
        "canonical_categories": [],
        "poi_terms": ["stores", "clinics", "grooming"],
        "dayparts": ["weekend"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="Find people in Montreal who visit pet clinics, grooming stores, or pet supply places during weekends.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert set(selected["primary_poi_type"].tolist()) == {"pet_store"}
    assert "store" not in set(selected["primary_poi_type"].tolist())
    assert report["schedule_qualifiers_detected"] == ["weekend"]


def test_weekday_is_schedule_qualifier_not_daypart_gate():
    safe = pd.DataFrame(
        {
            "location_name": ["montreal"],
            "primary_poi_type": ["car_wash"],
            "created_day_part": ["morning"],
            "quality_score": [0.7],
            "total_maid_volume": [10000],
            "privacy_status": ["passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "vehicle_service",
        "locations": ["montreal"],
        "requested_categories": ["car_wash"],
        "matched_available_poi_types": ["car_wash"],
        "canonical_categories": ["car_wash"],
        "poi_terms": ["vehicle", "car"],
        "dayparts": ["weekday", "morning"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="I want vehicle owners in Montreal who visit car washes during weekday mornings.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert selected.iloc[0]["primary_poi_type"] == "car_wash"
    assert report["dayparts_detected"] == ["morning"]
    assert report["schedule_qualifiers_detected"] == ["weekday"]
