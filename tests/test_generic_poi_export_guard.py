import pandas as pd

from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent


def test_specific_pet_intent_does_not_export_generic_store():
    safe = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal", "montreal"],
            "primary_poi_type": ["store", "pet_store", "store"],
            "created_day_part": ["evening", "evening", "morning"],
            "quality_score": [0.9, 0.4, 0.8],
            "total_maid_volume": [10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "pet_care",
        "locations": ["montreal"],
        "requested_categories": ["pet_store", "veterinary_care"],
        "matched_available_poi_types": ["pet_store"],
        "canonical_categories": ["pet_store"],
        "poi_terms": ["pet clinics", "grooming stores", "pet supply places"],
        "dayparts": ["evening", "morning"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="Find people in Montreal who visit pet clinics, grooming stores, or pet supply places during weekends.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert set(selected["primary_poi_type"].tolist()) == {"pet_store"}
    assert "store" not in set(selected["primary_poi_type"].tolist())


def test_generic_store_can_export_when_user_really_requests_generic_stores():
    safe = pd.DataFrame(
        {
            "location_name": ["montreal"],
            "primary_poi_type": ["store"],
            "created_day_part": ["evening"],
            "quality_score": [0.7],
            "total_maid_volume": [10000],
            "privacy_status": ["passed"],
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "store",
        "locations": ["montreal"],
        "requested_categories": ["store"],
        "matched_available_poi_types": ["store"],
        "canonical_categories": ["store"],
        "poi_terms": ["stores"],
        "dayparts": ["evening"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="Find people visiting stores in Montreal during evening.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert selected.iloc[0]["primary_poi_type"] == "store"
