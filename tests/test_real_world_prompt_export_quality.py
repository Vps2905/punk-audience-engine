import pandas as pd

from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent


def test_pet_service_prompt_exports_only_pet_supported_pois():
    safe = pd.DataFrame(
        {
            "location_name": [
                "montreal",
                "montreal",
                "montreal",
                "montreal",
                "montreal",
                "montreal",
            ],
            "primary_poi_type": [
                "pet_store",
                "pet_care",
                "grocery_store",
                "clothing_store",
                "hair_salon",
                "toy_store",
            ],
            "created_day_part": [
                "evening",
                "night",
                "evening",
                "evening",
                "afternoon",
                "morning",
            ],
            "quality_score": [0.4, 0.4, 0.9, 0.8, 0.7, 0.6],
            "total_maid_volume": [10000] * 6,
            "privacy_status": ["passed"] * 6,
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "pet_grooming_care",
        "locations": ["montreal"],
        "requested_categories": ["retail", "beauty"],
        "matched_available_poi_types": [],
        "canonical_categories": [],
        "poi_terms": ["pet", "pet stores", "pet care", "grooming", "animal care"],
        "dayparts": [],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="I run a premium pet grooming and care service in Montreal. I want to reach pet owners who recently visited pet stores, pet-care places, grooming-related stores, or animal-care locations.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert set(selected["primary_poi_type"].tolist()) == {"pet_store", "pet_care"}


def test_lifestyle_retail_prompt_does_not_export_pet_or_toy_store():
    safe = pd.DataFrame(
        {
            "location_name": [
                "montreal",
                "montreal",
                "montreal",
                "montreal",
                "montreal",
            ],
            "primary_poi_type": [
                "clothing_store",
                "shopping_mall",
                "pet_store",
                "toy_store",
                "grocery_store",
            ],
            "created_day_part": [
                "evening",
                "evening",
                "evening",
                "morning",
                "evening",
            ],
            "quality_score": [0.6, 0.6, 0.9, 0.8, 0.7],
            "total_maid_volume": [10000] * 5,
            "privacy_status": ["passed"] * 5,
        }
    )

    intent = {
        "llm_used": True,
        "resolver_mode": "llm_rag_primary",
        "business_intent": "premium_lifestyle_retail",
        "locations": ["montreal"],
        "requested_categories": ["retail", "store"],
        "matched_available_poi_types": [],
        "canonical_categories": [],
        "poi_terms": ["boutiques", "malls", "clothing stores", "home decor", "lifestyle retail"],
        "dayparts": ["weekend", "evening"],
    }

    selected, report = AutonomousPromptCohortSelectorAgent().select(
        prompt="I manage a premium lifestyle store in Montreal. Build an audience of people who visit boutiques, malls, clothing stores, home decor shops, or lifestyle retail places during weekend evenings.",
        cohorts=safe,
        intent=intent,
    )

    assert not selected.empty
    assert set(selected["primary_poi_type"].tolist()) == {"clothing_store", "shopping_mall"}
