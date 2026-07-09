from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent


def test_exact_safe_combo_requires_every_requested_location():
    agent = HybridSemanticIntentAgent()

    rag_context = {
        "safe_available_combinations": [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
            },
            {
                "location_name": "san francisco",
                "primary_poi_type": "coworking_space",
                "created_day_part": "evening",
            },
        ]
    }

    assert agent._exact_safe_combo_available(
        locations=["montreal", "san francisco"],
        poi_terms=["restaurant"],
        dayparts=["evening"],
        rag_context=rag_context,
    ) is False


def test_exact_safe_combo_true_when_all_requested_locations_have_match():
    agent = HybridSemanticIntentAgent()

    rag_context = {
        "safe_available_combinations": [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
            },
            {
                "location_name": "san francisco",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
            },
        ]
    }

    assert agent._exact_safe_combo_available(
        locations=["montreal", "san francisco"],
        poi_terms=["restaurant"],
        dayparts=["evening"],
        rag_context=rag_context,
    ) is True
