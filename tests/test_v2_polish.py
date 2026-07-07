import pandas as pd

from app.agents.autonomous_mutation_agent import AutonomousMutationAgent
from app.services.audience_match_ranking_service import DynamicAudienceRankingService


def test_v2_ranking_labels_daypart_mismatch_as_broader_daypart():
    df = pd.DataFrame(
        {
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
            "created_day_part": ["afternoon"],
            "quality_score": [0.7],
            "total_maid_volume": [10000],
            "privacy_status": ["passed"],
        }
    )

    intent = {
        "locations": ["montreal"],
        "canonical_categories": ["restaurant"],
        "dayparts": ["evening"],
    }

    ranked = DynamicAudienceRankingService().rank(df, intent, {"freshness_status": "fresh"})

    assert ranked.iloc[0]["match_type"] == "broader_daypart"


def test_v2_mutation_keeps_all_requested_data_gaps():
    ranked = pd.DataFrame(
        {
            "location_name": ["montreal"] * 20,
            "primary_poi_type": ["restaurant"] * 20,
            "created_day_part": ["evening"] * 20,
            "final_match_score": [0.5] * 20,
            "confidence_score": [0.5] * 20,
            "match_type": ["adjacent_category"] * 20,
            "match_reason": ["fallback"] * 20,
        }
    )

    intent = {
        "locations": ["san francisco"],
        "canonical_categories": ["restaurant", "cafe"],
        "dayparts": ["evening"],
    }

    result = AutonomousMutationAgent().generate_suggestions(
        prompt_intent=intent,
        ranked_cohorts=ranked,
        max_suggestions=10,
    )

    gaps = [
        item for item in result["mutation_suggestions"]
        if item["mutation_type"] == "data_gap"
    ]

    gap_keys = {(item["location_name"], item["primary_poi_type"]) for item in gaps}

    assert ("san francisco", "restaurant") in gap_keys
    assert ("san francisco", "cafe") in gap_keys
