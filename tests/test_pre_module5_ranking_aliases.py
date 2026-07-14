import pandas as pd

from app.services.audience_match_ranking_service import (
    DynamicAudienceRankingService,
)


def test_cafe_alias_is_exact_and_ranks_above_restaurant():
    cohorts = pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "quality_score": 0.8,
                "total_maid_volume": 100000,
                "privacy_status": "passed",
            },
            {
                "location_name": "montreal downtown",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "quality_score": 0.1,
                "total_maid_volume": 20000,
                "privacy_status": "passed",
            },
        ]
    )

    intent = {
        "locations": ["montreal"],
        "canonical_categories": ["cafe"],
        "dayparts": ["evening"],
    }

    ranked = DynamicAudienceRankingService().rank(
        cohorts,
        intent,
        {"freshness_status": "stale"},
    )

    assert ranked.iloc[0]["primary_poi_type"] == "coffee_shop"
    assert ranked.iloc[0]["category_match_score"] == 1.0
    assert ranked.iloc[0]["match_type"] == "exact_match"

    restaurant = ranked[
        ranked["primary_poi_type"] == "restaurant"
    ].iloc[0]

    assert restaurant["category_match_score"] < 0.99
    assert restaurant["match_type"] == "adjacent_category"
