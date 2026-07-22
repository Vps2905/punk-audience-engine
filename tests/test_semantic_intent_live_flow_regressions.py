import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
)
from app.agents.hybrid_semantic_intent_agent import (
    HybridSemanticIntentAgent,
)


def realistic_safe_cohorts():
    return pd.DataFrame(
        {
            "location_name": [
                "montreal",
                "montreal",
                "montreal",
                "new york",
                "new york",
                "new york",
            ],
            "primary_poi_type": [
                "cafe",
                "cafe",
                "restaurant",
                "shopping_mall",
                "restaurant",
                "fast_food_restaurant",
            ],
            "created_day_part": [
                "morning",
                "evening",
                "night",
                "afternoon",
                "afternoon",
                "afternoon",
            ],
            "quality_score": [0.8] * 6,
            "total_maid_volume": [1000] * 6,
            "privacy_status": ["passed"] * 6,
        }
    )


def enable_local_semantic(monkeypatch):
    monkeypatch.setenv(
        "ENABLE_LLM_INTENT",
        "false",
    )
    monkeypatch.setenv(
        "ENABLE_LOCAL_SEMANTIC_INTENT",
        "true",
    )


def test_semantic_location_survives_missing_source_coverage(
    monkeypatch,
):
    enable_local_semantic(monkeypatch)

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find people who go shopping after "
            "finishing work in Vancouver."
        ),
        safe_cohorts=realistic_safe_cohorts(),
    )

    assert result["resolver_mode"] == (
        "local_semantic_primary_llm_disabled"
    )
    assert result["local_semantic_used"] is True
    assert result["locations"] == ["vancouver"]
    assert "retail" in result["canonical_categories"]
    assert result["dayparts"] == ["evening"]
    assert result["data_gap_likely"] is True


def test_no_time_is_not_inferred_from_coffee_association(
    monkeypatch,
):
    enable_local_semantic(monkeypatch)

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find coffee shop visitors near Montreal."
        ),
        safe_cohorts=realistic_safe_cohorts(),
    )

    assert result["locations"] == ["montreal"]
    assert "cafe" in result["canonical_categories"]
    assert result["dayparts"] == []


def test_provider_restaurant_types_stay_one_category(
    monkeypatch,
):
    enable_local_semantic(monkeypatch)

    result = HybridSemanticIntentAgent().resolve(
        prompt=(
            "Find shopping mall and restaurant "
            "visitors in the afternoon near New York."
        ),
        safe_cohorts=realistic_safe_cohorts(),
    )

    assert set(result["canonical_categories"]) == {
        "retail",
        "restaurant",
    }
    assert result["dayparts"] == ["afternoon"]


def test_semantic_merge_replaces_partial_parser_result():
    agent = AudienceIntelligenceOrchestratorAgent()

    merged = (
        agent._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report={
                "locations_detected": ["new york"],
                "poi_terms_detected": [
                    "restaurant",
                    "food",
                    "meal_takeaway",
                ],
                "dayparts_detected": ["afternoon"],
                "filter_mode": "location+poi+daypart",
            },
            v2_result={
                "prompt_intent": {
                    "resolver_mode": (
                        "local_semantic_primary_llm_disabled"
                    ),
                    "local_semantic_used": True,
                    "llm_used": False,
                    "locations": ["new york"],
                    "requested_categories": [
                        "restaurant",
                        "retail",
                    ],
                    "canonical_categories": [
                        "restaurant",
                        "retail",
                    ],
                    "matched_available_poi_types": [
                        "restaurant",
                        "shopping_mall",
                        "store",
                    ],
                    "poi_terms": [
                        "restaurant",
                        "retail",
                        "shopping_mall",
                    ],
                    "dayparts": ["afternoon"],
                    "confidence_score": 0.93,
                }
            },
        )
    )

    assert "restaurant" in merged[
        "poi_terms_detected"
    ]
    assert "retail" in merged[
        "poi_terms_detected"
    ]
    assert "shopping_mall" in merged[
        "poi_terms_detected"
    ]
    assert merged[
        "semantic_intent_authoritative"
    ] is True


def test_multi_category_export_allowlist_is_a_union():
    agent = AudienceIntelligenceOrchestratorAgent()

    allowed = (
        agent._allowed_export_poi_terms_for_request(
            [
                "restaurant",
                "retail",
                "shopping_mall",
            ]
        )
    )

    assert "restaurant" in allowed
    assert "meal_takeaway" in allowed
    assert "shopping_mall" in allowed
    assert "clothing_store" in allowed
