import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
)


def agent():
    return object.__new__(
        AudienceIntelligenceOrchestratorAgent
    )


def test_greeting_abstains_instead_of_selecting_all():
    svc = agent()

    cohorts = pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
            },
            {
                "location_name": "san francisco",
                "primary_poi_type": "coworking_space",
                "created_day_part": "afternoon",
            },
        ]
    )

    selected, report = (
        svc._select_cohorts_for_prompt(
            prompt="Hello, how are you?",
            cohorts=cohorts,
            semantic_intent={},
        )
    )

    assert selected.empty
    assert report["filter_mode"] == (
        "needs_clarification"
    )
    assert report[
        "eligible_for_audience_selection"
    ] is False
    assert report["block_export"] is True


def test_merge_preserves_authoritative_quality():
    svc = agent()

    merged = (
        svc._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report={
                "quality_intent": "balanced",
            },
            v2_result={
                "prompt_intent": {
                    "business_intent": "restaurant",
                    "locations": ["montreal"],
                    "requested_categories": [
                        "restaurant"
                    ],
                    "matched_available_poi_types": [
                        "restaurant"
                    ],
                    "poi_terms": [
                        "restaurant",
                        "dependable people",
                    ],
                    "dayparts": ["evening"],
                    "quality_intent": "high",
                    "confidence_score": 0.95,
                    "llm_used": True,
                    "resolver_mode": (
                        "llm_rag_primary"
                    ),
                }
            },
        )
    )

    assert merged["quality_intent"] == "high"
    assert merged["locations_detected"] == [
        "montreal"
    ]
    assert merged["requested_categories"] == [
        "restaurant"
    ]
    assert merged["poi_terms_detected"] == [
        "restaurant"
    ]
    assert merged["dayparts_detected"] == [
        "evening"
    ]
    assert merged[
        "eligible_for_audience_selection"
    ] is True
    assert "dependable_people" in merged[
        "semantic_support_terms"
    ]


def test_empty_semantic_intent_needs_clarification():
    svc = agent()

    merged = (
        svc._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report={},
            v2_result={
                "prompt_intent": {
                    "business_intent": (
                        "unknown_business_intent"
                    ),
                    "locations": [],
                    "requested_categories": [],
                    "canonical_categories": [],
                    "poi_terms": [],
                    "dayparts": [],
                    "confidence_score": 0.15,
                    "llm_used": False,
                    "resolver_mode": (
                        "deterministic_fallback_llm_failed"
                    ),
                }
            },
        )
    )

    assert merged["filter_mode"] == (
        "needs_clarification"
    )
    assert merged[
        "audience_request_detected"
    ] is False
    assert merged[
        "eligible_for_audience_selection"
    ] is False
    assert set(
        merged["missing_required_constraints"]
    ) == {"location", "category"}
    assert merged["block_export"] is True


def test_v2_selector_does_not_rank_ineligible_prompt():
    svc = agent()

    selected, report = (
        svc._select_cohorts_from_v2_ranked(
            privacy_cohorts=pd.DataFrame(
                [
                    {
                        "cohort_id": "cohort_1",
                        "location_name": "montreal",
                    }
                ]
            ),
            v2_result={
                "ranked_matches": [
                    {
                        "cohort_id": "cohort_1",
                        "location_name": "montreal",
                        "primary_poi_type": (
                            "restaurant"
                        ),
                        "created_day_part": "evening",
                        "privacy_status": "passed",
                        "final_match_score": 0.9,
                        "category_match_score": 0.9,
                        "match_type": "exact_match",
                    }
                ]
            },
            prompt_filter_report={
                "filter_mode": (
                    "needs_clarification"
                ),
                "eligible_for_audience_selection": (
                    False
                ),
                "missing_required_constraints": [
                    "location",
                    "category",
                ],
            },
        )
    )

    assert selected.empty
    assert report["filter_mode"] == (
        "needs_clarification"
    )
    assert report["block_export"] is True
    assert report[
        "downstream_export_enabled"
    ] is False
