import pytest
import pandas as pd
from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent

def test_end_to_end_orchestrator_quality_intent_high():
    agent = AudienceIntelligenceOrchestratorAgent()
    privacy_cohorts = pd.DataFrame(
        {
            "cohort_id": ["c1", "c2", "c3"],
            "location_name": ["montreal", "montreal", "montreal"],
            "primary_poi_type": ["restaurant", "restaurant", "restaurant"],
            "created_day_part": ["evening", "evening", "evening"],
            "quality_score": [0.85, 0.40, 0.35],
            "privacy_status": ["passed", "passed", "passed"],
            "final_match_score": [0.9, 0.9, 0.9],
            "match_type": ["exact_match", "exact_match", "exact_match"],
            "category_match_score": [1.0, 1.0, 1.0],
        }
    )

    # We mock the V2 output where quality_intent is resolved as high
    v2_result = {
        "prompt_intent": {
            "quality_intent": "high",
            "llm_used": True,
            "confidence_score": 0.95,
        },
        "ranked_matches": privacy_cohorts.to_dict("records"),
        "status": "completed",
    }

    prompt_filter_report = {
        "locations_detected": ["montreal"],
        "poi_terms_detected": ["restaurant"],
        "dayparts_detected": ["evening"],
        "filter_mode": "location+poi+daypart"
    }

    selected_cohorts, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=privacy_cohorts,
        v2_result=v2_result,
        prompt_filter_report=agent._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report, v2_result
        ),
        max_rows=10
    )

    assert report["quality_intent"] == "high"
    assert report["quality_policy_report"]["quality_intent"] == "high"
    assert report["quality_policy_report"]["quality_policy_status"] == "pending_final_quality_evaluation"
    assert report["quality_policy_report"]["quality_candidates_after"] == report["quality_policy_report"]["quality_candidates_before"]
    assert len(selected_cohorts) == 3

def test_end_to_end_orchestrator_quality_intent_balanced():
    agent = AudienceIntelligenceOrchestratorAgent()
    privacy_cohorts = pd.DataFrame(
        {
            "cohort_id": ["c1", "c2", "c3"],
            "location_name": ["montreal", "montreal", "montreal"],
            "primary_poi_type": ["restaurant", "restaurant", "restaurant"],
            "created_day_part": ["evening", "evening", "evening"],
            "quality_score": [0.85, 0.40, 0.35],
            "privacy_status": ["passed", "passed", "passed"],
            "final_match_score": [0.9, 0.9, 0.9],
            "match_type": ["exact_match", "exact_match", "exact_match"],
            "category_match_score": [1.0, 1.0, 1.0],
        }
    )

    # Balanced
    v2_result = {
        "prompt_intent": {
            "quality_intent": "balanced",
            "llm_used": True,
            "confidence_score": 0.95,
        },
        "ranked_matches": privacy_cohorts.to_dict("records"),
        "status": "completed",
    }

    prompt_filter_report = {
        "locations_detected": ["montreal"],
        "poi_terms_detected": ["restaurant"],
        "dayparts_detected": ["evening"],
        "filter_mode": "location+poi+daypart"
    }

    selected_cohorts, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=privacy_cohorts,
        v2_result=v2_result,
        prompt_filter_report=agent._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report, v2_result
        ),
        max_rows=10
    )

    assert report["quality_intent"] == "balanced"
    assert report["quality_policy_report"]["quality_intent"] == "balanced"
    assert report["quality_policy_report"]["quality_candidates_after"] == report["quality_policy_report"]["quality_candidates_before"]
    assert len(selected_cohorts) == 3

def test_merge_report_does_not_overwrite_high_with_balanced():
    agent = AudienceIntelligenceOrchestratorAgent()

    prompt_filter_report = {
        "quality_intent": "high"
    }

    v2_result = {
        "prompt_intent": {
            # Maybe it didn't resolve quality intent or returned garbage
            "quality_intent": "invalid",
        }
    }

    merged = agent._merge_v2_intent_into_prompt_filter_report(
        prompt_filter_report, v2_result
    )

    # Precedence: v2_quality, then existing, then balanced. But v2_quality is invalid.
    # So it should be existing_quality = high, which is valid, so it should stay high!
    # Let's ensure raw_quality logic handles it properly.
    assert merged["quality_intent"] == "high"

def test_live_flow_propagation_high():
    agent = AudienceIntelligenceOrchestratorAgent()
    # prompt_intent high -> prompt_filter_report high -> quality_policy_report high -> final summary high
    prompt_filter_report = {
        "locations_detected": [
            "montreal"
        ],
        "poi_terms_detected": [
            "restaurant"
        ],
        "requested_categories": [
            "restaurant"
        ],
        "dayparts_detected": [
            "evening"
        ],
        "filter_mode": (
            "location+poi+daypart"
        ),
    }
    v2_result = {
        "prompt_intent": {
            "quality_intent": "high"
        }
    }

    merged = agent._merge_v2_intent_into_prompt_filter_report(
        prompt_filter_report, v2_result
    )
    assert merged["quality_intent"] == "high"

    privacy_cohorts = pd.DataFrame(
        {
            "cohort_id": ["c1"],
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
            "created_day_part": ["evening"],
            "quality_score": [0.85],
            "privacy_status": ["passed"],
            "final_match_score": [0.9],
            "match_type": ["exact_match"],
            "category_match_score": [1.0],
        }
    )
    v2_result["ranked_matches"] = privacy_cohorts.to_dict("records")

    selected_cohorts, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=privacy_cohorts,
        v2_result=v2_result,
        prompt_filter_report=merged,
        max_rows=10
    )
    assert report["quality_intent"] == "high"
    assert report["quality_policy_report"]["quality_intent"] == "high"
