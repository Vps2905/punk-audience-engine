import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent
from app.api.audience_intelligence_prompt import _build_business_summary


def test_gym_prompt_blocks_unrelated_montreal_fallback_audiences():
    df = pd.DataFrame(
        {
            "location_name": [
                "montreal downtown",
                "montreal qc",
                "montreal",
                "montreal",
            ],
            "primary_poi_type": [
                "grocery_store",
                "barber_shop",
                "shawarma_restaurant",
                "hair_salon",
            ],
            "created_day_part": [
                "evening",
                "evening",
                "evening",
                "evening",
            ],
            "quality_score": [0.55, 0.54, 0.53, 0.29],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="I run a premium gym in Montreal and want people who visit fitness centers after work.",
        cohorts=df,
    )

    assert selected.empty
    assert report["filter_mode"] == "location_category_gap_no_export"
    assert report["block_export"] is True
    assert "gym" in report["poi_terms_detected"]
    assert "fitness" in report["poi_terms_detected"]


def test_gym_prompt_allows_true_gym_evening_match():
    df = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal"],
            "primary_poi_type": ["gym", "barber_shop"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.8, 0.9],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="I run a premium gym in Montreal and want people who visit fitness centers after work.",
        cohorts=df,
    )

    assert not selected.empty
    assert selected.iloc[0]["primary_poi_type"] == "gym"
    assert report["filter_mode"] == "location+poi+daypart"


def test_raw_maid_request_blocks_with_privacy_reason():
    df = pd.DataFrame(
        {
            "location_name": ["montreal"],
            "primary_poi_type": ["cafe"],
            "created_day_part": ["evening"],
            "quality_score": [0.8],
            "total_maid_volume": [10000],
            "privacy_status": ["passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Can you give me the raw MAIDs or individual device IDs of people who visited coffee shops in Montreal?",
        cohorts=df,
    )

    assert selected.empty
    assert report["filter_mode"] == "privacy_identifier_request_blocked"
    assert report["block_export"] is True
    assert report["reason"] == "raw_identifier_request_blocked"


def test_export_action_only_prompt_does_not_generate_random_audiences():
    df = pd.DataFrame(
        {
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["shawarma_restaurant", "coworking_space"],
            "created_day_part": ["afternoon", "evening"],
            "quality_score": [0.9, 0.85],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="I want to export this audience to Meta immediately without manual approval.",
        cohorts=df,
    )

    assert selected.empty
    assert report["filter_mode"] == "export_action_requires_existing_audience"
    assert report["block_export"] is True
    assert report["reason"] == "export_requires_existing_approved_audience"


def test_business_summary_explains_raw_identifier_refusal():
    summary = _build_business_summary(
        {
            "prompt": "Can you give me raw MAIDs?",
            "run_id": "test_run",
            "source_mode": "postgres_safe_derived",
            "prompt_selected_cohorts": 0,
            "safe_export": {
                "exported_cohorts": 0,
                "exported_lookalike_pairs": 0,
                "approval_status": "blocked_no_safe_exact_match",
                "downstream_export_enabled": False,
                "outputs": {},
            },
            "prompt_filter_report": {
                "filter_mode": "privacy_identifier_request_blocked",
                "locations_detected": [],
                "poi_terms_detected": [],
                "dayparts_detected": [],
            },
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
            },
            "run_dir": "data/prompt_runs/test_run",
        }
    )

    assert "Safety decision" in summary
    assert "cannot be provided or exported" in summary
    assert "Only privacy-safe aggregated cohorts are allowed" in summary
