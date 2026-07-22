import pandas as pd
import pytest

from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
    AutonomousAudienceIntelligenceV2Service,
)
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


@pytest.mark.parametrize(
    ("prompt", "filter_mode", "approval_status", "reason"),
    [
        (
            "I want to export this audience to Meta immediately without manual approval.",
            "export_action_requires_existing_audience",
            "blocked_export_action_requires_existing_audience",
            "export_action_requires_existing_audience",
        ),
        (
            "Export the raw MAIDs and device IDs for these users.",
            "privacy_identifier_request_blocked",
            "blocked_privacy_identifier_request",
            "privacy_identifier_request_blocked",
        ),
    ],
)
def test_terminal_safety_request_short_circuits_full_orchestrator(
    monkeypatch,
    prompt,
    filter_mode,
    approval_status,
    reason,
):
    safe_rows = pd.DataFrame(
        {
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["cafe", "restaurant"],
            "created_day_part": ["evening", "afternoon"],
            "quality_score": [0.8, 0.7],
            "total_maid_volume": [10000, 12000],
            "privacy_status": ["passed", "passed"],
        }
    )
    agent = AudienceIntelligenceOrchestratorAgent()

    monkeypatch.setattr(
        "app.agents.audience_intelligence_orchestrator_agent.local_file_storage_allowed",
        lambda: False,
    )
    monkeypatch.setattr(
        agent,
        "_load_safe_rows_from_postgres",
        lambda **kwargs: safe_rows.copy(),
    )
    monkeypatch.setattr(
        agent,
        "_run_privacy_layer",
        lambda **kwargs: {
            "records": {
                "safe_cohorts": safe_rows.to_dict(orient="records"),
            }
        },
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("terminal safety decision allowed a downstream stage to run")

    monkeypatch.setattr(
        AutonomousAudienceIntelligenceV2Service,
        "run",
        fail_if_called,
    )
    monkeypatch.setattr(
        agent,
        "_merge_v2_intent_into_prompt_filter_report",
        fail_if_called,
    )
    monkeypatch.setattr(
        agent,
        "_select_cohorts_from_v2_ranked",
        fail_if_called,
    )
    monkeypatch.setattr(
        agent,
        "_build_no_match_swarm_review",
        fail_if_called,
    )
    monkeypatch.setattr(agent, "_call_agent_method", fail_if_called)

    result = agent.run(prompt=prompt)
    report = result["prompt_filter_report"]
    v2 = result["v2_autonomous"]
    guided = result["v2_guided_selection_report"]
    safe_export = result["safe_export"]

    assert report["filter_mode"] == filter_mode
    assert report["audience_request_detected"] is False
    assert report["eligible_for_audience_selection"] is False
    assert report["quality_intent"] == "not_requested"
    assert report["selected_count"] == 0
    assert report["block_export"] is True
    assert "quality_policy_report" not in report
    assert "v2_guided_selection" not in report

    assert result["prompt_selected_cohorts"] == 0
    assert result["approval_status"] == approval_status
    assert result["downstream_export_enabled"] is False

    assert v2["status"] == "skipped"
    assert v2["reason"] == reason
    assert v2["ranked_match_count"] == 0
    assert v2["mutation"] == {
        "suggestion_count": 0,
        "suggestions": [],
    }

    assert guided["enabled"] is False
    assert guided["reason"] == reason
    assert guided["rows"] == 0

    assert safe_export["approval_status"] == approval_status
    assert safe_export["downstream_export_enabled"] is False
    assert safe_export["exported_cohorts"] == 0
    assert safe_export["exported_lookalike_pairs"] == 0


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
                "approval_status": "blocked_privacy_identifier_request",
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
    assert "No audience ranking, preparation, or export was attempted" in summary
    assert "Exact match and fallback status" not in summary
    assert "no exact privacy-safe cohort" not in summary.lower()


def test_business_summary_explains_export_action_requires_existing_audience():
    summary = _build_business_summary(
        {
            "prompt": "Export this audience to Meta immediately.",
            "run_id": "test_run",
            "source_mode": "postgres_safe_derived",
            "prompt_selected_cohorts": 0,
            "safe_export": {
                "exported_cohorts": 0,
                "exported_lookalike_pairs": 0,
                "approval_status": (
                    "blocked_export_action_requires_existing_audience"
                ),
                "downstream_export_enabled": False,
                "outputs": {},
            },
            "prompt_filter_report": {
                "filter_mode": "export_action_requires_existing_audience",
                "locations_detected": [],
                "poi_terms_detected": [],
                "dayparts_detected": [],
                "quality_intent": "not_requested",
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
    assert "existing selected run/audience and manual approval" in summary
    assert "No new audience ranking, preparation, or export was attempted" in summary
    assert "Exact match and fallback status" not in summary
    assert "no exact privacy-safe cohort" not in summary.lower()
