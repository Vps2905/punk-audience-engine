import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_coffee_prompt_does_not_fallback_to_office_when_no_cafe_exists():
    df = pd.DataFrame(
        {
            "location_name": [
                "san francisco",
                "san francisco",
                "san francisco",
                "san francisco",
            ],
            "primary_poi_type": [
                "coworking_space",
                "corporate_office",
                "consultant",
                "real_estate_agency",
            ],
            "created_day_part": [
                "evening",
                "evening",
                "evening",
                "evening",
            ],
            "quality_score": [0.85, 0.7, 0.6, 0.5],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="I need people who grab espresso and snacks after work near San Francisco.",
        cohorts=df,
    )

    assert "cafe" in report["poi_terms_detected"]
    assert "coffee" in report["poi_terms_detected"]
    assert "evening" in report["dayparts_detected"]
    assert "san francisco" in report["locations_detected"]
    assert report["filter_mode"] == "location_category_gap_no_export"
    assert selected.empty


def test_coffee_prompt_does_not_fallback_to_montreal_cafe_when_sf_requested():
    df = pd.DataFrame(
        {
            "location_name": [
                "montreal downtown",
                "montreal",
                "san francisco",
            ],
            "primary_poi_type": [
                "cafe",
                "cafe",
                "coworking_space",
            ],
            "created_day_part": [
                "evening",
                "evening",
                "evening",
            ],
            "quality_score": [0.47, 0.38, 0.85],
            "total_maid_volume": [10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="I need people who grab espresso and snacks after work near San Francisco.",
        cohorts=df,
    )

    assert report["filter_mode"] == "location_category_gap_no_export"
    assert selected.empty


def test_strict_guardrail_blocks_location_category_gap_export():
    agent = AudienceIntelligenceOrchestratorAgent()

    selected = pd.DataFrame(
        columns=[
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "quality_score",
        ]
    )

    report = agent._strict_category_export_guardrail(
        prompt_filter_report={
            "filter_mode": "location_category_gap_no_export",
            "poi_terms_detected": ["cafe", "coffee"],
            "dayparts_detected": ["evening"],
            "locations_detected": ["san francisco"],
        },
        selected_cohorts=selected,
        v2_result={
            "coverage_warnings": [
                "san francisco was requested, but no exact safe cohort matched the requested category/daypart."
            ],
            "mutation": {"data_gap_count": 1},
        },
    )

    assert report["block_export"] is True
    assert report["reason"] == "strict_location_category_requested_but_no_safe_exact_match"


def test_strict_guardrail_blocks_selected_location_mismatch():
    agent = AudienceIntelligenceOrchestratorAgent()

    selected = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal downtown"],
            "primary_poi_type": ["cafe", "cafe"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.4, 0.5],
        }
    )

    report = agent._strict_category_export_guardrail(
        prompt_filter_report={
            "filter_mode": "poi+daypart",
            "poi_terms_detected": ["cafe", "coffee"],
            "dayparts_detected": ["evening"],
            "locations_detected": ["san francisco"],
        },
        selected_cohorts=selected,
        v2_result={
            "coverage_warnings": [],
            "mutation": {"data_gap_count": 1},
        },
    )

    assert report["block_export"] is True
    assert report["reason"] in {
        "strict_location_category_requested_but_selection_ignored_location",
        "strict_location_category_requested_but_selected_location_mismatch",
    }


def test_coworking_prompt_still_exports_true_category_matches():
    df = pd.DataFrame(
        {
            "location_name": [
                "san francisco",
                "san francisco",
                "montreal",
                "quebec",
            ],
            "primary_poi_type": [
                "coworking_space",
                "corporate_office",
                "gas_station",
                "casino",
            ],
            "created_day_part": [
                "evening",
                "evening",
                "evening",
                "evening",
            ],
            "quality_score": [0.8, 0.7, 0.9, 0.9],
            "total_maid_volume": [10000, 10000, 10000, 10000],
            "privacy_status": ["passed", "passed", "passed", "passed"],
        }
    )

    selected, report = AudienceIntelligenceOrchestratorAgent()._select_cohorts_for_prompt(
        prompt="Find the after-office crowd around coworking hubs and business centers.",
        cohorts=df,
    )

    guardrail = AudienceIntelligenceOrchestratorAgent()._strict_category_export_guardrail(
        prompt_filter_report=report,
        selected_cohorts=selected,
        v2_result={"mutation": {"data_gap_count": 0}, "coverage_warnings": []},
    )

    assert not selected.empty
    assert set(selected["primary_poi_type"]) == {"coworking_space", "corporate_office"}
    assert guardrail["block_export"] is False
