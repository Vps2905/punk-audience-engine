import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_partial_fulfillment_retains_supported_locations():
    agent = AudienceIntelligenceOrchestratorAgent()

    cohorts = pd.DataFrame(
        {
            "cohort_id": ["c1", "c2"],
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["restaurant", "restaurant"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.9, 0.9],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
            "final_match_score": [0.95, 0.95],
            "match_type": ["exact_match", "exact_match"],
            "category_match_score": [1.0, 1.0],
        }
    )

    # Imagine only Montreal was returned by the engine v2 for some reason (maybe sf lacked privacy safe cohorts)
    # But let's actually just test _select_cohorts_from_v2_ranked

    # Let's say v2 found NO matches for SF. So `v2_result["ranked_matches"]` only has Montreal.
    ranked_records = [
        {
            "cohort_id": "c1",
            "location_name": "montreal",
            "primary_poi_type": "restaurant",
            "created_day_part": "evening",
            "quality_score": 0.9,
            "total_maid_volume": 10000,
            "privacy_status": "passed",
            "final_match_score": 0.95,
            "match_type": "exact_match",
            "category_match_score": 1.0,
        }
    ]

    selected, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=cohorts,
        v2_result={"ranked_matches": ranked_records},
        prompt_filter_report={
            "locations_detected": ["montreal", "san francisco"],
            "poi_terms_detected": ["restaurant"],
            "dayparts_detected": ["evening"],
        }
    )

    assert not selected.empty
    assert len(selected) == 1
    assert selected.iloc[0]["location_name"] == "montreal"
    assert report["fulfillment_status"] == "partial"
    assert "san francisco" in report["missing_requested_locations"]
    assert "montreal" in report["matched_requested_locations"]
    assert report["block_export"] is False
    assert any("san francisco" in w for w in report["coverage_warnings"])


def test_complete_fulfillment_retains_all():
    agent = AudienceIntelligenceOrchestratorAgent()

    cohorts = pd.DataFrame(
        {
            "cohort_id": ["c1", "c2"],
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["restaurant", "restaurant"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.9, 0.9],
            "total_maid_volume": [10000, 10000],
            "privacy_status": ["passed", "passed"],
            "final_match_score": [0.95, 0.95],
            "match_type": ["exact_match", "exact_match"],
            "category_match_score": [1.0, 1.0],
        }
    )

    ranked_records = cohorts.to_dict(orient="records")

    selected, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=cohorts,
        v2_result={"ranked_matches": ranked_records},
        prompt_filter_report={
            "locations_detected": ["montreal", "san francisco"],
            "poi_terms_detected": ["restaurant"],
            "dayparts_detected": ["evening"],
        }
    )

    assert len(selected) == 2
    assert report["fulfillment_status"] == "complete"
    assert len(report["missing_requested_locations"]) == 0
    assert report["block_export"] is False


def test_blocked_fulfillment_when_no_match():
    agent = AudienceIntelligenceOrchestratorAgent()
    cohorts = pd.DataFrame(
        {
            "cohort_id": ["c1"],
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
            "created_day_part": ["evening"],
        }
    )
    ranked_records = [
        {
            "cohort_id": "c1",
            "location_name": "montreal",
            "primary_poi_type": "cafe", # wrong category
            "created_day_part": "evening",
            "privacy_status": "passed",
            "final_match_score": 0.4, # Too low
            "match_type": "exact_match",
            "category_match_score": 0.4,
        }
    ]

    selected, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=cohorts,
        v2_result={"ranked_matches": ranked_records},
        prompt_filter_report={
            "locations_detected": ["montreal", "san francisco"],
            "poi_terms_detected": ["restaurant"],
            "dayparts_detected": ["evening"],
        }
    )

    assert selected.empty
    assert report["fulfillment_status"] == "blocked"
    assert report["block_export"] is True
    assert "strict_location_category_requested_but_no_safe_exact_match" == report["reason"]


def test_top_k_fairness_preserves_both_locations():
    agent = AudienceIntelligenceOrchestratorAgent()

    # Create 10 Montreal records and 1 SF record
    records = []
    for i in range(10):
        records.append({
            "cohort_id": f"m{i}",
            "location_name": "montreal",
            "primary_poi_type": "restaurant",
            "created_day_part": "evening",
            "quality_score": 0.9,
            "total_maid_volume": 10000,
            "privacy_status": "passed",
            "final_match_score": 0.95 - (i * 0.01), # High scores
            "match_type": "exact_match",
            "category_match_score": 1.0,
        })
    records.append({
        "cohort_id": "sf1",
        "location_name": "san francisco",
        "primary_poi_type": "restaurant",
        "created_day_part": "evening",
        "quality_score": 0.8,
        "total_maid_volume": 10000,
        "privacy_status": "passed",
        "final_match_score": 0.7, # Lower score
        "match_type": "exact_match",
        "category_match_score": 1.0,
    })

    cohorts = pd.DataFrame(records)

    selected, report = agent._select_cohorts_from_v2_ranked(
        privacy_cohorts=cohorts,
        v2_result={"ranked_matches": records},
        prompt_filter_report={
            "locations_detected": ["montreal", "san francisco"],
            "poi_terms_detected": ["restaurant"],
            "dayparts_detected": ["evening"],
        },
        max_rows=5  # We only want 5 total
    )

    assert len(selected) == 5
    locations = set(selected["location_name"].tolist())
    # Even though SF had the lowest score, it must be included because we requested both.
    assert "san francisco" in locations
    assert "montreal" in locations
