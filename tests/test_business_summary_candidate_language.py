from app.api.audience_intelligence_prompt import (
    _build_business_summary,
)


def test_blocked_audience_is_described_as_candidate():
    summary = _build_business_summary(
        {
            "prompt": "Cafe visitors in Montreal",
            "run_id": "run_1",
            "source_mode": "postgres_safe_derived",
            "source_rows": 220,
            "prompt_selected_cohorts": 1,
            "prompt_filter_report": {
                "filter_mode": (
                    "location+poi+daypart"
                ),
                "locations_detected": [
                    "montreal"
                ],
                "poi_terms_detected": [
                    "cafe"
                ],
                "dayparts_detected": [
                    "evening"
                ],
            },
            "safe_export": {
                "exported_cohorts": 1,
                "exported_lookalike_pairs": 0,
                "approval_status": (
                    "blocked_stale_source"
                ),
                "downstream_export_enabled": False,
                "outputs": {},
            },
            "coverage_warnings": [],
            "privacy_guarantees": {},
        }
    )

    assert (
        "Prepared audience candidates: 1"
        in summary
    )
    assert "Exported audiences:" not in summary
    assert "Delivered audiences:" not in summary


def test_delivered_audience_uses_delivery_language():
    summary = _build_business_summary(
        {
            "prompt": "Approved audience",
            "run_id": "run_2",
            "source_mode": "postgres_safe_derived",
            "source_rows": 1,
            "prompt_selected_cohorts": 1,
            "prompt_filter_report": {},
            "safe_export": {
                "exported_cohorts": 1,
                "exported_lookalike_pairs": 0,
                "approval_status": "approved",
                "downstream_export_enabled": True,
                "outputs": {},
            },
            "coverage_warnings": [],
            "privacy_guarantees": {},
        }
    )

    assert "Delivered audiences: 1" in summary

def test_no_match_summary_does_not_claim_reviewable_package():
    summary = _build_business_summary(
        {
            "prompt": "Cafe visitors near Qubic",
            "run_id": "run_no_match",
            "source_mode": "postgres_safe_derived",
            "source_rows": 220,
            "prompt_selected_cohorts": 0,
            "prompt_filter_report": {
                "filter_mode": "location+poi+daypart",
                "locations_detected": ["qubic"],
                "poi_terms_detected": ["cafe"],
                "dayparts_detected": ["evening"],
            },
            "safe_export": {
                "exported_cohorts": 0,
                "exported_lookalike_pairs": 0,
                "approval_status": (
                    "blocked_no_safe_exact_match"
                ),
                "downstream_export_enabled": False,
                "outputs": {},
            },
            "coverage_warnings": [
                (
                    "requested location/category/daypart had no "
                    "exact safe cohort; export blocked instead of "
                    "falling back."
                )
            ],
            "v2_autonomous": {
                "status": "completed",
                "data_freshness": {
                    "freshness_status": "stale",
                },
            },
            "v2_swarm_review": {
                "overall_review_status": "blocked",
                "coverage_warning_count": 1,
                "data_gap_count": 0,
            },
            "privacy_guarantees": {},
        }
    )

    assert "Prepared audience candidates: 0" in summary
    assert (
        "No audience candidate was created for this run."
        in summary
    )
    assert (
        "No audience candidate was created because no exact "
        "privacy-safe cohort matched"
        in summary
    )
    assert "Export package is reviewable" not in summary
    assert (
        "Approval-gated candidates are stored in Postgres"
        not in summary
    )
