from pathlib import Path

from app.api.audience_intelligence_prompt import _build_prompt_api_response


def test_prompt_api_response_surfaces_stale_fail_closed_fields(tmp_path):
    response = _build_prompt_api_response(
        result={
            "status": "completed",
            "run_id": "run_1",
            "source_mode": "postgres_safe_derived",
            "source_rows": 220,
            "privacy_cohorts": 94,
            "final_summary_path": str(tmp_path / "final.json"),
            "run_dir": str(tmp_path),
            "prompt_selected_cohorts": 2,
            "prompt_filter_report": {},
            "coverage_warnings": [],
            "v2_autonomous": {
                "data_freshness": {
                    "freshness_status": "stale",
                    "source_rows_checked": 220,
                    "reason": "Latest source timestamp is older than 48 hours.",
                },
                "embedding_manifest": {
                    "embedding_store": "postgres",
                    "vector_backend": "postgres_array",
                },
            },
            "safe_export": {
                "approval_status": "blocked_stale_source",
                "downstream_export_enabled": False,
                "exported_cohorts": 2,
                "exported_lookalike_pairs": 2,
                "outputs": {},
            },
            "privacy_guarantees": {},
            "run_history": {},
        },
        business_summary="summary",
        business_summary_path=Path(tmp_path / "summary.md"),
    )

    assert response["freshness_status"] == "stale"
    assert response["approval_status"] == "blocked_stale_source"
    assert response["downstream_export_enabled"] is False
    assert response["block_export"] is True
    assert "48 hours" in response["block_export_reason"]

    assert response["safe_export"]["approval_status"] == "blocked_stale_source"
    assert response["safe_export"]["block_export"] is True
    assert response["safe_export"]["export_blocked_until_source_refresh"] is True
    assert response["safe_export"]["downstream_export_enabled"] is False
