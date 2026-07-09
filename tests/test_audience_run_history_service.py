import pandas as pd

from app.services.audience_run_history_service import AudienceRunHistoryService


def test_history_service_skips_without_db_url(monkeypatch):
    for key in [
        "AUDIENCE_HISTORY_DATABASE_URL",
        "ECHO_DATABASE_URL",
        "DATABASE_URL",
        "POSTGRES_URL",
        "POSTGRES_DATABASE_URL",
        "SUPABASE_DB_URL",
        "DB_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    report = AudienceRunHistoryService().persist_run(
        final_summary={
            "run_id": "run_test",
            "prompt": "restaurant audience",
            "status": "completed",
        },
        selected_cohorts=[],
    )

    assert report["status"] == "skipped"
    assert report["reason"] == "no_database_url"


def test_build_cohort_rows_maps_sensitive_risk():
    service = AudienceRunHistoryService()

    final_summary = {
        "run_id": "run_1",
        "sensitive_poi_privacy_risk": {
            "assessed_audiences": [
                {
                    "audience_name": "casino - evening - quebec",
                    "location_name": "quebec",
                    "primary_poi_type": "casino",
                    "created_day_part": "evening",
                    "decision": "review_required",
                    "risk_level": "regulated_or_sensitive_review",
                }
            ]
        },
    }

    selected = pd.DataFrame(
        [
            {
                "location_name": "quebec",
                "primary_poi_type": "casino",
                "created_day_part": "evening",
                "quality_score": 0.42,
            }
        ]
    )

    rows = service._build_cohort_rows(
        run_id="run_1",
        final_summary=final_summary,
        selected_cohorts=selected,
    )

    assert len(rows) == 1
    assert rows[0]["risk_decision"] == "review_required"
    assert rows[0]["risk_level"] == "regulated_or_sensitive_review"
    assert rows[0]["quality_score"] == 0.42


def test_build_artifact_rows_extracts_paths():
    service = AudienceRunHistoryService()

    rows = service._build_artifact_rows(
        run_id="run_1",
        final_summary={
            "final_summary_path": "data/prompt_runs/run_1/final_prompt_summary.json",
            "safe_export": {
                "outputs": {
                    "safe_export_manifest": "data/prompt_runs/run_1/05_safe_export/safe_export_manifest.json"
                }
            },
        },
    )

    paths = {row["artifact_path"] for row in rows}

    assert "data/prompt_runs/run_1/final_prompt_summary.json" in paths
    assert "data/prompt_runs/run_1/05_safe_export/safe_export_manifest.json" in paths
