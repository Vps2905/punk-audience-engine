from pathlib import Path

from scripts.run_audience_intelligence_prompt import (
    build_business_summary,
    persist_business_summary,
)


def test_postgres_run_reference_is_not_written_as_local_path(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    result = {
        "run_id": "prompt_test_123",
        "run_dir": (
            "postgres://audience_run_history"
            "?run_id=prompt_test_123"
        ),
        "final_summary_path": (
            "postgres://audience_run_history.final_summary"
            "?run_id=prompt_test_123"
        ),
    }

    reference = persist_business_summary(
        result,
        "# Historical audience result",
    )

    assert reference == (
        "postgres://audience_run_history.final_summary"
        "?run_id=prompt_test_123"
        "&section=business_summary"
    )
    assert not (tmp_path / "postgres:").exists()


def test_existing_postgres_business_summary_reference_is_reused(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    expected = (
        "postgres://audience_run_history.business_summary"
        "?run_id=prompt_test_456"
    )

    result = {
        "run_id": "prompt_test_456",
        "run_dir": (
            "postgres://audience_run_history"
            "?run_id=prompt_test_456"
        ),
        "business_summary_path": expected,
    }

    reference = persist_business_summary(
        result,
        "# Existing result",
    )

    assert reference == expected
    assert not (tmp_path / "postgres:").exists()


def test_local_run_writes_business_summary(
    tmp_path,
):
    run_dir = tmp_path / "prompt_local_test"

    reference = persist_business_summary(
        {
            "run_id": "prompt_local_test",
            "run_dir": str(run_dir),
        },
        "# Local audience result",
    )

    path = Path(reference)

    assert path == run_dir / "business_prompt_summary.md"
    assert path.read_text(encoding="utf-8") == (
        "# Local audience result"
    )



def test_blocked_stale_source_summary_uses_prepared_candidate_language():
    run_reference = (
        "postgres://audience_run_history"
        "?run_id=prompt_historical_test"
    )

    result = {
        "prompt": (
            "Find high-quality evening restaurant "
            "audiences in Montreal."
        ),
        "run_id": "prompt_historical_test",
        "run_dir": run_reference,
        "source_mode": "postgres_safe_derived",
        "source_rows": 220,
        "prompt_selected_cohorts": 7,
        "prompt_filter_report": {
            "filter_mode": "location+poi+daypart",
            "locations": ["montreal"],
            "poi_terms": ["restaurant"],
            "dayparts": ["evening"],
        },
        "safe_export": {
            "status": "completed",
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
            "exported_cohorts": 1,
            "exported_lookalike_pairs": 0,
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
            },
        },
    }

    summary = build_business_summary(result)

    assert (
        "Prepared safe export candidates: 1"
        in summary
    )
    assert "Prepared lookalike pairs: 0" in summary
    assert "Exported audiences:" not in summary
    assert (
        "Privacy-safe export candidates were prepared, "
        "but downstream delivery is blocked. "
        "Approval status: blocked_stale_source."
        in summary
    )
    assert f"Run reference: {run_reference}" in summary
    assert "postgres:/audience_run_history" not in summary


def test_pending_approval_summary_does_not_claim_delivery():
    result = {
        "prompt": "Build an audience",
        "run_id": "prompt_pending_test",
        "run_dir": "/tmp/prompt_pending_test",
        "source_mode": "safe_artifact",
        "source_rows": 10,
        "prompt_selected_cohorts": 2,
        "prompt_filter_report": {},
        "safe_export": {
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "exported_cohorts": 2,
            "exported_lookalike_pairs": 1,
            "privacy_guarantees": {},
        },
    }

    summary = build_business_summary(result)

    assert (
        "Privacy-safe export candidates were prepared, "
        "but downstream delivery is blocked until "
        "explicit approval."
        in summary
    )
    assert (
        "downstream delivery is enabled"
        not in summary
    )
    assert (
        "Run folder: /tmp/prompt_pending_test"
        in summary
    )
