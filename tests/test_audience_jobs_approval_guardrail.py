import pandas as pd

from app.api.audience_intelligence_jobs import (
    _collect_job_approval_blockers,
    _safe_result_for_job,
)


def _safe_privacy():
    return {
        "raw_maids_exported": False,
        "hashed_identifiers_exported": False,
        "raw_observations_exported": False,
        "raw_lat_lng_exported": False,
        "raw_email_exported": False,
        "raw_phone_exported": False,
        "individual_user_data_exported": False,
    }


def test_async_job_approval_blocks_stale_source_and_swarm_block():
    result = {
        "safe_export": {"approval_status": "pending_approval"},
        "privacy_guarantees": _safe_privacy(),
        "v2_autonomous": {
            "data_freshness": {"freshness_status": "stale"},
        },
        "v2_swarm_review": {
            "overall_review_status": "blocked",
        },
    }

    blockers = _collect_job_approval_blockers(
        result=result,
        manifest={},
        payload={},
        approval={},
        cohorts=pd.DataFrame({"audience_name": ["safe"], "export_status": ["pending_approval"]}),
    )

    assert any("stale" in blocker.lower() for blocker in blockers)
    assert any("swarm" in blocker.lower() for blocker in blockers)


def test_async_job_approval_blocks_prompt_filter_bypass_and_empty_cohorts():
    result = {
        "safe_export": {"approval_status": "pending_approval"},
        "privacy_guarantees": _safe_privacy(),
        "prompt_filter_report": {
            "filter_mode": "export_action_requires_existing_audience",
        },
    }

    blockers = _collect_job_approval_blockers(
        result=result,
        manifest={},
        payload={},
        approval={},
        cohorts=pd.DataFrame(),
    )

    assert any("no exportable cohorts" in blocker.lower() for blocker in blockers)
    assert any("prompt filter mode" in blocker.lower() for blocker in blockers)


def test_async_job_approval_blocks_privacy_leak_flag():
    result = {
        "safe_export": {"approval_status": "pending_approval"},
        "privacy_guarantees": {
            **_safe_privacy(),
            "raw_maids_exported": True,
        },
    }

    blockers = _collect_job_approval_blockers(
        result=result,
        manifest={},
        payload={},
        approval={},
        cohorts=pd.DataFrame({"audience_name": ["unsafe"]}),
    )

    assert any("raw_maids_exported" in blocker for blocker in blockers)


def test_async_job_approval_allows_clean_pending_package():
    result = {
        "safe_export": {"approval_status": "pending_approval"},
        "privacy_guarantees": _safe_privacy(),
        "v2_autonomous": {
            "data_freshness": {"freshness_status": "fresh"},
        },
        "v2_swarm_review": {
            "overall_review_status": "needs_human_review",
        },
        "prompt_filter_report": {
            "filter_mode": "location+poi+daypart",
        },
    }

    blockers = _collect_job_approval_blockers(
        result=result,
        manifest={},
        payload={},
        approval={},
        cohorts=pd.DataFrame({"audience_name": ["safe"], "export_status": ["pending_approval"]}),
    )

    assert blockers == []


def test_safe_result_for_job_preserves_approval_safety_metadata():
    raw = {
        "status": "completed",
        "run_id": "run_1",
        "prompt": "coffee shop in Montreal",
        "source_mode": "postgres_safe_derived",
        "source_rows": 220,
        "prompt_selected_cohorts": 2,
        "coverage_warnings": [],
        "prompt_filter_report": {"filter_mode": "location+poi+daypart"},
        "v2_autonomous": {"data_freshness": {"freshness_status": "stale"}},
        "v2_swarm_review": {"overall_review_status": "blocked"},
        "business_summary": "summary",
        "business_summary_path": "summary.md",
        "final_summary_path": "final.json",
        "run_dir": "data/prompt_runs/run_1",
        "safe_export": {"approval_status": "pending_approval"},
        "privacy_guarantees": _safe_privacy(),
    }

    safe = _safe_result_for_job(raw)

    assert safe["prompt_filter_report"]["filter_mode"] == "location+poi+daypart"
    assert safe["v2_autonomous"]["data_freshness"]["freshness_status"] == "stale"
    assert safe["v2_swarm_review"]["overall_review_status"] == "blocked"
