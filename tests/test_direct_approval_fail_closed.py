import pandas as pd

from app.api.audience_intelligence_jobs import _collect_job_approval_blockers
from app.services.audience_run_history_service import AudienceRunHistoryService


def test_run_history_blocks_new_stale_source_fields():
    service = AudienceRunHistoryService()

    blockers = service._approval_blockers(
        {
            "status": "completed",
            "freshness_status": "stale",
            "approval_status": "blocked_stale_source",
            "block_export": True,
            "downstream_export_enabled": False,
            "source_freshness": {
                "freshness_status": "stale",
                "block_export": True,
            },
            "safe_export": {
                "approval_status": "blocked_stale_source",
                "downstream_export_enabled": False,
                "block_export": True,
                "export_blocked_until_source_refresh": True,
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
        }
    )

    joined = " ".join(blockers).lower()

    assert blockers
    assert "blocked_stale_source" in joined
    assert "freshness is stale" in joined
    assert "explicitly blocked" in joined


def test_run_history_blocks_no_exact_match_status():
    service = AudienceRunHistoryService()

    blockers = service._approval_blockers(
        {
            "status": "completed",
            "approval_status": "blocked_no_safe_exact_match",
            "safe_export": {
                "approval_status": "blocked_no_safe_exact_match",
                "downstream_export_enabled": False,
            },
            "prompt_filter_report": {
                "filter_mode": "location_category_gap_no_export",
            },
            "privacy_guarantees": {},
        }
    )

    joined = " ".join(blockers).lower()

    assert "blocked_no_safe_exact_match" in joined
    assert "location_category_gap_no_export" in joined


def test_run_history_blocks_repeat_approval():
    service = AudienceRunHistoryService()

    blockers = service._approval_blockers(
        {
            "approval_status": "approved",
            "safe_export": {
                "approval_status": "approved",
                "downstream_export_enabled": True,
            },
            "privacy_guarantees": {},
        }
    )

    assert any("repeat approval is not allowed" in item for item in blockers)


def test_clean_pending_package_has_no_run_history_blockers():
    service = AudienceRunHistoryService()

    blockers = service._approval_blockers(
        {
            "status": "completed",
            "freshness_status": "fresh",
            "approval_status": "pending_approval",
            "block_export": False,
            "safe_export": {
                "approval_status": "pending_approval",
                "downstream_export_enabled": False,
                "block_export": False,
            },
            "v2_swarm_review": {
                "overall_review_status": "ready_for_human_approval",
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
        }
    )

    assert blockers == []


def test_async_direct_approval_blocks_blocked_stale_status():
    result = {
        "status": "completed",
        "freshness_status": "stale",
        "approval_status": "blocked_stale_source",
        "block_export": True,
        "safe_export": {
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
            "block_export": True,
        },
        "privacy_guarantees": {},
    }

    blockers = _collect_job_approval_blockers(
        result=result,
        manifest={"approval_status": "blocked_stale_source"},
        payload={"approval_status": "blocked_stale_source"},
        approval={"approval_status": "blocked_stale_source"},
        cohorts=pd.DataFrame(
            {
                "audience_name": ["safe cohort"],
                "export_status": ["pending_approval"],
            }
        ),
    )

    assert blockers
    joined = " ".join(blockers).lower()
    assert "blocked_stale_source" in joined or "source data is stale" in joined
