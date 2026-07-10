from app.services.audience_run_history_service import AudienceRunHistoryService


def test_approval_blockers_block_stale_and_swarm_blocked_run():
    service = AudienceRunHistoryService()

    final_summary = {
        "safe_export": {
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "export_blocked_until_approved": True,
        },
        "v2_autonomous": {
            "data_freshness": {
                "freshness_status": "stale",
                "stale_data_warning": True,
            }
        },
        "v2_swarm_review": {
            "overall_review_status": "blocked",
            "freshness_status": "stale",
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

    blockers = service._approval_blockers(final_summary)

    assert "Swarm review status is blocked." in blockers
    assert "Source freshness is stale; refresh or verify Echo/Postgres source before approval." in blockers
    assert "Freshness agent raised stale_data_warning." in blockers


def test_approval_blockers_block_privacy_leak_flags():
    service = AudienceRunHistoryService()

    final_summary = {
        "safe_export": {
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "export_blocked_until_approved": True,
        },
        "v2_autonomous": {
            "data_freshness": {
                "freshness_status": "fresh",
                "stale_data_warning": False,
            }
        },
        "v2_swarm_review": {
            "overall_review_status": "needs_human_review",
            "freshness_status": "fresh",
        },
        "privacy_guarantees": {
            "raw_maids_exported": True,
            "hashed_identifiers_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
        },
    }

    blockers = service._approval_blockers(final_summary)

    assert any("Privacy guarantee failed" in blocker for blocker in blockers)
    assert any("raw_maids_exported" in blocker for blocker in blockers)


def test_approval_blockers_allow_clean_review_ready_run():
    service = AudienceRunHistoryService()

    final_summary = {
        "safe_export": {
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "export_blocked_until_approved": True,
        },
        "v2_autonomous": {
            "data_freshness": {
                "freshness_status": "fresh",
                "stale_data_warning": False,
            }
        },
        "v2_swarm_review": {
            "overall_review_status": "needs_human_review",
            "freshness_status": "fresh",
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

    blockers = service._approval_blockers(final_summary)

    assert blockers == []
