from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
)


def test_v2_failure_blocks_export_and_nested_package():
    export_result = {
        "status": "completed",
        "approval_status": "pending_approval",
        "downstream_export_enabled": False,
        "exported_cohorts": 1,
        "exported_lookalike_pairs": 1,
        "package": {
            "cohorts": [
                {
                    "approval_status": "pending_approval",
                    "export_status": "pending_approval",
                    "allowed_destination": "approved_downstream_only",
                    "downstream_export_enabled": False,
                }
            ],
            "lookalikes": [
                {
                    "approval_status": "pending_approval",
                    "export_status": "pending_approval",
                }
            ],
            "payload": {
                "approval_status": "pending_approval",
                "exported_cohorts": 1,
            },
            "approval_request": {
                "status": "pending_approval",
            },
        },
    }

    result = (
        AudienceIntelligenceOrchestratorAgent()
        ._apply_v2_failure_fail_closed(
            export_result=export_result,
            v2_result={
                "status": "failed",
                "error": (
                    "server closed the connection unexpectedly"
                ),
            },
        )
    )

    assert result["status"] == "blocked"
    assert result["approval_status"] == "blocked_v2_failure"
    assert result["downstream_export_enabled"] is False
    assert result["block_export"] is True
    assert result["exported_cohorts"] == 0
    assert result["exported_lookalike_pairs"] == 0

    cohort = result["package"]["cohorts"][0]
    assert cohort["approval_status"] == "blocked_v2_failure"
    assert cohort["export_status"] == "blocked_v2_failure"
    assert cohort["allowed_destination"] == "none"
    assert cohort["downstream_export_enabled"] is False

    lookalike = result["package"]["lookalikes"][0]
    assert lookalike["approval_status"] == "blocked_v2_failure"
    assert lookalike["export_status"] == "blocked_v2_failure"

    assert (
        result["package"]["payload"]["approval_status"]
        == "blocked_v2_failure"
    )
    assert result["package"]["payload"]["exported_cohorts"] == 0
    assert (
        result["package"]["approval_request"]["status"]
        == "blocked_v2_failure"
    )


def test_successful_v2_leaves_export_result_unchanged():
    export_result = {
        "status": "completed",
        "approval_status": "pending_approval",
        "exported_cohorts": 1,
    }

    result = (
        AudienceIntelligenceOrchestratorAgent()
        ._apply_v2_failure_fail_closed(
            export_result=export_result,
            v2_result={"status": "completed"},
        )
    )

    assert result == export_result
