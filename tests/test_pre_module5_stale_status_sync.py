from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
)


def test_stale_status_propagates_to_nested_cohorts():
    result = {
        "approval_status": "pending_approval",
        "downstream_export_enabled": False,
        "package": {
            "cohorts": [
                {
                    "approval_status": "pending_approval",
                    "export_status": "pending_approval",
                    "allowed_destination": "approved_downstream_only",
                }
            ],
            "payload": {
                "approval_status": "pending_approval",
            },
            "approval_request": {
                "status": "pending_approval",
            },
        },
        "outputs": {
            "safe_export_manifest": (
                "postgres://audience_run_history.safe_export"
            )
        },
    }

    guarded = (
        AudienceIntelligenceOrchestratorAgent()
        ._apply_freshness_fail_closed(
            export_result=result,
            freshness_guardrail={
                "block_export": True,
                "block_export_reason": "stale",
            },
        )
    )

    cohort = guarded["package"]["cohorts"][0]

    assert guarded["approval_status"] == "blocked_stale_source"
    assert cohort["approval_status"] == "blocked_stale_source"
    assert cohort["export_status"] == "blocked_stale_source"
    assert cohort["allowed_destination"] == "none"
    assert guarded["package"]["payload"]["approval_status"] == (
        "blocked_stale_source"
    )
    assert guarded["package"]["approval_request"]["status"] == (
        "blocked_stale_source"
    )
