from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)


def service():
    return AudienceSupervisorDecisionService()


def test_stale_source_routes_to_blocked_refresh():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
        }
    )

    assert decision["route"] == "blocked"
    assert decision["stage"] == "freshness"
    assert decision["downstream_export_enabled"] is False
    assert "source_refresh_required" in decision["reason_codes"]


def test_broad_location_routes_to_clarification():
    decision = service().decide(
        {
            "status": "completed",
            "prompt_filter_report": {
                "filter_mode": "broad_location_no_export",
            },
            "downstream_export_enabled": False,
        }
    )

    assert decision["route"] == "needs_clarification"
    assert decision["awaiting_input"] is True
    assert decision["terminal"] is False


def test_no_exact_match_routes_to_coverage_block():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "blocked_no_safe_exact_match",
            "prompt_selected_cohorts": 0,
            "downstream_export_enabled": False,
        }
    )

    assert decision["route"] == "blocked"
    assert decision["stage"] == "coverage"
    assert "no_safe_exact_match" in decision["reason_codes"]


def test_pending_approval_route():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "approval_required": True,
            "downstream_export_enabled": False,
            "prompt_selected_cohorts": 2,
        }
    )

    assert decision["route"] == "pending_approval"
    assert decision["approval_required"] is True
    assert decision["awaiting_input"] is True


def test_approved_delivery_ready_route():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "approved",
            "approval_required": False,
            "downstream_export_enabled": True,
            "prompt_selected_cohorts": 2,
        }
    )

    assert decision["route"] == "delivery_ready"
    assert decision["stage"] == "delivery"
    assert decision["downstream_export_enabled"] is True


def test_downstream_without_approval_fails_closed():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": True,
            "prompt_selected_cohorts": 2,
        }
    )

    assert decision["route"] == "blocked"
    assert "inconsistent_downstream_state" in decision["reason_codes"]
    assert decision["downstream_export_enabled"] is False


def test_v2_failure_has_priority_over_pending_approval():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "v2_status": "failed",
            "downstream_export_enabled": False,
        }
    )

    assert decision["route"] == "blocked"
    assert decision["stage"] == "v2_intelligence"
    assert "v2_failure" in decision["reason_codes"]


def test_unknown_state_fails_closed():
    decision = service().decide({"status": "running"})

    assert decision["route"] == "blocked"
    assert decision["downstream_export_enabled"] is False
    assert "unknown_pipeline_state" in decision["reason_codes"]

def test_v2_autonomous_failure_uses_production_result_key():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "v2_autonomous": {
                "status": "failed",
            },
        }
    )

    assert decision["route"] == "blocked"
    assert decision["stage"] == "v2_intelligence"
    assert "v2_failure" in decision["reason_codes"]


def test_nested_v2_autonomous_freshness_fails_closed():
    decision = service().decide(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "v2_autonomous": {
                "status": "completed",
                "data_freshness": {
                    "freshness_status": "stale",
                },
            },
        }
    )

    assert decision["route"] == "blocked"
    assert decision["stage"] == "freshness"
    assert "source_refresh_required" in decision["reason_codes"]
