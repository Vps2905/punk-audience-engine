from app.services.audience_run_history_service import AudienceRunHistoryService


def test_privacy_budget_request_uses_final_summary_defaults():
    service = AudienceRunHistoryService()

    run = {
        "locations_detected": ["montreal"],
        "poi_terms_detected": ["cafe"],
        "dayparts_detected": ["evening"],
        "final_summary": {
            "safe_export": {},
            "prompt_filter_report": {
                "locations_detected": ["montreal"],
                "poi_terms_detected": ["cafe", "coffee"],
                "dayparts_detected": ["evening"],
            },
        },
    }

    request = service._privacy_budget_request_for_run(
        tenant_id="tenant-a",
        run_id="run_1",
        run=run,
        actor="reviewer",
        note="approval",
    )

    assert request.run_id == "run_1"
    assert request.tenant_id == "tenant-a"
    assert request.epsilon == 1.0
    assert request.delta == 1e-5
    assert request.sensitivity == 1.0
    assert request.max_budget == 5.0
    assert request.mechanism == "gaussian"
    assert request.query_type == "audience_export_approval"
    assert request.actor == "reviewer"
    assert request.budget_scope == "audience_export|montreal|cafe,coffee|evening"


def test_privacy_budget_request_allows_final_summary_override():
    service = AudienceRunHistoryService()

    run = {
        "final_summary": {
            "privacy_budget": {
                "budget_scope": "customer_123",
                "epsilon": 0.5,
                "delta": 1e-6,
                "sensitivity": 2.0,
                "max_budget": 3.0,
                "mechanism": "gaussian",
            }
        }
    }

    request = service._privacy_budget_request_for_run(
        tenant_id="tenant-a",
        run_id="run_2",
        run=run,
        actor="reviewer",
        note="approval",
    )

    assert request.budget_scope == "customer_123"
    assert request.epsilon == 0.5
    assert request.delta == 1e-6
    assert request.sensitivity == 2.0
    assert request.max_budget == 3.0
