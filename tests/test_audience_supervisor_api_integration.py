from __future__ import annotations

from pathlib import Path

from app.agents.audience_supervisor_agent import (
    AudienceSupervisorAgent,
    autonomous_supervisor_enabled,
)
from app.api import audience_intelligence_jobs as jobs_api
from app.api import audience_intelligence_prompt as prompt_api


class FakeOrchestrator:
    def run(self, **kwargs):
        return {
            "status": "completed",
            "run_id": "run_feature_flag_test",
            "prompt": kwargs.get("prompt"),
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "safe_export": {
                "approval_status": "pending_approval",
                "downstream_export_enabled": False,
            },
        }


def test_supervisor_flag_parser_is_explicit_and_safe():
    assert autonomous_supervisor_enabled("true") is True
    assert autonomous_supervisor_enabled("1") is True
    assert autonomous_supervisor_enabled("enabled") is True
    assert autonomous_supervisor_enabled("false") is False
    assert autonomous_supervisor_enabled("0") is False
    assert autonomous_supervisor_enabled("unexpected") is False


def test_prompt_api_uses_existing_orchestrator_when_disabled(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_AUTONOMOUS_SUPERVISOR",
        "false",
    )
    monkeypatch.setattr(
        prompt_api,
        "AudienceIntelligenceOrchestratorAgent",
        FakeOrchestrator,
    )

    agent = prompt_api._audience_execution_agent()

    assert isinstance(agent, FakeOrchestrator)


def test_prompt_api_uses_supervisor_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_AUTONOMOUS_SUPERVISOR",
        "true",
    )
    monkeypatch.setattr(
        prompt_api,
        "AudienceIntelligenceOrchestratorAgent",
        FakeOrchestrator,
    )

    agent = prompt_api._audience_execution_agent()
    result = agent.run(
        prompt="Find cafe visitors in Montreal"
    )

    assert isinstance(agent, AudienceSupervisorAgent)
    assert result["supervisor_route"] == "pending_approval"
    assert result["supervisor_stage"] == "approval"
    assert result["supervisor_decision"]["route"] == (
        "pending_approval"
    )


def test_jobs_api_uses_existing_orchestrator_when_disabled(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_AUTONOMOUS_SUPERVISOR",
        "false",
    )
    monkeypatch.setattr(
        jobs_api,
        "AudienceIntelligenceOrchestratorAgent",
        FakeOrchestrator,
    )

    agent = jobs_api._audience_execution_agent()

    assert isinstance(agent, FakeOrchestrator)


def test_jobs_api_uses_supervisor_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv(
        "ENABLE_AUTONOMOUS_SUPERVISOR",
        "true",
    )
    monkeypatch.setattr(
        jobs_api,
        "AudienceIntelligenceOrchestratorAgent",
        FakeOrchestrator,
    )

    agent = jobs_api._audience_execution_agent()
    result = agent.run(
        prompt="Find cafe visitors in Montreal"
    )

    assert isinstance(agent, AudienceSupervisorAgent)
    assert result["supervisor_route"] == "pending_approval"
    assert result["supervisor_stage"] == "approval"


def test_prompt_response_contract_is_unchanged_without_supervisor():
    result = {
        "status": "completed",
        "run_id": "run_without_supervisor",
        "safe_export": {},
        "v2_autonomous": {},
    }

    response = prompt_api._build_prompt_api_response(
        result=result,
        business_summary="summary",
        business_summary_path=Path("/tmp/summary.md"),
    )

    assert "supervisor_decision" not in response
    assert "supervisor_trace" not in response


def test_prompt_response_exposes_supervisor_metadata_when_enabled():
    result = {
        "status": "completed",
        "run_id": "run_with_supervisor",
        "safe_export": {},
        "v2_autonomous": {},
        "supervisor_decision": {
            "route": "blocked",
            "stage": "freshness",
        },
        "supervisor_route": "blocked",
        "supervisor_stage": "freshness",
        "supervisor_reason_codes": [
            "source_refresh_required",
        ],
        "supervisor_trace": [
            {
                "event": "supervisor_decision",
                "route": "blocked",
            }
        ],
    }

    response = prompt_api._build_prompt_api_response(
        result=result,
        business_summary="summary",
        business_summary_path=Path("/tmp/summary.md"),
    )

    assert response["supervisor_route"] == "blocked"
    assert response["supervisor_stage"] == "freshness"
    assert response["supervisor_reason_codes"] == [
        "source_refresh_required"
    ]
    assert response["supervisor_trace"][0]["event"] == (
        "supervisor_decision"
    )
