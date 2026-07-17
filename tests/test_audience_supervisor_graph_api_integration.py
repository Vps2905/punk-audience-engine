from __future__ import annotations

from pathlib import Path

from app.agents.audience_supervisor_agent import (
    AudienceSupervisorAgent,
    autonomous_supervisor_graph_enabled,
)
from app.agents.audience_supervisor_graph import (
    AudienceSupervisorGraph,
)
from app.api import audience_intelligence_jobs as jobs_api
from app.api import audience_intelligence_prompt as prompt_api


class FakeOrchestrator:
    def run(self, **kwargs):
        return {
            "status": "completed",
            "run_id": "run_graph_feature_flag_test",
            "prompt": kwargs.get("prompt"),
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "safe_export": {
                "approval_status": "pending_approval",
                "downstream_export_enabled": False,
            },
        }


def set_flags(
    monkeypatch,
    *,
    supervisor: str,
    graph: str,
):
    monkeypatch.setenv(
        "ENABLE_AUTONOMOUS_SUPERVISOR",
        supervisor,
    )
    monkeypatch.setenv(
        "ENABLE_AUTONOMOUS_SUPERVISOR_GRAPH",
        graph,
    )


def patch_api_factories(monkeypatch):
    monkeypatch.setattr(
        prompt_api,
        "AudienceIntelligenceOrchestratorAgent",
        FakeOrchestrator,
    )
    monkeypatch.setattr(
        jobs_api,
        "AudienceIntelligenceOrchestratorAgent",
        FakeOrchestrator,
    )


def test_graph_flag_parser_is_explicit_and_safe():
    assert autonomous_supervisor_graph_enabled("true") is True
    assert autonomous_supervisor_graph_enabled("1") is True
    assert autonomous_supervisor_graph_enabled("enabled") is True
    assert autonomous_supervisor_graph_enabled("false") is False
    assert autonomous_supervisor_graph_enabled("0") is False
    assert autonomous_supervisor_graph_enabled("unexpected") is False


def test_graph_flag_cannot_bypass_disabled_parent_gate(
    monkeypatch,
):
    set_flags(
        monkeypatch,
        supervisor="false",
        graph="true",
    )
    patch_api_factories(monkeypatch)

    assert isinstance(
        prompt_api._audience_execution_agent(),
        FakeOrchestrator,
    )
    assert isinstance(
        jobs_api._audience_execution_agent(),
        FakeOrchestrator,
    )


def test_supervisor_wrapper_selected_when_graph_disabled(
    monkeypatch,
):
    set_flags(
        monkeypatch,
        supervisor="true",
        graph="false",
    )
    patch_api_factories(monkeypatch)

    assert isinstance(
        prompt_api._audience_execution_agent(),
        AudienceSupervisorAgent,
    )
    assert isinstance(
        jobs_api._audience_execution_agent(),
        AudienceSupervisorAgent,
    )


def test_graph_selected_for_both_api_paths(
    monkeypatch,
):
    set_flags(
        monkeypatch,
        supervisor="true",
        graph="true",
    )
    patch_api_factories(monkeypatch)

    assert isinstance(
        prompt_api._audience_execution_agent(),
        AudienceSupervisorGraph,
    )
    assert isinstance(
        jobs_api._audience_execution_agent(),
        AudienceSupervisorGraph,
    )


def test_unknown_graph_flag_preserves_supervisor_wrapper(
    monkeypatch,
):
    set_flags(
        monkeypatch,
        supervisor="true",
        graph="unexpected",
    )
    patch_api_factories(monkeypatch)

    assert isinstance(
        prompt_api._audience_execution_agent(),
        AudienceSupervisorAgent,
    )


def test_graph_run_matches_api_agent_contract():
    graph = AudienceSupervisorGraph(
        orchestrator_factory=FakeOrchestrator,
    )

    result = graph.run(
        prompt="Find cafe visitors in Montreal"
    )

    assert result["supervisor_route"] == "pending_approval"
    assert result["graph_terminal_status"] == (
        "pending_approval"
    )
    assert result["supervisor_reason_codes"]
    assert result["supervisor_trace"] == (
        result["supervisor_graph_trace"]
    )


def test_prompt_response_exposes_graph_metadata():
    graph = AudienceSupervisorGraph(
        orchestrator_factory=FakeOrchestrator,
    )
    result = graph.run(
        prompt="Find cafe visitors in Montreal"
    )

    response = prompt_api._build_prompt_api_response(
        result=result,
        business_summary="summary",
        business_summary_path=Path("/tmp/summary.md"),
    )

    assert response["supervisor_route"] == (
        "pending_approval"
    )
    assert response["graph_terminal_status"] == (
        "pending_approval"
    )
    assert response["supervisor_graph_trace"][-1] == {
        "event": "graph_terminated",
        "terminal_status": "pending_approval",
    }
