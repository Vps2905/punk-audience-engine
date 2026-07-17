from __future__ import annotations

import pytest

from app.agents.audience_supervisor_graph import (
    AudienceSupervisorGraph,
)


class FakeOrchestrator:
    def __init__(self, result):
        self.result = result

    def run(self, **kwargs):
        output = dict(self.result)
        output.setdefault("prompt", kwargs.get("prompt"))
        return output


def build_graph(result):
    return AudienceSupervisorGraph(
        orchestrator_factory=lambda: FakeOrchestrator(
            result
        )
    )


@pytest.mark.parametrize(
    ("result", "expected_route", "expected_terminal"),
    [
        (
            {
                "status": "completed",
                "approval_status": "pending_approval",
                "downstream_export_enabled": False,
                "safe_export": {
                    "approval_status": "pending_approval",
                    "downstream_export_enabled": False,
                },
            },
            "pending_approval",
            "pending_approval",
        ),
        (
            {
                "status": "completed",
                "approval_status": "blocked_stale_source",
                "downstream_export_enabled": False,
                "source_freshness": {
                    "freshness_status": "stale",
                },
            },
            "blocked",
            "blocked",
        ),
        (
            {
                "status": "completed",
                "approval_status": "blocked_no_safe_exact_match",
                "downstream_export_enabled": False,
            },
            "blocked",
            "blocked",
        ),
        (
            {
                "status": "completed",
                "approval_status": "approved",
                "downstream_export_enabled": True,
            },
            "delivery_ready",
            "delivery_ready",
        ),
        (
            {
                "status": "completed",
                "approval_status": "not_required",
                "downstream_export_enabled": False,
            },
            "completed_safe",
            "completed_safe",
        ),
    ],
)
def test_graph_routes_supervisor_decisions(
    result,
    expected_route,
    expected_terminal,
):
    output = build_graph(result).invoke(
        prompt="Find cafe visitors in Montreal"
    )

    assert output["supervisor_route"] == expected_route
    assert output["graph_terminal_status"] == (
        expected_terminal
    )
    assert output["supervisor_graph_trace"][0] == {
        "event": "graph_started"
    }
    assert output["supervisor_graph_trace"][-1] == {
        "event": "graph_terminated",
        "terminal_status": expected_terminal,
    }


def test_graph_converts_orchestrator_exception_to_failed_route():
    class BrokenOrchestrator:
        def run(self, **kwargs):
            raise RuntimeError("provider secret must not leak")

    graph = AudienceSupervisorGraph(
        orchestrator_factory=BrokenOrchestrator
    )
    output = graph.invoke(
        prompt="Find cafe visitors in Montreal"
    )

    assert output["status"] == "failed"
    assert output["supervisor_route"] == "failed"
    assert output["graph_terminal_status"] == "failed"
    assert output["supervisor_graph_error_type"] == (
        "RuntimeError"
    )
    assert "provider secret" not in str(output)
    assert "Find cafe visitors" not in str(
        output["supervisor_graph_trace"]
    )


def test_graph_rejects_empty_prompt():
    graph = build_graph(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
        }
    )

    with pytest.raises(ValueError):
        graph.invoke(prompt="   ")


def test_compiled_graph_contains_expected_nodes():
    graph = build_graph(
        {
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
        }
    )

    graph_nodes = set(graph.graph.get_graph().nodes)
    assert {
        "__start__",
        "run_orchestrator",
        "evaluate_supervisor",
        "route_blocked",
        "route_failed",
        "route_needs_clarification",
        "route_needs_existing_approval",
        "route_pending_approval",
        "route_delivery_ready",
        "route_completed_safe",
        "__end__",
    }.issubset(graph_nodes)
