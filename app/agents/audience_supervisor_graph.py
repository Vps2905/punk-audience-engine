from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)


class OrchestratorProtocol(Protocol):
    def run(self, **kwargs: Any) -> Dict[str, Any]: ...


OrchestratorFactory = Callable[[], OrchestratorProtocol]


class AudienceWorkflowState(TypedDict, total=False):
    execution_kwargs: Dict[str, Any]
    pipeline_result: Dict[str, Any]
    supervisor_decision: Dict[str, Any]
    supervisor_route: str
    supervisor_stage: str
    terminal_status: str
    graph_trace: list[Dict[str, Any]]
    error_type: str


def _default_orchestrator_factory() -> OrchestratorProtocol:
    from app.agents.audience_intelligence_orchestrator_agent import (
        AudienceIntelligenceOrchestratorAgent,
    )

    return AudienceIntelligenceOrchestratorAgent()


class AudienceSupervisorGraph:
    # Isolated LangGraph workflow around the current production orchestrator.
    # This graph is not connected to the live API path yet. It preserves the
    # existing orchestrator and deterministic guardrails, then uses the
    # supervisor decision service only for conditional routing.

    ROUTE_NODE_MAP = {
        "blocked": "route_blocked",
        "failed": "route_failed",
        "needs_clarification": "route_needs_clarification",
        "needs_existing_approval": "route_needs_existing_approval",
        "pending_approval": "route_pending_approval",
        "delivery_ready": "route_delivery_ready",
        "completed_safe": "route_completed_safe",
    }

    TERMINAL_STATUS_MAP = {
        "route_blocked": "blocked",
        "route_failed": "failed",
        "route_needs_clarification": "needs_clarification",
        "route_needs_existing_approval": "needs_existing_approval",
        "route_pending_approval": "pending_approval",
        "route_delivery_ready": "delivery_ready",
        "route_completed_safe": "completed_safe",
    }

    def __init__(
        self,
        *,
        orchestrator_factory: OrchestratorFactory | None = None,
        decision_service: AudienceSupervisorDecisionService | None = None,
    ) -> None:
        self.orchestrator_factory = (
            orchestrator_factory
            or _default_orchestrator_factory
        )
        self.decision_service = (
            decision_service
            or AudienceSupervisorDecisionService()
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AudienceWorkflowState)

        builder.add_node(
            "run_orchestrator",
            self._run_orchestrator_node,
        )
        builder.add_node(
            "evaluate_supervisor",
            self._evaluate_supervisor_node,
        )

        for node_name in self.TERMINAL_STATUS_MAP:
            builder.add_node(
                node_name,
                self._terminal_node(node_name),
            )

        builder.add_edge(START, "run_orchestrator")
        builder.add_edge(
            "run_orchestrator",
            "evaluate_supervisor",
        )
        builder.add_conditional_edges(
            "evaluate_supervisor",
            self._route_after_supervisor,
            {
                node_name: node_name
                for node_name in self.TERMINAL_STATUS_MAP
            },
        )

        for node_name in self.TERMINAL_STATUS_MAP:
            builder.add_edge(node_name, END)

        return builder.compile()

    def invoke(self, **kwargs: Any) -> Dict[str, Any]:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        initial_state: AudienceWorkflowState = {
            "execution_kwargs": dict(kwargs),
            "graph_trace": [
                {
                    "event": "graph_started",
                }
            ],
        }

        final_state = self.graph.invoke(initial_state)
        result = dict(final_state.get("pipeline_result") or {})

        decision = dict(
            final_state.get("supervisor_decision") or {}
        )
        result["supervisor_decision"] = decision
        result["supervisor_route"] = final_state.get(
            "supervisor_route"
        )
        result["supervisor_stage"] = final_state.get(
            "supervisor_stage"
        )
        result["graph_terminal_status"] = final_state.get(
            "terminal_status"
        )
        result["supervisor_graph_trace"] = list(
            final_state.get("graph_trace") or []
        )

        error_type = final_state.get("error_type")
        if error_type:
            result["supervisor_graph_error_type"] = error_type

        return result

    def _run_orchestrator_node(
        self,
        state: AudienceWorkflowState,
    ) -> AudienceWorkflowState:
        trace = list(state.get("graph_trace") or [])

        try:
            result = self.orchestrator_factory().run(
                **dict(state.get("execution_kwargs") or {})
            )
            if not isinstance(result, dict):
                raise TypeError(
                    "Audience orchestrator must return a dictionary"
                )

            trace.append(
                {
                    "event": "orchestrator_completed",
                    "run_id": result.get("run_id"),
                    "pipeline_status": result.get("status"),
                }
            )
            return {
                "pipeline_result": dict(result),
                "graph_trace": trace,
            }
        except Exception as exc:
            error_type = type(exc).__name__
            trace.append(
                {
                    "event": "orchestrator_failed",
                    "error_type": error_type,
                }
            )
            return {
                "pipeline_result": {
                    "status": "failed",
                    "error": "orchestrator_execution_failed",
                },
                "error_type": error_type,
                "graph_trace": trace,
            }

    def _evaluate_supervisor_node(
        self,
        state: AudienceWorkflowState,
    ) -> AudienceWorkflowState:
        result = dict(state.get("pipeline_result") or {})
        decision = self.decision_service.decide(result)
        route = str(decision.get("route") or "").strip()

        if route not in self.ROUTE_NODE_MAP:
            decision = {
                "route": "blocked",
                "stage": "supervisor",
                "terminal": True,
                "awaiting_input": False,
                "approval_required": True,
                "downstream_export_enabled": False,
                "reason_codes": [
                    "unknown_supervisor_route",
                ],
                "next_action": (
                    "Inspect the supervisor decision contract "
                    "before retrying."
                ),
            }
            route = "blocked"

        trace = list(state.get("graph_trace") or [])
        trace.append(
            {
                "event": "supervisor_evaluated",
                "route": route,
                "stage": decision.get("stage"),
                "reason_codes": list(
                    decision.get("reason_codes") or []
                ),
            }
        )

        return {
            "supervisor_decision": dict(decision),
            "supervisor_route": route,
            "supervisor_stage": decision.get("stage"),
            "graph_trace": trace,
        }

    def _route_after_supervisor(
        self,
        state: AudienceWorkflowState,
    ) -> str:
        route = str(state.get("supervisor_route") or "")
        return self.ROUTE_NODE_MAP.get(
            route,
            "route_blocked",
        )

    def _terminal_node(
        self,
        node_name: str,
    ):
        terminal_status = self.TERMINAL_STATUS_MAP[node_name]

        def node(
            state: AudienceWorkflowState,
        ) -> AudienceWorkflowState:
            trace = list(state.get("graph_trace") or [])
            trace.append(
                {
                    "event": "graph_terminated",
                    "terminal_status": terminal_status,
                }
            )
            return {
                "terminal_status": terminal_status,
                "graph_trace": trace,
            }

        return node
