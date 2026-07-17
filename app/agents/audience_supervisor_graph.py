from __future__ import annotations

import time
from typing import Any, Callable, Dict, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)
from app.services.audience_supervisor_recovery_service import (
    AudienceSupervisorRecoveryService,
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
    recovery: Dict[str, Any]
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
        recovery_service: AudienceSupervisorRecoveryService | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.orchestrator_factory = (
            orchestrator_factory
            or _default_orchestrator_factory
        )
        self.decision_service = (
            decision_service
            or AudienceSupervisorDecisionService()
        )
        self.recovery_service = (
            recovery_service
            or AudienceSupervisorRecoveryService()
        )
        self.sleep_fn = sleep_fn or time.sleep
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

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        return self.invoke(**kwargs)

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
        result["supervisor_reason_codes"] = list(
            decision.get("reason_codes") or []
        )
        result["graph_terminal_status"] = final_state.get(
            "terminal_status"
        )
        graph_trace = list(
            final_state.get("graph_trace") or []
        )
        result["supervisor_graph_trace"] = graph_trace
        result["supervisor_trace"] = graph_trace

        recovery = dict(final_state.get("recovery") or {})
        if recovery:
            result["supervisor_recovery"] = recovery

        error_type = final_state.get("error_type")
        if error_type:
            result["supervisor_graph_error_type"] = error_type

        return result

    def _run_orchestrator_node(
        self,
        state: AudienceWorkflowState,
    ) -> AudienceWorkflowState:
        trace = list(state.get("graph_trace") or [])
        execution_kwargs = dict(
            state.get("execution_kwargs") or {}
        )
        max_attempts = self.recovery_service.max_attempts()
        backoff_seconds = (
            self.recovery_service.retry_backoff_seconds()
        )
        last_decision: Dict[str, Any] = {}
        orchestrator = self.orchestrator_factory()

        for attempt in range(1, max_attempts + 1):
            trace.append(
                {
                    "event": "orchestrator_attempt_started",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                }
            )

            try:
                result = orchestrator.run(
                    **execution_kwargs
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
                        "attempt": attempt,
                    }
                )
                return {
                    "pipeline_result": dict(result),
                    "recovery": {
                        "attempt_count": attempt,
                        "max_attempts": max_attempts,
                        "retried": attempt > 1,
                        "exhausted": False,
                        "last_error_category": (
                            last_decision.get("error_category")
                        ),
                    },
                    "graph_trace": trace,
                }
            except Exception as exc:
                last_decision = (
                    self.recovery_service.classify(exc)
                )
                retryable = bool(
                    last_decision.get("retryable")
                )
                safe_error_type = str(
                    last_decision.get("safe_error_type")
                    or type(exc).__name__
                )
                error_category = str(
                    last_decision.get("error_category")
                    or "non_transient"
                )
                should_retry = (
                    retryable and attempt < max_attempts
                )

                trace.append(
                    {
                        "event": "orchestrator_attempt_failed",
                        "attempt": attempt,
                        "error_type": safe_error_type,
                        "error_category": error_category,
                        "retryable": retryable,
                    }
                )

                if should_retry:
                    delay = backoff_seconds * attempt
                    trace.append(
                        {
                            "event": "orchestrator_retry_scheduled",
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "delay_seconds": delay,
                        }
                    )
                    if delay:
                        self.sleep_fn(delay)
                    continue

                trace.append(
                    {
                        "event": "orchestrator_failed",
                        "error_type": safe_error_type,
                        "error_category": error_category,
                        "attempt_count": attempt,
                        "retry_exhausted": (
                            retryable
                            and attempt >= max_attempts
                        ),
                    }
                )
                return {
                    "pipeline_result": {
                        "status": "failed",
                        "error": "orchestrator_execution_failed",
                    },
                    "recovery": {
                        "attempt_count": attempt,
                        "max_attempts": max_attempts,
                        "retried": attempt > 1,
                        "exhausted": (
                            retryable
                            and attempt >= max_attempts
                        ),
                        "last_error_category": error_category,
                    },
                    "error_type": safe_error_type,
                    "graph_trace": trace,
                }

        return {
            "pipeline_result": {
                "status": "failed",
                "error": "orchestrator_execution_failed",
            },
            "recovery": {
                "attempt_count": max_attempts,
                "max_attempts": max_attempts,
                "retried": max_attempts > 1,
                "exhausted": True,
                "last_error_category": "unknown",
            },
            "error_type": "UnknownError",
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
