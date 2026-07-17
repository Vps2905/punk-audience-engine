from __future__ import annotations

import os
from typing import Any, Callable, Dict, Protocol

from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)


class OrchestratorProtocol(Protocol):
    def run(self, **kwargs: Any) -> Dict[str, Any]: ...


OrchestratorFactory = Callable[[], OrchestratorProtocol]


_TRUE_FLAG_VALUES = {
    "1",
    "true",
    "yes",
    "on",
    "enabled",
}
_FALSE_FLAG_VALUES = {
    "",
    "0",
    "false",
    "no",
    "off",
    "disabled",
}


def autonomous_supervisor_enabled(
    value: Any | None = None,
) -> bool:
    # Unknown values preserve the existing orchestrator path. Once explicitly
    # enabled, supervisor execution remains fail-closed.
    raw = (
        os.getenv(
            "ENABLE_AUTONOMOUS_SUPERVISOR",
            "false",
        )
        if value is None
        else value
    )
    normalized = str(raw or "").strip().lower()

    if normalized in _TRUE_FLAG_VALUES:
        return True
    if normalized in _FALSE_FLAG_VALUES:
        return False

    return False


def autonomous_supervisor_graph_enabled(
    value: Any | None = None,
) -> bool:
    raw = (
        os.getenv(
            "ENABLE_AUTONOMOUS_SUPERVISOR_GRAPH",
            "false",
        )
        if value is None
        else value
    )
    normalized = str(raw or "").strip().lower()

    if normalized in _TRUE_FLAG_VALUES:
        return True
    if normalized in _FALSE_FLAG_VALUES:
        return False

    return False


def build_audience_execution_agent(
    *,
    orchestrator_factory: OrchestratorFactory | None = None,
) -> OrchestratorProtocol:
    # The parent supervisor gate must be enabled before graph execution can
    # be selected. Unknown flag values preserve the safer legacy path.
    factory = (
        orchestrator_factory
        or _default_orchestrator_factory
    )

    if not autonomous_supervisor_enabled():
        return factory()

    if autonomous_supervisor_graph_enabled():
        from app.agents.audience_supervisor_graph import (
            AudienceSupervisorGraph,
        )

        return AudienceSupervisorGraph(
            orchestrator_factory=factory,
        )

    return AudienceSupervisorAgent(
        orchestrator_factory=factory,
    )


def _default_orchestrator_factory() -> OrchestratorProtocol:
    # Lazy import keeps the supervisor decision layer independently testable
    # and avoids loading database/provider dependencies until execution time.
    from app.agents.audience_intelligence_orchestrator_agent import (
        AudienceIntelligenceOrchestratorAgent,
    )

    return AudienceIntelligenceOrchestratorAgent()


class AudienceSupervisorAgent:
    """
    Backward-compatible supervisor wrapper around the existing production
    orchestrator.

    Stage 1 does not replace the orchestrator or alter API routing. It attaches
    deterministic supervisor metadata that can later drive conditional graph
    edges behind a feature flag.
    """

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

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        result = self.orchestrator_factory().run(**kwargs)
        if not isinstance(result, dict):
            raise TypeError("Audience orchestrator must return a dictionary")

        decision = self.decision_service.decide(result)
        output = dict(result)
        output["supervisor_decision"] = decision
        output["supervisor_route"] = decision["route"]
        output["supervisor_stage"] = decision["stage"]
        output["supervisor_reason_codes"] = list(
            decision.get("reason_codes") or []
        )
        output["supervisor_trace"] = [
            {
                "event": "pipeline_completed",
                "run_id": output.get("run_id"),
                "pipeline_status": output.get("status"),
            },
            {
                "event": "supervisor_decision",
                "route": decision["route"],
                "stage": decision["stage"],
                "reason_codes": list(
                    decision.get("reason_codes") or []
                ),
            },
        ]
        return output

    def evaluate_result(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.decision_service.decide(result)
