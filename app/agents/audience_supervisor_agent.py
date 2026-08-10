from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any, Protocol

from app.models.production_bounded_autonomy_contracts import AutonomyGoal
from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)
from app.services.production_bounded_autonomy_shadow_service import (
    ProductionBoundedAutonomyShadowComparisonService,
)


class OrchestratorProtocol(Protocol):
    def run(self, **kwargs: Any) -> dict[str, Any]: ...


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


def bounded_autonomy_dual_run_enabled(value: Any | None = None) -> bool:
    raw = (
        os.getenv("MODULE5_BOUNDED_AUTONOMY_DUAL_RUN_ENABLED", "false")
        if value is None
        else value
    )
    return str(raw or "").strip().lower() in _TRUE_FLAG_VALUES


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
        from app.agents.autonomous_decision_core_agent import (
            AutonomousDecisionCoreAgent,
        )

        return AutonomousDecisionCoreAgent(
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
        shadow_comparison_service: (
            ProductionBoundedAutonomyShadowComparisonService | None
        ) = None,
        dual_run_enabled: bool | None = None,
    ) -> None:
        self.orchestrator_factory = (
            orchestrator_factory
            or _default_orchestrator_factory
        )
        self.decision_service = (
            decision_service
            or AudienceSupervisorDecisionService()
        )
        # Keep the observer lazy so malformed shadow-only configuration cannot
        # affect the authoritative path while the feature is disabled.
        self.shadow_comparison_service = shadow_comparison_service
        self.dual_run_enabled = dual_run_enabled

    def run(self, **kwargs: Any) -> dict[str, Any]:
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
        if bounded_autonomy_dual_run_enabled(self.dual_run_enabled):
            output["bounded_autonomy_shadow_comparison"] = (
                self._run_shadow_comparison(
                    prompt=prompt,
                    result=output,
                    kwargs=kwargs,
                )
            )
        return output

    def _run_shadow_comparison(
        self,
        *,
        prompt: str,
        result: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        tenant_id = str(
            kwargs.get("tenant_id")
            or kwargs.get("audience_tenant_id")
            or ""
        ).strip()
        if not tenant_id:
            return {
                "status": "engineering_preview_blocked",
                "reason_codes": ["tenant_context_required"],
                "shadow_only": True,
                "production_routing_changed": False,
                "downstream_export_enabled": False,
            }
        run_id = str(result.get("run_id") or "shadow-run")
        lineage = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
        request_id = str(kwargs.get("request_id") or f"shadow-request-{lineage}")
        try:
            goal = AutonomyGoal(
                tenant_id=tenant_id,
                request_id=request_id,
                goal_id=f"shadow-goal-{lineage}",
                objective=prompt,
                requested_outcomes=("governed_recommendation",),
                execution_mode="shadow",
            )
            service = (
                self.shadow_comparison_service
                or ProductionBoundedAutonomyShadowComparisonService()
            )
            return service.run(
                goal=goal,
                legacy_result=result,
            )
        except (TypeError, ValueError):
            return {
                "status": "engineering_preview_blocked",
                "reason_codes": ["shadow_comparison_contract_failure"],
                "shadow_only": True,
                "production_routing_changed": False,
                "downstream_export_enabled": False,
            }

    def evaluate_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return self.decision_service.decide(result)
