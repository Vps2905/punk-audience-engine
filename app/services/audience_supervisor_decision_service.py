from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class AudienceSupervisorDecision:
    """Deterministic, fail-closed routing decision for one audience run."""

    route: str
    stage: str
    terminal: bool
    awaiting_input: bool
    approval_required: bool
    downstream_export_enabled: bool
    reason_codes: tuple[str, ...]
    next_action: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


class AudienceSupervisorDecisionService:
    """
    Converts the existing Audience Intelligence result into one authoritative
    supervisor route.

    This service is intentionally framework-independent. It can be used by the
    current orchestrator today and by a LangGraph supervisor later without
    changing the production guardrail semantics.
    """

    BLOCKED_FILTER_MODES = {
        "privacy_identifier_request_blocked",
        "export_action_requires_existing_audience",
        "location_category_gap_no_export",
        "broad_location_no_export",
    }

    def decide(self, result: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return self._decision(
                route="failed",
                stage="supervisor",
                terminal=True,
                awaiting_input=False,
                approval_required=True,
                downstream=False,
                reasons=("invalid_pipeline_result",),
                next_action="Investigate the pipeline result contract before retrying.",
            )

        safe_export = self._dict(result.get("safe_export"))
        prompt_filter = self._dict(result.get("prompt_filter_report"))
        v2 = self._dict(
            result.get("v2_autonomous")
            or result.get("v2")
            or result.get("autonomous_v2")
            or result.get("v2_result")
        )
        freshness = self._dict(
            result.get("source_freshness")
            or safe_export.get("source_freshness")
            or v2.get("data_freshness")
        )
        sensitive = self._dict(
            result.get("sensitive_poi_privacy_risk")
        )

        status = self._norm(result.get("status"))
        approval_status = self._norm(
            result.get("approval_status")
            or safe_export.get("approval_status")
            or safe_export.get("status")
        )
        filter_mode = self._norm(prompt_filter.get("filter_mode"))
        freshness_status = self._norm(
            result.get("freshness_status")
            or freshness.get("freshness_status")
            or freshness.get("status")
        )
        v2_status = self._norm(
            v2.get("status")
            or result.get("v2_status")
        )

        approval_required_value = result.get("approval_required")
        if approval_required_value is None:
            approval_required_value = safe_export.get("approval_required")
        approval_required = (
            bool(approval_required_value)
            if approval_required_value is not None
            else False
        )
        downstream = bool(
            result.get("downstream_export_enabled", False)
            or safe_export.get("downstream_export_enabled", False)
        )
        block_export = bool(
            result.get("block_export", False)
            or safe_export.get("block_export", False)
            or safe_export.get("export_blocked", False)
        )

        if status in {"failed", "error"} or result.get("error"):
            return self._decision(
                route="failed",
                stage="pipeline",
                terminal=True,
                awaiting_input=False,
                approval_required=True,
                downstream=False,
                reasons=("pipeline_failed",),
                next_action="Inspect the failed stage and retry only after the error is resolved.",
            )

        if (
            v2_status == "failed"
            or approval_status == "blocked_v2_failure"
        ):
            return self._blocked(
                stage="v2_intelligence",
                reasons=("v2_failure",),
                next_action="Restore the autonomous V2 stage before approval or delivery.",
            )

        sensitive_decision = self._norm(sensitive.get("decision"))
        if (
            sensitive_decision == "block_export"
            or filter_mode == "privacy_identifier_request_blocked"
            or approval_status in {
                "blocked_privacy",
                "blocked_sensitive",
                "blocked_privacy_budget",
            }
        ):
            return self._blocked(
                stage="privacy",
                reasons=("privacy_guardrail_blocked",),
                next_action="Use only privacy-safe aggregated audience criteria.",
            )

        if (
            freshness_status in {"stale", "expired", "outdated"}
            or "stale" in freshness_status
            or approval_status == "blocked_stale_source"
            or bool(safe_export.get("export_blocked_until_source_refresh"))
        ):
            return self._blocked(
                stage="freshness",
                reasons=("source_refresh_required",),
                next_action="Refresh the upstream source and rerun the audience workflow.",
            )

        if filter_mode == "broad_location_no_export":
            return self._decision(
                route="needs_clarification",
                stage="intent",
                terminal=False,
                awaiting_input=True,
                approval_required=True,
                downstream=False,
                reasons=("location_too_broad",),
                next_action="Ask for a city, area, ZIP code, or exact business location.",
            )

        if filter_mode == "export_action_requires_existing_audience":
            return self._decision(
                route="needs_existing_approval",
                stage="approval",
                terminal=False,
                awaiting_input=True,
                approval_required=True,
                downstream=False,
                reasons=("existing_approved_audience_required",),
                next_action="Select an existing approved audience run before requesting delivery.",
            )

        if approval_status == "blocked_requested_quality_unmet":
            return self._blocked(
                stage="coverage",
                reasons=("requested_quality_unmet",),
                next_action="Wait for stronger/fresher cohorts, use balanced quality when broader coverage is acceptable, or review the quality requirement.",
            )

        if (
            filter_mode == "location_category_gap_no_export"
            or approval_status == "blocked_no_safe_exact_match"
            or self._zero_candidates(result, safe_export)
        ):
            return self._blocked(
                stage="coverage",
                reasons=("no_safe_exact_match",),
                next_action="Wait for matching privacy-safe coverage or revise the targeting criteria.",
            )

        if approval_status.startswith("blocked_") or block_export:
            reason = approval_status or "export_guardrail_blocked"
            return self._blocked(
                stage="guardrail",
                reasons=(reason,),
                next_action="Resolve the blocking guardrail before approval or delivery.",
            )

        if downstream and approval_status != "approved":
            return self._blocked(
                stage="approval",
                reasons=("inconsistent_downstream_state",),
                next_action="Disable downstream delivery until an authoritative approval is recorded.",
            )

        if approval_status == "approved" and downstream:
            return self._decision(
                route="delivery_ready",
                stage="delivery",
                terminal=False,
                awaiting_input=False,
                approval_required=False,
                downstream=True,
                reasons=("approved_for_controlled_delivery",),
                next_action="Execute the explicit downstream delivery action.",
            )

        if approval_required or approval_status in {
            "pending_approval",
            "pending",
        }:
            return self._decision(
                route="pending_approval",
                stage="approval",
                terminal=False,
                awaiting_input=True,
                approval_required=True,
                downstream=False,
                reasons=("manual_approval_required",),
                next_action="Request authorized human review and approval.",
            )

        if status in {"completed", "success", "succeeded"}:
            return self._decision(
                route="completed_safe",
                stage="completed",
                terminal=True,
                awaiting_input=False,
                approval_required=False,
                downstream=False,
                reasons=("completed_without_delivery",),
                next_action="No downstream action is enabled for this run.",
            )

        return self._blocked(
            stage="supervisor",
            reasons=("unknown_pipeline_state",),
            next_action="Review the run state before allowing any approval or delivery action.",
        )

    def _blocked(
        self,
        *,
        stage: str,
        reasons: Iterable[str],
        next_action: str,
    ) -> Dict[str, Any]:
        return self._decision(
            route="blocked",
            stage=stage,
            terminal=True,
            awaiting_input=False,
            approval_required=True,
            downstream=False,
            reasons=tuple(reasons),
            next_action=next_action,
        )

    def _decision(
        self,
        *,
        route: str,
        stage: str,
        terminal: bool,
        awaiting_input: bool,
        approval_required: bool,
        downstream: bool,
        reasons: Iterable[str],
        next_action: str,
    ) -> Dict[str, Any]:
        decision = AudienceSupervisorDecision(
            route=route,
            stage=stage,
            terminal=bool(terminal),
            awaiting_input=bool(awaiting_input),
            approval_required=bool(approval_required),
            downstream_export_enabled=bool(downstream),
            reason_codes=tuple(str(item) for item in reasons if str(item)),
            next_action=str(next_action),
        )
        return decision.to_dict()

    def _zero_candidates(
        self,
        result: Dict[str, Any],
        safe_export: Dict[str, Any],
    ) -> bool:
        values = [
            result.get("prompt_selected_cohorts"),
            safe_export.get("exported_cohorts"),
        ]
        explicit = [value for value in values if value is not None]
        return bool(explicit) and all(self._int(value) == 0 for value in explicit)

    def _dict(self, value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _norm(self, value: Any) -> str:
        return str(value or "").strip().lower()

    def _int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
