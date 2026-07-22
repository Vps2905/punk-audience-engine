from typing import Any, Callable, Dict
import time
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.models.autonomous_decision_state import (
    AutonomousDecisionState,
    REPAIR_IMPACT_REEXECUTE_PIPELINE,
    REPAIR_IMPACT_REASSEMBLE_STATE,
)
from app.services.constraint_ledger_service import ConstraintLedgerService
from app.services.autonomous_plan_service import AutonomousPlanService
from app.services.decision_verification_service import DecisionVerificationService
from app.services.decision_repair_service import DecisionRepairService
from app.services.autonomous_explanation_service import AutonomousExplanationService

# In order to invoke Module 1-4 logic, we'll wrap the existing orchestrator call.
from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent
from app.services.local_semantic_intent_service import LocalSemanticIntentService


class AutonomousDecisionCoreAgent:
    def __init__(self, orchestrator_factory: Callable[[], Any] = None):
        self.orchestrator_factory = orchestrator_factory or AudienceIntelligenceOrchestratorAgent
        self.constraint_service = ConstraintLedgerService()
        self.plan_service = AutonomousPlanService()
        self.verification_service = DecisionVerificationService()
        self.repair_service = DecisionRepairService()
        self.explanation_service = AutonomousExplanationService()
        # Graph will be built dynamically in run() to inject the correct checkpointer

    def _build_graph(self, checkpointer):
        builder = StateGraph(AutonomousDecisionState)

        builder.add_node("capture_constraints", self._capture_constraints)
        builder.add_node("interpret_semantics", self._interpret_semantics)
        builder.add_node("validate_intent", self._validate_intent)
        builder.add_node("build_execution_plan", self._build_execution_plan)
        builder.add_node("execute_audience_pipeline", self._execute_audience_pipeline)
        builder.add_node("assemble_world_state", self._assemble_world_state)
        builder.add_node("verify_decision", self._verify_decision)
        builder.add_node("plan_repair", self._plan_repair)
        builder.add_node("apply_repair", self._apply_repair)
        builder.add_node("request_clarification", self._request_clarification)
        builder.add_node("block_decision", self._block_decision)
        builder.add_node("finalize_decision", self._finalize_decision)

        builder.add_edge(START, "capture_constraints")
        builder.add_edge("capture_constraints", "interpret_semantics")
        builder.add_edge("interpret_semantics", "validate_intent")

        builder.add_conditional_edges(
            "validate_intent",
            self._route_validation,
            {
                "valid": "build_execution_plan",
                "ambiguous": "request_clarification",
                "invalid": "block_decision"
            }
        )

        builder.add_edge("build_execution_plan", "execute_audience_pipeline")
        builder.add_edge("execute_audience_pipeline", "assemble_world_state")
        builder.add_edge("assemble_world_state", "verify_decision")

        builder.add_conditional_edges(
            "verify_decision",
            self._route_verification,
            {
                "verified": "finalize_decision",
                "repairable": "plan_repair",
                "clarification": "request_clarification",
                "blocked": "block_decision"
            }
        )

        builder.add_edge("plan_repair", "apply_repair")

        # Conditional routing after repair: constraint repairs re-execute pipeline,
        # state-only repairs skip directly to reassembly.
        builder.add_conditional_edges(
            "apply_repair",
            self._route_repair,
            {
                "reexecute": "execute_audience_pipeline",
                "reassemble": "assemble_world_state",
                "blocked": "block_decision",
            }
        )

        builder.add_edge("request_clarification", END)
        builder.add_edge("block_decision", END)
        builder.add_edge("finalize_decision", END)

        return builder.compile(checkpointer=checkpointer)


    def run(self, **kwargs: Any) -> Dict[str, Any]:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        initial_state = AutonomousDecisionState(
            request_id=kwargs.get("request_id") or str(uuid.uuid4()),
            run_id=kwargs.get("run_id") or str(uuid.uuid4()),
            original_prompt=prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="interpreting",
            execution_context=dict(kwargs)
        )

        backend = os.getenv("AUDIENCE_JOB_STORE_BACKEND", "local").strip().lower()
        if backend in {"postgres", "postgresql", "pg"}:
            db_url = os.getenv("ECHO_DATABASE_URL") or os.getenv("DATABASE_URL")
            if not db_url:
                raise RuntimeError("Persistent checkpointing requires DATABASE_URL")

            try:
                from psycopg_pool import ConnectionPool
                from langgraph.checkpoint.postgres import PostgresSaver

                with ConnectionPool(db_url) as pool:
                    checkpointer = PostgresSaver(pool)
                    checkpointer.setup()
                    self.checkpointer = checkpointer
                    self.graph = self._build_graph(checkpointer)

                    config = {"configurable": {"thread_id": initial_state.request_id}}
                    final_state_dict = self.graph.invoke(initial_state, config=config)
                    final_state = AutonomousDecisionState(**final_state_dict)
            except Exception as e:
                # Fail closed if production requires persistent checkpointing but backend unavailable
                raise RuntimeError(f"Failed to connect to production checkpointer: {e}")
        else:
            self.checkpointer = MemorySaver()
            self.graph = self._build_graph(self.checkpointer)

            config = {"configurable": {"thread_id": initial_state.request_id}}
            final_state_dict = self.graph.invoke(initial_state, config=config)
            final_state = AutonomousDecisionState(**final_state_dict)

        from app.utils.serialization import make_serializable

        # Build explanation and artifacts
        output_dir = Path(kwargs.get("output_root") or "/tmp") / final_state.run_id
        explanation = self.explanation_service.build_explanation(final_state, output_dir)

        # Return a structure compatible with existing API
        # Apply serialization normalization to prevent msgpack or json errors
        result = make_serializable(dict(final_state.tool_results))

        result.update({
            "status": final_state.status,
            "run_id": final_state.run_id,
            "supervisor_decision": {
                "route": final_state.status,
                "stage": "autonomous_decision",
                "reason_codes": [i.issue_code for i in final_state.contradictions],
            },
            "supervisor_route": final_state.status,
            "supervisor_stage": "autonomous_decision",
            "supervisor_reason_codes": [i.issue_code for i in final_state.contradictions] or ["approval_gate_unmet"],
            "graph_terminal_status": final_state.status,
            "autonomous_explanation": explanation,
            "supervisor_graph_trace": [{"event": "graph_terminated", "terminal_status": final_state.status}],
            "supervisor_trace": [{"event": "graph_terminated", "terminal_status": final_state.status}],
            "pipeline_execution_count": final_state.pipeline_execution_count,
        })

        return result

    # --- Node Implementations ---

    def _capture_constraints(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        state.constraint_ledger = self.constraint_service.capture(state.original_prompt)
        return state

    def _interpret_semantics(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        # Instead of calling HybridSemanticIntentAgent which requires safe cohorts,
        # we will use the LocalSemanticIntentService which uses standard sentence transformers.
        from app.models.autonomous_decision_state import SemanticDecision

        local_service = LocalSemanticIntentService()

        try:
            # We mock the rag context for now just to get basic semantic scores
            res = local_service.resolve(prompt=state.original_prompt, rag_context={})
        except Exception as e:
            res = {}

        # Combine with constraints
        ledger = state.constraint_ledger

        locs = [l.normalized_value for l in ledger.locations]
        cats = [c.normalized_value for c in ledger.categories]
        dps = [d.normalized_value for d in ledger.dayparts]

        prompt_text = state.original_prompt.lower()

        qi = "balanced"
        relation = "co_optimize"
        reasoning = "Defaulted to balanced quality relationship"
        conf = 0.5
        source = "default"

        # 1. Explicit equal-weight or balance relationship
        if any(term in prompt_text for term in ["balance", "equally important", "sensible", "reasonable", "both important", "co-optimized", "neither should dominate", "without sacrificing"]):
            qi = "balanced"
            relation = "co_optimize"
            reasoning = "Explicit balance or equal-weight relationship detected"
            conf = 0.9
            source = "explicit_relationship"
        # 2. Explicit sacrifice/trade-off statement (e.g. prioritize X even with smaller Y, or maximize X and accept varying Y)
        elif "prioritize" in prompt_text and ("smaller" in prompt_text or "reduced" in prompt_text):
            qi = "high"
            relation = "quality_dominant"
            reasoning = "Explicit trade-off prioritizing quality over scale"
            conf = 0.9
            source = "explicit_tradeoff"
        elif "maximize reach" in prompt_text and "varying quality" in prompt_text:
            qi = "broad"
            relation = "reach_dominant"
            reasoning = "Explicit trade-off prioritizing reach over quality"
            conf = 0.9
            source = "explicit_tradeoff"
        elif "accept" in prompt_text and ("lower" in prompt_text or "variable" in prompt_text or "weaker" in prompt_text):
            qi = "broad"
            relation = "reach_dominant"
            reasoning = "Explicit trade-off accepting lower quality"
            conf = 0.9
            source = "explicit_tradeoff"
        # 3 & 4. Full-sentence semantic / keywords
        elif "high quality" in prompt_text or "premium" in prompt_text or "dependable" in prompt_text or "best" in prompt_text:
            qi = "high"
            relation = "quality_dominant"
            reasoning = "Quality-prioritizing keywords detected"
            conf = 0.7
            source = "keyword_inference"
        elif "reach" in prompt_text or "maximum" in prompt_text or "widest" in prompt_text or "volume" in prompt_text or "scale" in prompt_text:
            qi = "broad"
            relation = "reach_dominant"
            reasoning = "Reach-prioritizing keywords detected"
            conf = 0.7
            source = "keyword_inference"

        state.semantic_decision = SemanticDecision(
            business_intent=state.original_prompt,
            locations=locs,
            categories=cats,
            dayparts=dps,
            quality_intent=qi,
                objective_relationship=relation,
                quality_reasoning=reasoning,
                confidence=conf,
                evidence_source=source,
            )

        return state

    def _validate_intent(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        if not state.semantic_decision:
            state.status = "blocked_verification_failed"
        elif state.constraint_ledger and state.constraint_ledger.ambiguous_constraints:
            state.status = "needs_clarification"
        else:
            state.status = "planning"
        return state

    def _route_validation(self, state: AutonomousDecisionState) -> str:
        if state.status == "planning": return "valid"
        if state.status == "needs_clarification": return "ambiguous"
        return "invalid"

    def _build_execution_plan(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        state.execution_plan = self.plan_service.build_plan(state.request_id, state.original_prompt)
        state.status = "executing"
        return state

    def _execute_audience_pipeline(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        # Invoke the existing orchestrator using the extracted semantic info as kwargs
        kwargs = dict(state.execution_context)

        # Inject the parsed semantic intent into kwargs so the orchestrator
        # uses our verified quality relationship rather than re-inferring.
        if state.semantic_decision:
            kwargs["quality_intent"] = state.semantic_decision.quality_intent
            kwargs["semantic_intent"] = {
                "quality_intent": state.semantic_decision.quality_intent,
                "objective_relationship": state.semantic_decision.objective_relationship,
                "locations": state.semantic_decision.locations,
                "categories": state.semantic_decision.categories,
                "dayparts": state.semantic_decision.dayparts,
            }

        orchestrator = self.orchestrator_factory()

        from app.utils.serialization import make_serializable

        try:
            result = orchestrator.run(**kwargs)
            state.tool_results = make_serializable(result)
            state.pipeline_execution_count += 1
            if state.execution_plan:
                state.completed_steps = [s.step_id for s in state.execution_plan.steps]
        except Exception as e:
            state.tool_results = {"status": "failed", "error": str(e)}
            state.pipeline_execution_count += 1

        return state

    def _assemble_world_state(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        # Translate the orchestrator's output into world state using canonical schema
        from app.models.autonomous_decision_state import CoverageState, QualityState

        res = state.tool_results

        # The orchestrator returns data under prompt_filter_report (not filter_report)
        pfr = res.get("prompt_filter_report") or {}

        # v2_guided_selection contains the authoritative coverage/fulfillment data
        v2gs = pfr.get("v2_guided_selection") or {}

        # safe_export contains approval and downstream state
        safe_export = res.get("safe_export") or {}

        # Assemble Coverage from canonical schema paths
        cov = CoverageState(
            requested_locations=[l.normalized_value for l in state.constraint_ledger.locations],
            requested_categories=[c.normalized_value for c in state.constraint_ledger.categories],
            requested_dayparts=[d.normalized_value for d in state.constraint_ledger.dayparts],
        )

        # Matched locations: prefer v2_guided_selection → prompt_filter_report → filter_report
        cov.matched_locations = (
            v2gs.get("matched_requested_locations")
            or pfr.get("matched_requested_locations")
            or pfr.get("locations_detected")
            or []
        )

        # Missing locations
        missing_from_report = (
            v2gs.get("missing_requested_locations")
            or pfr.get("missing_requested_locations")
            or []
        )
        # Also compute from ledger diff
        for req_loc in cov.requested_locations:
            matched = any(
                req_loc.lower() in m.lower() or m.lower() in req_loc.lower()
                for m in cov.matched_locations
            )
            if not matched and req_loc not in cov.missing_locations:
                cov.missing_locations.append(req_loc)
        for m in missing_from_report:
            if m not in cov.missing_locations:
                cov.missing_locations.append(m)

        # Coverage status: always recompute from ledger truth, never trust orchestrator's
        # claim since it doesn't know about our constraint ledger.
        if not cov.missing_locations:
            cov.coverage_status = "complete"
        elif cov.matched_locations:
            cov.coverage_status = "partial"
        else:
            cov.coverage_status = "blocked"

        # Exact candidate count
        cov.exact_candidate_count = (
            int(v2gs.get("rows") or 0)
            or int(res.get("prompt_selected_cohorts") or 0)
        )

        state.coverage_state = cov

        # Assemble Quality from prompt_filter_report.quality_policy_report or v2gs
        qpr = pfr.get("quality_policy_report") or {}
        qi = v2gs.get("quality_intent") or pfr.get("quality_intent") or qpr.get("quality_intent") or "unknown"

        # Get semantic intent fields if available
        sd = state.semantic_decision

        state.quality_state = QualityState(
            quality_intent=qi,
            objective_relationship=sd.objective_relationship if sd else "ambiguous",
            quality_reasoning=sd.quality_reasoning if sd else "",
            evidence_source=sd.evidence_source if sd else "",
            confidence=sd.confidence if sd else 0.0,
            evaluated_locations=list(cov.matched_locations),
            locations_meeting_quality=qpr.get("locations_meeting_quality", []),
            locations_missing_quality=qpr.get("locations_missing_quality", []),
            candidates_before=int(qpr.get("quality_candidates_before") or 0),
            candidates_after=int(qpr.get("quality_candidates_after") or 0),
            excluded_count=int(qpr.get("quality_excluded_count") or 0),
            quality_score_field=qpr.get("quality_score_field", ""),
            quality_score_sources=qpr.get("quality_score_sources", []),
            thresholds=qpr.get("quality_thresholds_by_location") or {},
            quality_policy_status=qpr.get("quality_policy_status") or v2gs.get("quality_policy_report", {}).get("quality_policy_status") or "pending_final_quality_evaluation"
        )

        # Assemble Privacy, Freshness, Approval, Export from canonical paths
        state.privacy_state = res.get("privacy_guarantees") or {}

        # Freshness: block_export at root level
        state.freshness_state = {
            "source": res.get("source_mode"),
            "block_export": bool(
                res.get("block_export")
                or safe_export.get("block_export")
                or v2gs.get("block_export")
            ),
        }

        # Approval from safe_export (canonical) or root
        approval_status = (
            safe_export.get("approval_status")
            or res.get("approval_status")
            or "unknown"
        )
        state.approval_state = {
            "approval_status": approval_status,
            "approval_required": res.get("approval_required", True),
        }

        # Export state from safe_export
        state.export_state = {
            "downstream_export_enabled": bool(
                res.get("downstream_export_enabled")
                or safe_export.get("downstream_export_enabled")
            ),
            "prepared_cohorts": int(safe_export.get("exported_cohorts") or 0),
            "prepared_lookalike_pairs": int(safe_export.get("exported_lookalike_pairs") or 0),
        }

        state.status = "verifying"
        return state

    def _verify_decision(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        issues = self.verification_service.verify(state)
        state.contradictions = issues

        if not issues:
            state.status = "completed"
        else:
            if any(i.severity == "critical" and not i.repairable for i in issues):
                state.status = "blocked_verification_failed"
            elif any(i.repairable for i in issues):
                if state.repair_attempts >= 2:
                    state.status = "blocked_verification_failed"
                else:
                    state.status = "repairing"
            else:
                state.status = "completed" # Warning only
        return state

    def _route_verification(self, state: AutonomousDecisionState) -> str:
        if state.status == "completed": return "verified"
        if state.status == "repairing": return "repairable"
        if state.status == "needs_clarification": return "clarification"
        return "blocked"

    def _plan_repair(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        # Just logging or preparation
        return state

    def _apply_repair(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        repaired = self.repair_service.repair(state)
        if not repaired:
            state.status = "blocked_verification_failed"
        return state

    def _route_repair(self, state: AutonomousDecisionState) -> str:
        """Conditional routing after repair based on repair impact type."""
        if state.status == "blocked_verification_failed":
            return "blocked"
        if state.last_repair_impact == REPAIR_IMPACT_REEXECUTE_PIPELINE:
            return "reexecute"
        return "reassemble"

    def _request_clarification(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        state.status = "needs_clarification"
        return state

    def _block_decision(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        if not state.status.startswith("blocked"):
            state.status = "blocked_verification_failed"
        # Always disable export on block
        if state.export_state:
            state.export_state["downstream_export_enabled"] = False
        return state

    def _finalize_decision(self, state: AutonomousDecisionState) -> AutonomousDecisionState:
        """Determine final status using strict precedence rules.

        Precedence (highest to lowest):
        1. privacy or verification failure
        2. unresolved required constraint
        3. no exact safe coverage (only when coverage truly blocked AND zero candidates)
        4. requested quality unmet
        5. stale source
        6. pending approval
        7. approved/completed
        """
        # 1. Privacy or verification failure
        if state.privacy_state and state.privacy_state.get("individual_fields_exposed"):
            state.status = "blocked_verification_failed"
        # 2. Unresolved required constraint
        elif state.constraint_ledger and state.constraint_ledger.unresolved_constraints:
            state.status = "blocked_unresolved_constraint"
        # 3. No exact safe coverage — only when truly blocked with zero candidates
        elif (state.coverage_state
              and state.coverage_state.coverage_status == "blocked"
              and state.coverage_state.exact_candidate_count == 0
              and not state.coverage_state.matched_locations):
            state.status = "blocked_no_safe_exact_match"
        # 4. Requested quality unmet
        elif state.quality_state and state.quality_state.quality_policy_status == "unmet":
            state.status = "blocked_requested_quality_unmet"
        # 5. Stale source
        elif state.freshness_state and state.freshness_state.get("block_export"):
            state.status = "blocked_stale_source"
        # 6. Pending approval
        elif state.approval_state and state.approval_state.get("approval_status") in {
            "pending_approval", "blocked_stale_source"
        }:
            approval = state.approval_state["approval_status"]
            if approval == "blocked_stale_source":
                state.status = "blocked_stale_source"
            else:
                state.status = "pending_approval"
        else:
            state.status = "completed"

        # Ensure export is disabled if not completed or approved
        if state.status not in ["completed", "approved"]:
            if state.export_state:
                state.export_state["downstream_export_enabled"] = False

        return state
