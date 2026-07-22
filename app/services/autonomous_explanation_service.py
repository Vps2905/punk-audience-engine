import json
from pathlib import Path
from app.models.autonomous_decision_state import AutonomousDecisionState

class AutonomousExplanationService:
    def build_explanation(self, state: AutonomousDecisionState, output_dir: Path) -> str:
        lines = []
        lines.append(f"# Audience Autonomous Decision Report")
        lines.append(f"Request ID: {state.request_id}")
        lines.append(f"Prompt: {state.original_prompt}")
        lines.append(f"Status: {state.status}")
        lines.append("")

        if state.constraint_ledger:
            lines.append("## Explicit Constraints")

            matched_locs = [m.lower() for m in (state.coverage_state.matched_locations if state.coverage_state else [])]
            for loc in state.constraint_ledger.locations:
                status = loc.resolution_status
                if status == "unresolved" and state.coverage_state and state.coverage_state.coverage_status == "complete":
                    if loc.normalized_value in matched_locs or any(loc.normalized_value in m for m in matched_locs):
                        status = "matched"

                lines.append(f"- explicit constraint: {loc.raw_text}")
                lines.append(f"  - coverage result: {status}")

            for cat in state.constraint_ledger.categories:
                lines.append(f"- explicit constraint: {cat.raw_text}")
                lines.append(f"  - coverage result: {cat.resolution_status}")
            for dp in state.constraint_ledger.dayparts:
                lines.append(f"- explicit constraint: {dp.raw_text}")
                lines.append(f"  - coverage result: {dp.resolution_status}")
            if state.constraint_ledger.quality_objective:
                lines.append(f"- Quality Objective: {state.constraint_ledger.quality_objective.raw_text}")
            if state.constraint_ledger.reach_objective:
                lines.append(f"- Reach Objective: {state.constraint_ledger.reach_objective.raw_text}")
            lines.append("")

        if state.semantic_decision:
            lines.append("## Semantic Interpretation")
            lines.append(f"Business Intent: {state.semantic_decision.business_intent}")
            lines.append(f"Quality Intent: {state.semantic_decision.quality_intent}")
            lines.append(f"Objective Relationship: {state.semantic_decision.objective_relationship}")
            lines.append(f"Resolver Mode: {state.semantic_decision.resolver_mode}")
            lines.append("")

        if state.coverage_state:
            lines.append("## Coverage")
            lines.append(f"Status: {state.coverage_state.coverage_status}")
            lines.append(f"Matched Locations: {', '.join(state.coverage_state.matched_locations)}")
            if state.coverage_state.missing_locations:
                lines.append(f"Missing Locations: {', '.join(state.coverage_state.missing_locations)}")
            lines.append("")

        if state.quality_state:
            lines.append("## Quality Enforcement")
            lines.append(f"Status: {state.quality_state.quality_policy_status}")
            if state.quality_state.quality_intent == "balanced":
                lines.append("Explanation: Explicitly satisfied co-optimization of reliability and audience scale.")
            lines.append(f"Evaluated Locations: {', '.join(state.quality_state.evaluated_locations)}")
            lines.append(f"Locations Meeting Quality: {', '.join(state.quality_state.locations_meeting_quality)}")
            lines.append(f"Candidates Before: {state.quality_state.candidates_before}")
            lines.append(f"Candidates After: {state.quality_state.candidates_after}")
            lines.append("")

        if state.repair_history:
            lines.append("## Repair History")
            lines.append(f"Repair Attempts: {state.repair_attempts}")
            for r in state.repair_history:
                lines.append(f"Attempt {r.attempt}: fixed {', '.join(r.issue_codes)}")
            lines.append("")

        explanation = "\n".join(lines)

        # Also write the state artifacts
        output_dir.mkdir(parents=True, exist_ok=True)
        decision_dir = output_dir / "07_autonomous_decision"
        decision_dir.mkdir(parents=True, exist_ok=True)

        from app.utils.serialization import make_serializable

        if state.constraint_ledger:
            with open(decision_dir / "constraint_ledger.json", "w") as f:
                json.dump(make_serializable(state.constraint_ledger.model_dump()), f, indent=2)

        if state.semantic_decision:
            with open(decision_dir / "semantic_decision.json", "w") as f:
                json.dump(make_serializable(state.semantic_decision.model_dump()), f, indent=2)

        if state.execution_plan:
            with open(decision_dir / "execution_plan.json", "w") as f:
                json.dump(make_serializable(state.execution_plan.model_dump()), f, indent=2)

        with open(decision_dir / "verification_report.json", "w") as f:
            json.dump(make_serializable([c.model_dump() for c in (state.contradictions or [])]), f, indent=2)

        with open(decision_dir / "repair_history.json", "w") as f:
            json.dump(make_serializable([r.model_dump() for r in (state.repair_history or [])]), f, indent=2)

        with open(decision_dir / "final_decision.json", "w") as f:
            json.dump(make_serializable({
                "status": state.status,
                "request_id": state.request_id,
                "run_id": state.run_id,
                "explanation": explanation
            }), f, indent=2)

        with open(decision_dir / "world_state.json", "w") as f:
            json.dump(make_serializable(state.model_dump()), f, indent=2)

        with open(decision_dir / "final_explanation.md", "w") as f:
            f.write(explanation)

        return explanation
