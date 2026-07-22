import datetime
from app.models.autonomous_decision_state import (
    AutonomousDecisionState,
    RepairRecord,
    REPAIR_IMPACT_REEXECUTE_PIPELINE,
    REPAIR_IMPACT_REASSEMBLE_STATE,
    REPAIR_IMPACT_EXPLANATION_ONLY,
    REPAIR_IMPACT_NON_REPAIRABLE,
)

class DecisionRepairService:
    def repair(self, state: AutonomousDecisionState) -> bool:
        if not state.contradictions:
            return False

        repaired_any = False
        codes = []
        fields = []
        impact = REPAIR_IMPACT_REASSEMBLE_STATE

        for issue in state.contradictions:
            if not issue.repairable:
                continue

            if issue.issue_code == "explicit_location_dropped":
                if state.constraint_ledger.locations:
                    state.semantic_decision.locations = [loc.normalized_value for loc in state.constraint_ledger.locations]
                    codes.append(issue.issue_code)
                    fields.append("semantic_decision.locations")
                    # Constraint repair requires pipeline re-execution
                    impact = REPAIR_IMPACT_REEXECUTE_PIPELINE
                    repaired_any = True

            elif issue.issue_code == "explicit_category_dropped":
                if state.constraint_ledger.categories:
                    state.semantic_decision.categories = [cat.normalized_value for cat in state.constraint_ledger.categories]
                    codes.append(issue.issue_code)
                    fields.append("semantic_decision.categories")
                    impact = REPAIR_IMPACT_REEXECUTE_PIPELINE
                    repaired_any = True

            elif issue.issue_code == "explicit_daypart_dropped":
                if state.constraint_ledger.dayparts:
                    state.semantic_decision.dayparts = [dp.normalized_value for dp in state.constraint_ledger.dayparts]
                    codes.append(issue.issue_code)
                    fields.append("semantic_decision.dayparts")
                    impact = REPAIR_IMPACT_REEXECUTE_PIPELINE
                    repaired_any = True

            elif issue.issue_code == "unsupported_location_evaluated":
                if state.quality_state and state.coverage_state:
                    state.quality_state.evaluated_locations = [
                        loc for loc in state.quality_state.evaluated_locations
                        if loc not in state.coverage_state.missing_locations
                    ]
                    codes.append(issue.issue_code)
                    fields.append("quality_state.evaluated_locations")
                    repaired_any = True

            elif issue.issue_code in {"unapproved_downstream_export", "stale_source_with_export_enabled", "blocked_result_described_as_approved"}:
                if state.export_state:
                    state.export_state["downstream_export_enabled"] = False
                    codes.append(issue.issue_code)
                    fields.append("export_state.downstream_export_enabled")
                    repaired_any = True

            elif issue.issue_code == "individual_field_leakage":
                if state.privacy_state:
                    state.privacy_state["individual_fields_exposed"] = False
                    codes.append(issue.issue_code)
                    fields.append("privacy_state.individual_fields_exposed")
                    repaired_any = True

            elif issue.issue_code == "excluded_cohort_in_lookalike":
                if state.tool_results and "lookalikes" in state.tool_results:
                    state.tool_results["lookalikes"] = []
                    codes.append(issue.issue_code)
                    fields.append("tool_results.lookalikes")
                    repaired_any = True

            elif issue.issue_code == "incompatible_pipeline_coverage":
                # Pipeline returned results for wrong locations; clear incompatible state
                # and trigger re-execution with the correct constraints
                if state.coverage_state:
                    requested_set = {r.lower() for r in state.coverage_state.requested_locations}
                    state.coverage_state.matched_locations = [
                        m for m in state.coverage_state.matched_locations
                        if m.lower() in requested_set
                    ]
                codes.append(issue.issue_code)
                fields.append("coverage_state.matched_locations")
                impact = REPAIR_IMPACT_REEXECUTE_PIPELINE
                repaired_any = True

            elif issue.issue_code == "status_coverage_contradiction":
                # State-only repair: status will be recomputed by finalizer
                codes.append(issue.issue_code)
                fields.append("status")
                repaired_any = True

        if repaired_any:
            state.repair_attempts += 1
            state.last_repair_impact = impact
            state.repair_history.append(RepairRecord(
                attempt=state.repair_attempts,
                issue_codes=list(set(codes)),
                repair_strategy="deterministic_restore",
                repair_impact=impact,
                fields_changed=list(set(fields)),
                before_state_reference="N/A",
                after_state_reference="N/A",
                verification_result="pending",
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            ))

        return repaired_any
