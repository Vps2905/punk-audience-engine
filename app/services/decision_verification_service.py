from typing import List, Dict, Any
from app.models.autonomous_decision_state import (
    AutonomousDecisionState,
    VerificationIssue,
    ConstraintLedger,
    SemanticDecision,
)
from app.utils.location_matcher import location_matches_request


class DecisionVerificationService:
    def verify(self, state: AutonomousDecisionState) -> List[VerificationIssue]:
        issues = []
        ledger = state.constraint_ledger
        semantic = state.semantic_decision

        if not ledger or not semantic:
            issues.append(VerificationIssue(
                issue_code="missing_state",
                severity="critical",
                description="Missing ledger or semantic decision state.",
                expected="Both states present",
                observed="Missing",
                evidence="N/A",
                repairable=True,
                recommended_action="Re-run interpretation"
            ))
            return issues

        # 1. Explicit location dropped
        for loc_constraint in ledger.locations:
            loc_req = loc_constraint.normalized_value
            # Did the semantic decision keep it?
            if not any(location_matches_request(loc_req, sm) or location_matches_request(sm, loc_req) for sm in semantic.locations):
                issues.append(VerificationIssue(
                    issue_code="explicit_location_dropped",
                    severity="critical",
                    description=f"Explicit location '{loc_constraint.raw_text}' was dropped from semantic decision.",
                    expected=f"Semantic decision includes '{loc_constraint.raw_text}'",
                    observed=f"Semantic decision locations: {semantic.locations}",
                    evidence="ConstraintLedger vs SemanticDecision",
                    repairable=True,
                    recommended_action="Restore explicit location to semantic locations"
                ))

        # 2. Explicit category dropped
        for cat_constraint in ledger.categories:
            cat_req = cat_constraint.normalized_value
            # Very basic inclusion check
            if not any(cat_req in sm.lower() or sm.lower() in cat_req for sm in semantic.categories):
                issues.append(VerificationIssue(
                    issue_code="explicit_category_dropped",
                    severity="critical",
                    description=f"Explicit category '{cat_constraint.raw_text}' was dropped from semantic decision.",
                    expected=f"Semantic decision includes '{cat_constraint.raw_text}'",
                    observed=f"Semantic decision categories: {semantic.categories}",
                    evidence="ConstraintLedger vs SemanticDecision",
                    repairable=True,
                    recommended_action="Restore explicit category"
                ))

        # 3. Explicit daypart dropped
        for dp_constraint in ledger.dayparts:
            dp_req = dp_constraint.normalized_value
            dropped_in_semantic = not any(dp_req in sm.lower() for sm in semantic.dayparts)
            
            tool_results_contradict = False
            if state.tool_results:
                v2gs = state.tool_results.get("v2_guided_selection", {})
                if v2gs:
                    # check if requested_dayparts has it
                    v2_req_dp = v2gs.get("requested_dayparts", [])
                    if not any(dp_req in str(d).lower() for d in v2_req_dp):
                        tool_results_contradict = True

            if dropped_in_semantic or tool_results_contradict:
                issues.append(VerificationIssue(
                    issue_code="explicit_daypart_dropped",
                    severity="critical",
                    description=f"Explicit daypart '{dp_constraint.raw_text}' was dropped.",
                    expected=f"Semantic decision and tool results include '{dp_constraint.raw_text}'",
                    observed=f"Semantic dayparts: {semantic.dayparts}, Tool dayparts: {state.tool_results.get('v2_guided_selection', {}).get('requested_dayparts', []) if state.tool_results else []}",
                    evidence="ConstraintLedger vs SemanticDecision/ToolResults",
                    repairable=True,
                    recommended_action="Restore explicit daypart"
                ))

        # Check coverage vs quality
        cov = state.coverage_state
        qual = state.quality_state
        if cov and qual:
            for loc in qual.evaluated_locations:
                if loc in cov.missing_locations:
                    issues.append(VerificationIssue(
                        issue_code="unsupported_location_evaluated",
                        severity="critical",
                        description=f"Location '{loc}' had no coverage but was evaluated for quality.",
                        expected="Only locations with exact candidates are evaluated",
                        observed=f"'{loc}' evaluated for quality",
                        evidence="CoverageState vs QualityState",
                        repairable=True,
                        recommended_action="Remove missing locations from quality evaluation"
                    ))

        # Coverage-ledger consistency: if requested locations are missing from pipeline results
        # and matched locations contain locations NOT in the request, the pipeline returned
        # incompatible results and must re-execute with corrected constraints.
        if cov and cov.missing_locations and ledger.locations:
            # Check if any matched location is not in requested locations
            requested_set = {r.lower() for r in cov.requested_locations}
            incompatible_matches = [
                m for m in cov.matched_locations
                if m.lower() not in requested_set
            ]
            if incompatible_matches:
                issues.append(VerificationIssue(
                    issue_code="incompatible_pipeline_coverage",
                    severity="critical",
                    description=f"Pipeline returned coverage for {incompatible_matches} but requested locations {list(requested_set)} are missing.",
                    expected=f"Coverage for requested locations: {list(requested_set)}",
                    observed=f"Coverage for: {incompatible_matches}, missing: {cov.missing_locations}",
                    evidence="CoverageState vs ConstraintLedger",
                    repairable=True,
                    recommended_action="Re-execute pipeline with corrected constraints"
                ))

        # Downstream & Approval & Stale
        if state.export_state:
            downstream = state.export_state.get("downstream_export_enabled")
            approval = state.approval_state.get("approval_status") if state.approval_state else None
            if downstream and approval != "approved":
                issues.append(VerificationIssue(
                    issue_code="unapproved_downstream_export",
                    severity="critical",
                    description="Downstream export enabled without approval.",
                    expected="approval_status == 'approved'",
                    observed=f"approval_status == {approval}",
                    evidence="ExportState",
                    repairable=True,
                    recommended_action="Disable downstream export"
                ))
            if downstream and state.freshness_state and state.freshness_state.get("block_export"):
                issues.append(VerificationIssue(
                    issue_code="stale_source_with_export_enabled",
                    severity="critical",
                    description="Export is enabled but source is stale.",
                    expected="downstream_export_enabled == False",
                    observed="downstream_export_enabled == True",
                    evidence="FreshnessState vs ExportState",
                    repairable=True,
                    recommended_action="Disable downstream export"
                ))
            if downstream and state.status and state.status.startswith("blocked"):
                issues.append(VerificationIssue(
                    issue_code="blocked_result_described_as_approved",
                    severity="critical",
                    description="Result is blocked but export is enabled.",
                    expected="downstream_export_enabled == False",
                    observed="downstream_export_enabled == True",
                    evidence="State status vs ExportState",
                    repairable=True,
                    recommended_action="Disable downstream export"
                ))

        # Privacy leakage
        if state.privacy_state and state.privacy_state.get("individual_fields_exposed"):
            issues.append(VerificationIssue(
                issue_code="individual_field_leakage",
                severity="critical",
                description="Individual fields exposed in privacy state.",
                expected="No individual fields exposed",
                observed="individual_fields_exposed == True",
                evidence="PrivacyState",
                repairable=True,
                recommended_action="Remove individual fields from payload"
            ))

        # Tool failure represented as success
        if state.tool_results and state.tool_results.get("status") == "failed" and state.status == "completed":
            issues.append(VerificationIssue(
                issue_code="failed_tool_represented_as_success",
                severity="critical",
                description="Tool failed but state is completed.",
                expected="State status matches tool failure",
                observed="State status is completed",
                evidence="ToolResults vs State status",
                repairable=False,
                recommended_action="Update status to failed"
            ))

        # Skipped mandatory plan capabilities
        if state.execution_plan:
            for step in state.execution_plan.mandatory_steps:
                if step not in state.completed_steps:
                    issues.append(VerificationIssue(
                        issue_code="skipped_mandatory_capability",
                        severity="critical",
                        description=f"Mandatory capability {step} was skipped.",
                        expected=f"{step} in completed_steps",
                        observed=f"{step} missing",
                        evidence="ExecutionPlan vs CompletedSteps",
                        repairable=False,
                        recommended_action="Fail execution or retry step"
                    ))

        # Excluded cohorts in lookalikes
        if state.tool_results and state.tool_results.get("lookalikes"):
            # Check if any lookalike uses an excluded cohort
            excluded = [c for c in ledger.categories if getattr(c, "exclusion", False)]
            if excluded:
                # If a category is excluded, make sure lookalikes don't include it
                for ll in state.tool_results.get("lookalikes", []):
                    if any(ex.normalized_value in ll.get("categories", []) for ex in excluded):
                        issues.append(VerificationIssue(
                            issue_code="excluded_cohort_in_lookalike",
                            severity="critical",
                            description="Lookalike contains excluded cohort.",
                            expected="Lookalikes do not contain excluded cohorts",
                            observed="Excluded cohort present in lookalike",
                            evidence="ToolResults vs ConstraintLedger",
                            repairable=True,
                            recommended_action="Remove invalid lookalikes"
                        ))

        # Candidate count consistency
        if state.coverage_state and state.export_state:
            # If we say we exported X cohorts, but coverage found Y candidates, they should be logically consistent
            # E.g. exported > 0 implies candidates > 0
            prepared = state.export_state.get("prepared_cohorts") or state.export_state.get("exported_cohorts", 0)
            candidates = state.coverage_state.exact_candidate_count
            if prepared > 0 and candidates == 0:
                issues.append(VerificationIssue(
                    issue_code="candidate_artifact_count_mismatch",
                    severity="critical",
                    description="Exported artifacts exist but candidate count is zero.",
                    expected="candidates > 0 if exported > 0",
                    observed=f"exported={prepared}, candidates={candidates}",
                    evidence="CoverageState vs ExportState",
                    repairable=False,
                    recommended_action="Block export"
                ))

        # Unresolved mandatory constraint
        if ledger.unresolved_constraints:
            for unres in ledger.unresolved_constraints:
                issues.append(VerificationIssue(
                    issue_code="unresolved_mandatory_constraint",
                    severity="critical",
                    description=f"Constraint '{unres.raw_text}' is unresolved.",
                    expected="Resolved or blocked",
                    observed="Unresolved",
                    evidence="ConstraintLedger",
                    repairable=False,
                    recommended_action="Block or clarify"
                ))

        # Status-coverage contradiction checks
        if state.coverage_state:
            cov = state.coverage_state
            # Coverage is complete but status says no exact match
            if cov.coverage_status == "complete" and state.status == "blocked_no_safe_exact_match":
                issues.append(VerificationIssue(
                    issue_code="status_coverage_contradiction",
                    severity="critical",
                    description="Coverage is complete but status says no safe exact match.",
                    expected="Status consistent with coverage_status=complete",
                    observed=f"status={state.status}, coverage_status={cov.coverage_status}",
                    evidence="CoverageState vs Status",
                    repairable=True,
                    recommended_action="Recompute status from canonical fields"
                ))
            # Candidates exist but status says no exact match
            if cov.exact_candidate_count > 0 and state.status == "blocked_no_safe_exact_match":
                issues.append(VerificationIssue(
                    issue_code="status_coverage_contradiction",
                    severity="critical",
                    description=f"Candidates exist ({cov.exact_candidate_count}) but status says no safe exact match.",
                    expected="Status consistent with candidate count > 0",
                    observed=f"status={state.status}, candidates={cov.exact_candidate_count}",
                    evidence="CoverageState vs Status",
                    repairable=True,
                    recommended_action="Recompute status from canonical fields"
                ))
            # Approval status and final status contradict
            if state.approval_state:
                approval = state.approval_state.get("approval_status")
                if approval and approval.startswith("blocked") and state.status == "completed":
                    issues.append(VerificationIssue(
                        issue_code="status_approval_contradiction",
                        severity="critical",
                        description=f"Approval is '{approval}' but status is 'completed'.",
                        expected="Status consistent with approval_status",
                        observed=f"approval={approval}, status={state.status}",
                        evidence="ApprovalState vs Status",
                        repairable=True,
                        recommended_action="Recompute status from approval fields"
                    ))

        return issues
