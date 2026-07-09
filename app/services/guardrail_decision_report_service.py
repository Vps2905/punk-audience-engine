from __future__ import annotations

from typing import Any, Dict


class GuardrailDecisionReportService:
    def build_report(
        self,
        coverage_report: Dict[str, Any],
        retrieval_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected = retrieval_report.get("selected_candidates") or []
        ranked = retrieval_report.get("ranked_candidates") or []
        coverage_decision = coverage_report.get("coverage_decision")

        unsafe_blocked = [
            {
                "audience_name": c.get("audience_name"),
                "location_name": c.get("location_name"),
                "primary_poi_type": c.get("primary_poi_type"),
                "created_day_part": c.get("created_day_part"),
                "decision": c.get("decision"),
                "reason_codes": c.get("reason_codes"),
            }
            for c in ranked
            if str(c.get("decision", "")).startswith("blocked")
        ]

        if coverage_decision != "exact_safe_combo_available" and not selected:
            decision = "blocked"
            reason_code = coverage_decision
            safe_to_export = False
        elif selected:
            decision = "review_required"
            reason_code = "safe_candidates_available_approval_required"
            safe_to_export = False
        else:
            decision = "blocked"
            reason_code = "no_candidates_selected"
            safe_to_export = False

        return {
            "pipeline": "guardrail_decision_report",
            "decision": decision,
            "reason_code": reason_code,
            "safe_to_export": safe_to_export,
            "approval_required": True,
            "downstream_export_enabled": False,
            "coverage_decision": coverage_decision,
            "selected_count": len(selected),
            "blocked_candidate_count": len(unsafe_blocked),
            "unsafe_fallbacks_blocked": unsafe_blocked[:25],
            "recommended_next_action": coverage_report.get("recommended_next_action"),
        }
