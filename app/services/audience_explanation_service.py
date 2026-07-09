from __future__ import annotations

from typing import Any, Dict, List


class AudienceExplanationService:
    def explain_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        location = candidate.get("location_name")
        poi = candidate.get("primary_poi_type")
        daypart = candidate.get("created_day_part")
        decision = candidate.get("decision")
        scores = candidate.get("score_breakdown") or {}

        reasons: List[str] = []

        if scores.get("location_score", 0) >= 0.62:
            reasons.append(f"Location matched requested market: {location}.")
        else:
            reasons.append("Location did not strongly match the requested market.")

        if scores.get("category_score", 0) >= 0.62:
            reasons.append(f"POI/category matched requested intent: {poi}.")
        else:
            reasons.append("POI/category did not strongly match the requested intent.")

        if scores.get("daypart_score", 0) >= 0.80:
            reasons.append(f"Daypart matched requested time window: {daypart}.")
        else:
            reasons.append("Daypart did not strongly match the requested time window.")

        if scores.get("privacy_score", 0) >= 1:
            reasons.append("Privacy status passed for safe cohort usage.")
        else:
            reasons.append("Privacy status did not pass safe cohort requirements.")

        if decision in {"selected_for_review", "strong_candidate"}:
            reasons.append("Audience is eligible for approval review, not direct downstream export.")
        elif str(decision or "").startswith("blocked"):
            reasons.append(f"Candidate blocked by decision engine: {decision}.")
        else:
            reasons.append("Candidate was not selected because score was below threshold.")

        return {
            "audience_name": candidate.get("audience_name"),
            "decision": decision,
            "final_score": candidate.get("final_score"),
            "score_breakdown": scores,
            "selection_reason": reasons,
        }

    def explain_report(self, retrieval_report: Dict[str, Any]) -> Dict[str, Any]:
        selected = retrieval_report.get("selected_candidates") or []
        ranked = retrieval_report.get("ranked_candidates") or []

        return {
            "pipeline": "audience_explanation",
            "selected_explanations": [self.explain_candidate(c) for c in selected],
            "top_ranked_explanations": [self.explain_candidate(c) for c in ranked[:10]],
        }
