from __future__ import annotations

import re
from typing import Any, Dict, List


BLOCKED_SENSITIVE_POI = {
    "hospital",
    "clinic",
    "medical_center",
    "diagnostic_center",
    "mental_health",
    "rehab",
    "addiction_treatment",
    "fertility_clinic",
    "abortion_clinic",
    "religious_place",
    "church",
    "mosque",
    "temple",
    "synagogue",
    "school",
    "college",
    "university",
    "daycare",
    "political_office",
    "union",
    "court",
    "courthouse",
    "prison",
    "jail",
    "shelter",
    "domestic_violence_shelter",
    "immigration_office",
}

REVIEW_REQUIRED_POI = {
    "casino",
    "bar",
    "pub",
    "night_club",
    "nightclub",
    "liquor_store",
    "adult_entertainment",
    "cannabis_store",
    "pharmacy",
    "dentist",
    "veterinary_care",
    "bank",
    "atm",
    "insurance_agency",
}

LOW_RISK_POI = {
    "restaurant",
    "fast_food_restaurant",
    "shawarma_restaurant",
    "middle_eastern_restaurant",
    "lebanese_restaurant",
    "cafe",
    "coffee_shop",
    "shopping_mall",
    "clothing_store",
    "womens_clothing_store",
    "shoe_store",
    "book_store",
    "toy_store",
    "store",
    "pet_store",
    "gym",
    "yoga_studio",
    "coworking_space",
    "corporate_office",
    "gas_station",
    "car_wash",
    "hair_salon",
    "barber_shop",
}


class SensitivePOIPrivacyRiskService:
    def assess(
        self,
        *,
        prompt: str,
        selected_cohorts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rows = [row for row in selected_cohorts if isinstance(row, dict)]
        assessed_rows = []

        max_risk_score = 0.0
        blocked_count = 0
        review_count = 0

        for row in rows:
            poi = self._norm(row.get("primary_poi_type"))
            risk = self._classify_poi(poi)

            if risk["decision"] == "block_export":
                blocked_count += 1
            elif risk["decision"] == "review_required":
                review_count += 1

            max_risk_score = max(max_risk_score, risk["risk_score"])

            assessed_rows.append(
                {
                    "audience_name": self._display(row.get("audience_name")) or self._build_name(row),
                    "location_name": self._display(row.get("location_name")),
                    "primary_poi_type": poi,
                    "created_day_part": self._display(row.get("created_day_part")),
                    "risk_level": risk["risk_level"],
                    "risk_score": risk["risk_score"],
                    "decision": risk["decision"],
                    "reason_codes": risk["reason_codes"],
                    "human_reason": risk["human_reason"],
                }
            )

        prompt_risk = self._assess_prompt_text(prompt)

        if prompt_risk["decision"] == "block_export" or blocked_count > 0:
            overall_decision = "block_export"
        elif review_count > 0 or prompt_risk["decision"] == "review_required":
            overall_decision = "review_required"
        else:
            overall_decision = "allow_approval_gated_export"

        return {
            "pipeline": "sensitive_poi_privacy_risk",
            "overall_decision": overall_decision,
            "safe_to_export": False,
            "approval_required": True,
            "downstream_export_enabled": False,
            "selected_count": len(rows),
            "blocked_sensitive_count": blocked_count,
            "review_required_count": review_count,
            "max_risk_score": round(max(max_risk_score, prompt_risk["risk_score"]), 3),
            "prompt_risk": prompt_risk,
            "assessed_audiences": assessed_rows,
        }

    def _classify_poi(self, poi: str) -> Dict[str, Any]:
        if not poi:
            return self._risk("unknown_review", 0.55, "review_required", ["unknown_poi_type"], "POI type is missing or unknown.")

        if self._matches_any(poi, BLOCKED_SENSITIVE_POI):
            return self._risk("blocked_sensitive", 1.0, "block_export", ["sensitive_poi_blocked"], "Sensitive POI category must not be exported.")

        if self._matches_any(poi, REVIEW_REQUIRED_POI):
            return self._risk("regulated_or_sensitive_review", 0.75, "review_required", ["regulated_or_sensitive_poi_review_required"], "Regulated or sensitive-adjacent POI requires human review.")

        if self._matches_any(poi, LOW_RISK_POI):
            return self._risk("low", 0.2, "allow_approval_gated_export", ["low_risk_business_poi"], "Normal commercial POI category.")

        return self._risk("medium_unknown", 0.5, "review_required", ["unclassified_poi_review_required"], "Unclassified POI requires human review.")

    def _assess_prompt_text(self, prompt: str) -> Dict[str, Any]:
        text = self._norm(prompt)

        blocked_terms = {
            "hospital",
            "hospitals",
            "clinic",
            "clinics",
            "medical center",
            "medical centers",
            "diagnostic center",
            "diagnostic centers",
            "healthcare",
            "health care",
            "health checkup",
            "health_checkup",
            "medical condition",
            "mental health",
            "addiction",
            "rehab",
            "religion",
            "religious",
            "political",
            "pregnancy",
            "abortion",
            "domestic violence",
            "immigration status",
        }

        review_terms = {
            "casino",
            "gambling",
            "alcohol",
            "nightlife",
            "pharmacy",
        }

        if any(self._norm(term) in text for term in blocked_terms):
            return self._risk("blocked_sensitive_prompt", 1.0, "block_export", ["prompt_mentions_sensitive_personal_attribute"], "Prompt appears to target sensitive personal attributes.")

        if any(self._norm(term) in text for term in review_terms):
            return self._risk("prompt_review_required", 0.7, "review_required", ["prompt_mentions_regulated_or_sensitive_category"], "Prompt references a regulated or sensitive-adjacent category.")

        return self._risk("low", 0.1, "allow_approval_gated_export", ["prompt_low_risk"], "Prompt does not appear to target sensitive personal attributes.")

    def _risk(self, level: str, score: float, decision: str, codes: list[str], reason: str) -> Dict[str, Any]:
        return {
            "risk_level": level,
            "risk_score": score,
            "decision": decision,
            "reason_codes": codes,
            "human_reason": reason,
        }

    def _matches_any(self, poi: str, values: set[str]) -> bool:
        poi_norm = self._norm(poi)
        for value in values:
            value_norm = self._norm(value)
            if poi_norm == value_norm or value_norm in poi_norm or poi_norm in value_norm:
                return True
        return False

    def _norm(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    def _display(self, value: Any) -> str:
        return str(value or "").strip()

    def _build_name(self, row: Dict[str, Any]) -> str:
        parts = [
            self._display(row.get("primary_poi_type")),
            self._display(row.get("created_day_part")),
            self._display(row.get("location_name")),
        ]
        return " - ".join([p for p in parts if p])
