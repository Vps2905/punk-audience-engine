import os
import math
import pandas as pd
from typing import Any, Dict, List, Tuple


class AudienceQualityPolicyService:
    """
    Enforces semantic audience quality intent dynamically.
    Replaces static keyword checks with normalized threshold strategies.
    """

    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
        # Configuration
        self.absolute_floor = self._clamp_float("AUDIENCE_HIGH_QUALITY_ABSOLUTE_FLOOR", 0.60, 0.0, 1.0)
        self.percentile = self._clamp_float("AUDIENCE_HIGH_QUALITY_PERCENTILE", 0.70, 0.0, 1.0)
        self.max_gap_from_best = self._clamp_float("AUDIENCE_HIGH_QUALITY_MAX_GAP_FROM_BEST", 0.15, 0.0, 1.0)

        min_c = self._safe_env_int("AUDIENCE_HIGH_QUALITY_MIN_CANDIDATES", 1)
        if min_c < 1:
            self.logger.warning(f"AUDIENCE_HIGH_QUALITY_MIN_CANDIDATES out of bounds: {min_c}. Clamping to 1.")
            min_c = 1
        self.min_candidates = min_c

    def _clamp_float(self, key: str, default: float, min_val: float, max_val: float) -> float:
        val = self._safe_env_float(key, default)
        if val < min_val or val > max_val:
            self.logger.warning(f"{key} out of bounds [{min_val}, {max_val}]: {val}. Clamping.")
            return max(min_val, min(max_val, val))
        return val

    def _safe_env_float(self, key: str, default: float) -> float:
        val = os.getenv(key)
        if not val:
            return default
        try:
            val_float = float(val)
            if math.isnan(val_float) or math.isinf(val_float):
                return default
            return val_float
        except ValueError:
            return default

    def _safe_env_int(self, key: str, default: int) -> int:
        val = os.getenv(key)
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def normalize_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes quality scores across candidates.
        Prefers management_quality_score, falls back to quality_score.
        Converts 0-100 scale to 0-1 scale safely.
        Adds an internal column '_effective_quality_score'.
        """
        if df.empty:
            df["_effective_quality_score"] = 0.0
            df["_quality_source"] = "none"
            return df

        scores = []
        sources = []
        for _, row in df.iterrows():
            score = 0.0
            source = "none"

            mqs = row.get("management_quality_score")
            qs = row.get("quality_score")

            if mqs is not None and not pd.isna(mqs):
                val = mqs
                source = "management_quality_score"
            else:
                val = qs
                source = "quality_score"

            if val is not None and not pd.isna(val):
                try:
                    v = float(val)
                    if not math.isnan(v) and not math.isinf(v):
                        # Detect 0-100 scale (if any value > 1, assume 0-100)
                        if v > 1.0:
                            v = v / 100.0
                        v = max(0.0, min(1.0, v))
                        score = v
                except (ValueError, TypeError):
                    source = "invalid"

            scores.append(score)
            sources.append(source)

        df["_effective_quality_score"] = scores
        df["_quality_source"] = sources
        return df

    def apply_policy(
        self,
        candidates: pd.DataFrame,
        quality_intent: str,
        requested_locations: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies quality filtering based on the resolved intent.
        quality_intent in ['high', 'balanced', 'broad']
        Returns (filtered_candidates, policy_report)
        """
        if not quality_intent or quality_intent not in ["high", "balanced", "broad"]:
            quality_intent = "unknown"

        df = candidates.copy()

        # Ensure normalization
        if "_effective_quality_score" not in df.columns:
            df = self.normalize_quality_scores(df)

        before_count = len(df)

        from app.utils.location_matcher import location_matches_request, norm_text

        if quality_intent in ["balanced", "broad", "unknown"] or df.empty:
            locations_meeting = set()
            locations_missing = set()

            if not requested_locations:
                requested_locations = ["_global_"]
                df["_temp_loc"] = "_global_"
                loc_col = "_temp_loc"
            else:
                loc_col = "location_name"

            for req_loc in requested_locations:
                req_loc_norm = norm_text(req_loc)
                if req_loc_norm == "_global_":
                    loc_df = df
                else:
                    if loc_col in df.columns:
                        loc_df = df[df[loc_col].apply(lambda x: location_matches_request(x, req_loc))]
                    else:
                        loc_df = pd.DataFrame()
                if loc_df.empty:
                    locations_missing.add(req_loc)
                else:
                    locations_meeting.add(req_loc)

            if "_temp_loc" in df.columns:
                df.drop(columns=["_temp_loc"], inplace=True)

            if quality_intent == "unknown":
                status = "not_requested"
            else:
                if not locations_meeting and req_loc_norm != "_global_":
                    status = "unmet"
                elif locations_missing and req_loc_norm != "_global_":
                    status = "partial"
                else:
                    status = "satisfied"

            report = self._build_report(
                df=df,
                quality_intent=quality_intent,
                status=status,
                before_count=before_count,
                after_count=before_count,
                excluded_count=0,
                thresholds={},
                loc_meeting=sorted(list(locations_meeting - {"_global_"})),
                loc_missing=sorted(list(locations_missing - {"_global_"})),
            )
            return df, report

        # HIGH QUALITY POLICY
        selected_indices = []
        thresholds_by_location = {}
        locations_meeting = set()
        locations_missing = set()

        # Determine which column we used primarily (for reporting)
        q_field = "_effective_quality_score"

        from app.utils.location_matcher import location_matches_request, norm_text

        if not requested_locations:
            # Global fallback if no location specified
            requested_locations = ["_global_"]
            df["_temp_loc"] = "_global_"
            loc_col = "_temp_loc"
        else:
            loc_col = "location_name"

        for req_loc in requested_locations:
            req_loc_norm = norm_text(req_loc)

            if req_loc_norm == "_global_":
                loc_df = df
            else:
                if loc_col in df.columns:
                    loc_df = df[df[loc_col].apply(lambda x: location_matches_request(x, req_loc))]
                else:
                    loc_df = pd.DataFrame()

            if loc_df.empty:
                locations_missing.add(req_loc)
                continue

            scores = loc_df["_effective_quality_score"].tolist()
            if not scores:
                locations_missing.add(req_loc)
                continue

            max_score = max(scores)

            # If highest score is below absolute floor, no candidates meet the policy
            if max_score < self.absolute_floor:
                thresholds_by_location[req_loc] = self.absolute_floor
                locations_missing.add(req_loc)
                continue

            # Dynamic threshold (intersection of strictest rules)
            percentile_threshold = pd.Series(scores).quantile(self.percentile)
            gap_threshold = max_score - self.max_gap_from_best

            # For strict high-quality behaviour, prefer the stricter combined threshold
            threshold = max(percentile_threshold, gap_threshold)
            threshold = max(threshold, self.absolute_floor)

            thresholds_by_location[req_loc] = threshold

            # Select candidates meeting the threshold
            valid_loc_df = loc_df[loc_df["_effective_quality_score"] >= threshold]

            if valid_loc_df.empty:
                locations_missing.add(req_loc)
            else:
                locations_meeting.add(req_loc)
                selected_indices.extend(valid_loc_df.index.tolist())

        # Clean up global temp col if used
        if "_temp_loc" in df.columns:
            df.drop(columns=["_temp_loc"], inplace=True)

        # Stable deduplication
        unique_indices = list(dict.fromkeys(selected_indices))
        final_selected = df.loc[unique_indices].copy()

        # Policy status
        if req_loc_norm != "_global_":
            if not locations_meeting:
                status = "unmet"
            elif locations_missing:
                status = "partial"
            else:
                status = "satisfied"
        else:
            status = "satisfied" if not final_selected.empty else "unmet"

        report = self._build_report(
            df=df,
            quality_intent=quality_intent,
            status=status,
            before_count=before_count,
            after_count=len(final_selected),
            excluded_count=before_count - len(final_selected),
            thresholds=thresholds_by_location,
            loc_meeting=sorted(list(locations_meeting - {"_global_"})),
            loc_missing=sorted(list(locations_missing - {"_global_"})),
            q_field=q_field,
        )

        return final_selected, report

    def _build_report(
        self,
        df: pd.DataFrame,
        quality_intent: str,
        status: str,
        before_count: int,
        after_count: int,
        excluded_count: int,
        thresholds: Dict[str, float],
        loc_meeting: List[str],
        loc_missing: List[str],
        q_field: str = "quality_score",
    ) -> Dict[str, Any]:
        source_counts = {}
        if not df.empty and "_quality_source" in df.columns:
            source_counts = df["_quality_source"].value_counts().to_dict()

        return {
            "quality_intent": quality_intent,
            "quality_policy_status": status,
            "quality_score_field": q_field,
            "quality_score_sources": sorted(list(source_counts.keys())),
            "quality_score_source_counts": source_counts,
            "normalized_score_scale": "0_to_1",
            "quality_threshold_strategy": "adaptive_per_location_intersection" if quality_intent == "high" else "none",
            "quality_thresholds_by_location": thresholds,
            "quality_candidates_before": before_count,
            "quality_candidates_after": after_count,
            "quality_excluded_count": excluded_count,
            "locations_meeting_quality": loc_meeting,
            "locations_missing_quality": loc_missing,
        }
