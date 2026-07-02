from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


SENSITIVE_KEYWORDS = [
    "email", "phone", "mobile", "device", "maid", "aaid", "idfa", "adid",
    "client_id", "user_id", "customer_id", "person_id", "ip", "mac",
    "lat", "lon", "latitude", "longitude"
]

TIME_KEYWORDS = ["time", "timestamp", "date", "created", "updated", "seen", "visit"]
LOCATION_KEYWORDS = ["city", "state", "country", "zip", "postal", "area", "location"]
DEMOGRAPHIC_KEYWORDS = ["age", "gender", "income", "language"]
BEHAVIOR_KEYWORDS = ["interest", "category", "affinity", "intent", "visit", "dwell", "frequency"]


class SchemaProfilerAgent:
    """
    Inspects incoming data and identifies:
    - possible PII/sensitive fields
    - useful cohort fields
    - embedding fields
    - metric fields
    - data quality issues

    This agent is data-source agnostic.
    It can profile CSV now and Postgres later.
    """

    def profile_dataframe(self, df: pd.DataFrame, source_name: str = "unknown") -> Dict[str, Any]:
        row_count = len(df)
        column_profiles: List[Dict[str, Any]] = []

        for col in df.columns:
            series = df[col]
            col_profile = self._profile_column(col, series, row_count)
            column_profiles.append(col_profile)

        pii_columns = [c["name"] for c in column_profiles if c["is_sensitive"]]
        cohort_candidate_columns = [
            c["name"]
            for c in column_profiles
            if c["recommended_use"] in ["cohort_trait", "location_trait", "time_trait", "behavior_trait"]
        ]
        embedding_candidate_columns = [
            c["name"]
            for c in column_profiles
            if c["recommended_use"] in ["cohort_trait", "location_trait", "time_trait", "behavior_trait", "numeric_signal"]
        ]

        quality_warnings = self._build_quality_warnings(row_count, column_profiles)

        return {
            "agent": "schema_profiler_agent",
            "source_name": source_name,
            "row_count": row_count,
            "column_count": len(df.columns),
            "columns": column_profiles,
            "pii_columns": pii_columns,
            "cohort_candidate_columns": cohort_candidate_columns,
            "embedding_candidate_columns": embedding_candidate_columns,
            "quality_warnings": quality_warnings,
            "status": "profiled",
        }

    def _profile_column(self, col: str, series: pd.Series, row_count: int) -> Dict[str, Any]:
        lower = col.lower()
        non_null = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        dtype = str(series.dtype)
        missing_pct = round((missing_count / row_count) * 100, 2) if row_count else 0
        unique_pct = round((unique_count / row_count) * 100, 2) if row_count else 0

        is_sensitive = any(k in lower for k in SENSITIVE_KEYWORDS)
        semantic_role = self._semantic_role(lower, series)
        recommended_use = self._recommended_use(semantic_role, is_sensitive)

        sample_values = []
        try:
            values = series.dropna().astype(str).head(5).tolist()
            sample_values = ["MASKED" if is_sensitive else v[:80] for v in values]
        except Exception:
            sample_values = []

        return {
            "name": col,
            "dtype": dtype,
            "non_null": non_null,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "is_sensitive": is_sensitive,
            "semantic_role": semantic_role,
            "recommended_use": recommended_use,
            "sample_values": sample_values,
        }

    def _semantic_role(self, lower: str, series: pd.Series) -> str:
        if any(k in lower for k in SENSITIVE_KEYWORDS):
            return "sensitive_identifier"

        if any(k in lower for k in TIME_KEYWORDS):
            return "time_signal"

        if any(k in lower for k in LOCATION_KEYWORDS):
            return "location_signal"

        if any(k in lower for k in DEMOGRAPHIC_KEYWORDS):
            return "demographic_signal"

        if any(k in lower for k in BEHAVIOR_KEYWORDS):
            return "behavior_signal"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric_signal"

        return "categorical_signal"

    def _recommended_use(self, semantic_role: str, is_sensitive: bool) -> str:
        if is_sensitive:
            return "hash_or_remove"

        mapping = {
            "time_signal": "time_trait",
            "location_signal": "location_trait",
            "demographic_signal": "cohort_trait",
            "behavior_signal": "behavior_trait",
            "numeric_signal": "numeric_signal",
            "categorical_signal": "cohort_trait",
        }

        return mapping.get(semantic_role, "review")

    def _build_quality_warnings(self, row_count: int, columns: List[Dict[str, Any]]) -> List[str]:
        warnings = []

        if row_count < 1000:
            warnings.append("Row count is below 1000. Production k-anonymity may block cohorts.")

        if not any(c["recommended_use"] in ["behavior_trait", "cohort_trait"] for c in columns):
            warnings.append("No strong cohort trait columns detected.")

        if not any(c["is_sensitive"] for c in columns):
            warnings.append("No obvious PII columns detected. Confirm if upstream system already anonymized data.")

        high_missing = [c["name"] for c in columns if c["missing_pct"] > 50]
        if high_missing:
            warnings.append(f"Columns with more than 50% missing values: {high_missing}")

        return warnings
