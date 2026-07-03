from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


class SafeSchemaValidator:
    """
    Fail-closed validator for privacy-safe module boundaries.
    """

    SAFE_AGGREGATE_COLUMNS = {
        "total_maid_volume",
        "noisy_maid_volume",
        "safe_maid_volume",
        "total_observations",
        "observation_count",
        "observations_count",
        "safe_observation_count",
    }

    DANGEROUS_EXACT_COLUMNS = {
        "maid",
        "maids",
        "raw_maid",
        "raw_maids",
        "device_id",
        "device_ids",
        "email",
        "emails",
        "phone",
        "phones",
        "lat",
        "lng",
        "latitude",
        "longitude",
        "observation",
        "observations",
        "raw_observations",
    }

    DANGEROUS_COLUMN_TOKENS = [
        "raw_maid",
        "device_id",
        "email",
        "phone",
        "latitude",
        "longitude",
        "observations",
        "raw_observation",
    ]

    SECRET_VALUE_PATTERNS = [
        re.compile(r"postgresql://", re.IGNORECASE),
        re.compile(r"BEGIN\s+PRIVATE\s+KEY", re.IGNORECASE),
        re.compile(r"api[_-]?key", re.IGNORECASE),
        re.compile(r"secret", re.IGNORECASE),
        re.compile(r"password", re.IGNORECASE),
    ]

    def validate_safe_cohort_dataframe(self, df: pd.DataFrame, context: str = "dataframe") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{context} must be a pandas DataFrame.")

        if df.empty:
            raise ValueError(f"{context} is empty.")

        self.validate_no_blocked_columns(df.columns, context=context)
        self.validate_no_secret_values(df, context=context)

    def validate_no_blocked_columns(self, columns: Iterable[str], context: str = "dataframe") -> None:
        for col in columns:
            lower = str(col).strip().lower()

            if lower in self.SAFE_AGGREGATE_COLUMNS:
                continue

            if lower in self.DANGEROUS_EXACT_COLUMNS:
                raise ValueError(f"{context} contains blocked sensitive column: {col}")

            for token in self.DANGEROUS_COLUMN_TOKENS:
                if token in lower:
                    raise ValueError(f"{context} contains blocked sensitive column pattern: {col}")

            # Block raw lat/lng variants but avoid blocking safe names like location_name.
            if lower in {"lat", "lng"} or lower.endswith("_lat") or lower.endswith("_lng"):
                raise ValueError(f"{context} contains blocked raw coordinate column: {col}")

    def validate_no_secret_values(self, df: pd.DataFrame, context: str = "dataframe") -> None:
        object_cols = df.select_dtypes(include=["object"]).columns

        for col in object_cols:
            sample = df[col].dropna().astype(str).head(200)
            for value in sample:
                for pattern in self.SECRET_VALUE_PATTERNS:
                    if pattern.search(value):
                        raise ValueError(f"{context} contains secret-like value in column: {col}")
