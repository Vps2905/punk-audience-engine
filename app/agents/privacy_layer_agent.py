from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.audit_logger import AuditLogger
from app.core.privacy_budget_ledger import PrivacyBudgetLedger
from app.core.production_config import load_production_config
from app.core.schema_validator import SafeSchemaValidator


class PrivacyLayerAgent:
    """
    Production PrivacyLayerAgent.

    Responsibilities:
    - Accept raw/semi-raw audience observations.
    - Hash identifiers in memory for dedupe/counting.
    - Never export raw or hashed identifiers.
    - Never export raw lat/lng or observations.
    - Aggregate into safe cohorts.
    - Enforce k-anonymity.
    - Apply Laplace differential privacy noise to aggregate counts.
    - Write privacy report, lineage report, schema report, and safe feature table.
    """

    DEFAULT_GROUPING_COLUMNS = [
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
    ]

    SAFE_OUTPUT_COLUMNS = [
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
        "sessions",
        "total_maid_volume",
        "noisy_maid_volume",
        "total_observations",
        "privacy_status",
        "quality_score",
        "trait_text",
    ]

    IDENTIFIER_CANDIDATES = [
        "maid",
        "raw_maid",
        "device_id",
        "mac",
        "mac_address",
        "maid_hash",
        "mac_hash",
        "hashed_mac",
        "client_id",
        "user_id",
    ]

    COUNT_CANDIDATES = [
        "maid_count",
        "count",
        "total_maid_volume",
        "safe_maid_volume",
        "noisy_maid_volume",
    ]

    TIMESTAMP_CANDIDATES = [
        "created_at",
        "timestamp",
        "event_time",
        "seen_at",
        "last_seen",
        "first_seen",
    ]

    LOCATION_CANDIDATES = [
        "location_name",
        "city",
        "place_name",
        "area",
        "region",
    ]

    POI_CANDIDATES = [
        "primary_poi_type",
        "poi_type",
        "place_type",
        "category",
        "business_category",
    ]

    RAW_COORDINATE_COLUMNS = {
        "lat",
        "lng",
        "latitude",
        "longitude",
        "raw_lat",
        "raw_lng",
    }

    RAW_VALUE_BLOCKLIST = [
        "postgresql://",
        "BEGIN PRIVATE KEY",
        "api_key",
        "apikey",
        "password",
        "secret",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.config = load_production_config()
        self.validator = SafeSchemaValidator()
        self.audit_logger = AuditLogger(self.config.audit_log_path)
        self.ledger = PrivacyBudgetLedger(
            self.config.privacy_ledger_path,
            max_epsilon_per_run=self.config.max_epsilon_per_run,
        )

    def run(
        self,
        data: pd.DataFrame,
        output_dir: str | Path,
        grouping_columns: Optional[List[str]] = None,
        identifier_column: Optional[str] = None,
        count_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        location_column: Optional[str] = None,
        poi_column: Optional[str] = None,
        k_min: Optional[int] = None,
        epsilon: Optional[float] = None,
        run_id: Optional[str] = None,
        hash_secret: Optional[str] = None,
        persist_artifacts: bool = True,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)

        if persist_artifacts:
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        run_id = run_id or f"privacy_run_{uuid.uuid4().hex[:12]}"
        grouping_columns = grouping_columns or self.DEFAULT_GROUPING_COLUMNS
        k_min = int(k_min if k_min is not None else self.config.k_min)
        epsilon = float(epsilon if epsilon is not None else self.config.synthetic_epsilon)

        self._validate_runtime_config(k_min=k_min, epsilon=epsilon)
        self._validate_raw_dataframe(data)

        resolved = self._resolve_columns(
            data=data,
            identifier_column=identifier_column,
            count_column=count_column,
            timestamp_column=timestamp_column,
            location_column=location_column,
            poi_column=poi_column,
        )

        self.audit_logger.log(
            "privacy_layer_started",
            {
                "run_id": run_id,
                "input_rows": int(len(data)),
                "resolved_columns": resolved,
                "grouping_columns": grouping_columns,
                "k_min": k_min,
                "epsilon": epsilon,
            },
        )

        working = self._build_working_dataframe(
            data=data,
            resolved=resolved,
            hash_secret=hash_secret,
        )

        aggregated = self._aggregate(
            working_df=working,
            grouping_columns=grouping_columns,
        )

        safe_df, blocked_cohorts = self._apply_k_anonymity_and_dp(
            aggregated=aggregated,
            k_min=k_min,
            epsilon=epsilon,
        )

        safe_df = self._finalize_features(safe_df)
        self.validator.validate_safe_cohort_dataframe(safe_df, context="privacy_output")

        clean_feature_path = output_dir / "clean_feature_table.csv"
        privacy_report_path = output_dir / "privacy_report.json"
        lineage_report_path = output_dir / "lineage_report.json"
        schema_report_path = output_dir / "privacy_input_schema.json"

        if persist_artifacts:
            safe_df.to_csv(
                clean_feature_path,
                index=False,
            )

        budget_record = self.ledger.record_spend(
            run_id=run_id,
            module="privacy_layer",
            epsilon=epsilon,
            engine="laplace_dp_noise",
            metadata={
                "input_rows": int(len(data)),
                "output_safe_cohorts": int(len(safe_df)),
                "blocked_cohorts": int(blocked_cohorts),
                "k_min": k_min,
            },
        )

        privacy_report = {
            "module": "Hashing & Privacy Layer",
            "status": "completed",
            "run_id": run_id,
            "input_rows": int(len(data)),
            "output_safe_cohorts": int(len(safe_df)),
            "blocked_cohorts": int(blocked_cohorts),
            "k_anonymity": {
                "enabled": True,
                "k_min": k_min,
            },
            "differential_privacy": {
                "enabled": True,
                "noise_type": "laplace",
                "epsilon": epsilon,
                "privacy_budget_recorded": True,
                "privacy_budget_record": budget_record,
            },
            "hashing": {
                "enabled": resolved["identifier_column"] is not None,
                "hash_algorithm": "HMAC-SHA256",
                "raw_identifiers_exported": False,
                "hashed_identifiers_exported": False,
            },
            "export_safety": {
                "raw_maids_exported": False,
                "hashed_real_maids_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "aggregated_only": True,
            },
            "storage_backend": (
                "local_files"
                if persist_artifacts
                else "run_history_jsonb"
            ),
            "outputs": (
                {
                    "clean_feature_table": str(clean_feature_path),
                    "privacy_report": str(privacy_report_path),
                    "lineage_report": str(lineage_report_path),
                    "input_schema": str(schema_report_path),
                }
                if persist_artifacts
                else {
                    "clean_feature_table": (
                        "postgres://audience_run_history.final_summary"
                        f"?run_id={run_id}"
                        "&section=privacy"
                        "&artifact=clean_feature_table"
                    ),
                    "privacy_report": (
                        "postgres://audience_run_history.final_summary"
                        f"?run_id={run_id}"
                        "&section=privacy"
                        "&artifact=privacy_report"
                    ),
                    "lineage_report": (
                        "postgres://audience_run_history.final_summary"
                        f"?run_id={run_id}"
                        "&section=privacy"
                        "&artifact=lineage_report"
                    ),
                    "input_schema": (
                        "postgres://audience_run_history.final_summary"
                        f"?run_id={run_id}"
                        "&section=privacy"
                        "&artifact=input_schema"
                    ),
                }
            ),
            "records": {
                "safe_cohorts": self._json_safe(
                    safe_df.to_dict(orient="records")
                ),
            },
        }

        lineage_report = {
            "module": "Privacy Layer Lineage",
            "run_id": run_id,
            "source": "input_dataframe",
            "input_columns": list(map(str, data.columns)),
            "resolved_columns": resolved,
            "transformations": [
                "validate_raw_dataframe",
                "resolve_source_columns",
                "hash_identifier_in_memory_if_available",
                "derive_safe_location_trait",
                "derive_safe_poi_trait",
                "derive_daypart",
                "derive_lookback_bucket",
                "drop_raw_identifiers_and_coordinates",
                "aggregate_safe_traits",
                "apply_k_anonymity",
                "apply_laplace_dp_noise",
                "create_trait_text",
                "write_safe_feature_table",
            ],
            "output_columns": list(safe_df.columns),
            "privacy_controls": {
                "k_min": k_min,
                "epsilon": epsilon,
                "raw_identifiers_exported": False,
                "raw_coordinates_exported": False,
            },
        }

        schema_report = {
            "module": "Privacy Layer Input Schema",
            "run_id": run_id,
            "input_columns": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "resolved_columns": resolved,
            "raw_coordinate_columns_present": [
                col for col in data.columns if str(col).lower() in self.RAW_COORDINATE_COLUMNS
            ],
            "identifier_column_used_for_internal_hashing": resolved["identifier_column"],
            "output_is_aggregated_only": True,
        }

        privacy_report["records"]["lineage_report"] = self._json_safe(
            lineage_report
        )
        privacy_report["records"]["input_schema"] = self._json_safe(
            schema_report
        )

        if persist_artifacts:
            self._write_json(privacy_report_path, privacy_report)
            self._write_json(lineage_report_path, lineage_report)
            self._write_json(schema_report_path, schema_report)

        self.audit_logger.log(
            "privacy_layer_completed",
            {
                "run_id": run_id,
                "safe_cohorts": int(len(safe_df)),
                "blocked_cohorts": int(blocked_cohorts),
                "clean_feature_table": str(clean_feature_path),
            },
        )

        return privacy_report

    def _validate_runtime_config(self, *, k_min: int, epsilon: float) -> None:
        if k_min <= 0:
            raise ValueError("k_min must be greater than 0.")

        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

    def _validate_raw_dataframe(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("data cannot be empty.")

        for col in data.select_dtypes(include=["object"]).columns:
            values = data[col].dropna().astype(str).head(200)
            for value in values:
                lower = value.lower()
                for token in self.RAW_VALUE_BLOCKLIST:
                    if token.lower() in lower:
                        raise ValueError(f"Secret-like value detected in input column: {col}")

    def _resolve_columns(
        self,
        *,
        data: pd.DataFrame,
        identifier_column: Optional[str],
        count_column: Optional[str],
        timestamp_column: Optional[str],
        location_column: Optional[str],
        poi_column: Optional[str],
    ) -> Dict[str, Optional[str]]:
        columns = set(data.columns)

        resolved_identifier = identifier_column or self._first_existing(columns, self.IDENTIFIER_CANDIDATES)
        resolved_count = count_column or self._first_existing(columns, self.COUNT_CANDIDATES)
        resolved_timestamp = timestamp_column or self._first_existing(columns, self.TIMESTAMP_CANDIDATES)
        resolved_location = location_column or self._first_existing(columns, self.LOCATION_CANDIDATES)
        resolved_poi = poi_column or self._first_existing(columns, self.POI_CANDIDATES)

        if resolved_location is None:
            raise ValueError("No safe location column found. Provide location_column or pre-enrich location_name.")

        if resolved_poi is None:
            raise ValueError("No POI/category column found. Provide poi_column or primary_poi_type.")

        for name, value in {
            "identifier_column": resolved_identifier,
            "count_column": resolved_count,
            "timestamp_column": resolved_timestamp,
            "location_column": resolved_location,
            "poi_column": resolved_poi,
        }.items():
            if value is not None and value not in columns:
                raise ValueError(f"{name} does not exist in dataframe: {value}")

        return {
            "identifier_column": resolved_identifier,
            "count_column": resolved_count,
            "timestamp_column": resolved_timestamp,
            "location_column": resolved_location,
            "poi_column": resolved_poi,
        }

    def _first_existing(self, columns: set, candidates: List[str]) -> Optional[str]:
        lower_map = {str(col).lower(): col for col in columns}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        return None

    def _build_working_dataframe(
        self,
        *,
        data: pd.DataFrame,
        resolved: Dict[str, Optional[str]],
        hash_secret: Optional[str],
    ) -> pd.DataFrame:
        df = data.copy()

        location_col = resolved["location_column"]
        poi_col = resolved["poi_column"]
        count_col = resolved["count_column"]
        identifier_col = resolved["identifier_column"]
        timestamp_col = resolved["timestamp_column"]

        working = pd.DataFrame()

        working["location_name"] = self._clean_text_series(df[location_col], "unknown_location")
        working["primary_poi_type"] = self._clean_text_series(df[poi_col], "unknown_poi_type")

        if timestamp_col:
            parsed_ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
            working["created_day_part"] = parsed_ts.apply(self._daypart_from_timestamp)
            working["lookback_bucket"] = parsed_ts.apply(self._lookback_bucket_from_timestamp)
        else:
            working["created_day_part"] = "all"
            working["lookback_bucket"] = "all"

        if count_col:
            working["maid_weight"] = (
                pd.to_numeric(df[count_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .clip(lower=0)
            )
        else:
            working["maid_weight"] = 1

        if identifier_col:
            secret = hash_secret or os.getenv("HASH_SECRET", "local_dev_hash_secret")
            working["_hashed_identifier"] = (
                df[identifier_col]
                .astype(str)
                .fillna("unknown")
                .apply(lambda value: self._hmac_sha256(value, secret))
            )
        else:
            working["_hashed_identifier"] = [
                f"row_{i}" for i in range(len(working))
            ]

        working["observation_count"] = 1

        return working

    def _aggregate(self, *, working_df: pd.DataFrame, grouping_columns: List[str]) -> pd.DataFrame:
        missing = [col for col in grouping_columns if col not in working_df.columns]
        if missing:
            raise ValueError(f"Grouping columns missing from working dataframe: {missing}")

        aggregated = (
            working_df.groupby(grouping_columns, dropna=False)
            .agg(
                sessions=("_hashed_identifier", "nunique"),
                total_maid_volume=("maid_weight", "sum"),
                total_observations=("observation_count", "sum"),
            )
            .reset_index()
        )

        aggregated["total_maid_volume"] = aggregated["total_maid_volume"].round().astype(int)
        aggregated["sessions"] = aggregated["sessions"].astype(int)
        aggregated["total_observations"] = aggregated["total_observations"].astype(int)

        return aggregated

    def _apply_k_anonymity_and_dp(
        self,
        *,
        aggregated: pd.DataFrame,
        k_min: int,
        epsilon: float,
    ) -> tuple[pd.DataFrame, int]:
        passed = aggregated[aggregated["total_maid_volume"] >= k_min].copy()
        blocked_cohorts = len(aggregated) - len(passed)

        if passed.empty:
            raise ValueError(
                "No cohorts passed k-anonymity. Increase data volume, reduce grouping granularity, "
                "or lower k_min only if policy allows."
            )

        rng = np.random.default_rng(self.random_state)
        noise = rng.laplace(loc=0.0, scale=1.0 / epsilon, size=len(passed))

        passed["noisy_maid_volume"] = (
            passed["total_maid_volume"].astype(float) + noise
        ).clip(lower=k_min).round().astype(int)

        passed["privacy_status"] = "passed"

        return passed, blocked_cohorts

    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        max_volume = max(float(df["noisy_maid_volume"].max()), 1.0)
        max_sessions = max(float(df["sessions"].max()), 1.0)

        volume_score = df["noisy_maid_volume"] / max_volume
        session_score = df["sessions"] / max_sessions

        df["quality_score"] = ((volume_score * 0.75) + (session_score * 0.25)).clip(0, 1)

        df["trait_text"] = df.apply(
            lambda row: " | ".join(
                [
                    f"location {row.get('location_name', 'unknown_location')}",
                    f"poi {row.get('primary_poi_type', 'unknown_poi_type')}",
                    f"daypart {row.get('created_day_part', 'all')}",
                    f"lookback {row.get('lookback_bucket', 'all')}",
                    f"quality {round(float(row.get('quality_score', 0)), 3)}",
                    f"sessions {int(row.get('sessions', 0))}",
                ]
            ),
            axis=1,
        )

        safe_cols = [col for col in self.SAFE_OUTPUT_COLUMNS if col in df.columns]
        return df[safe_cols].reset_index(drop=True)

    def _clean_text_series(self, series: pd.Series, default: str) -> pd.Series:
        return (
            series.replace([np.nan, None], default)
            .astype(str)
            .str.strip()
            .replace(
                {
                    "": default,
                    "nan": default,
                    "NaN": default,
                    "None": default,
                    "none": default,
                    "NULL": default,
                    "null": default,
                    "0": default,
                    "0.0": default,
                }
            )
            .str.lower()
        )

    def _daypart_from_timestamp(self, value: Any) -> str:
        if pd.isna(value):
            return "all"

        hour = int(value.hour)

        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    def _lookback_bucket_from_timestamp(self, value: Any) -> str:
        if pd.isna(value):
            return "all"

        now = pd.Timestamp.utcnow()
        age_days = max((now - value).days, 0)

        if age_days <= 7:
            return "0_7d"
        if age_days <= 30:
            return "8_30d"
        if age_days <= 90:
            return "31_90d"
        return "90d_plus"

    def _hmac_sha256(self, value: str, secret: str) -> str:
        return hmac.new(
            key=secret.encode("utf-8"),
            msg=value.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.write_text(json.dumps(self._json_safe(data), indent=2, allow_nan=False))

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._json_safe(v) for v in value]

        if isinstance(value, tuple):
            return [self._json_safe(v) for v in value]

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        return value
