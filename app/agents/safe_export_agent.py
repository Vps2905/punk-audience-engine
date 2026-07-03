from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.core.audit_logger import AuditLogger
from app.core.production_config import load_production_config
from app.core.schema_validator import SafeSchemaValidator


class SafeExportAgent:
    """
    Production SafeExportAgent.

    Responsibilities:
    - Accept only CohortManagementAgent safe outputs.
    - Create approval-gated safe export package.
    - Export only aggregated cohort-level data.
    - Never export raw MAIDs, hashed IDs, raw observations, lat/lng, email, phone,
      device IDs, or individual user rows.
    """

    SAFE_COHORT_COLUMNS = [
        "cohort_index",
        "cluster_id",
        "cluster_size",
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
        "sessions",
        "total_maid_volume",
        "noisy_maid_volume",
        "quality_score",
        "cluster_coherence_score",
        "management_quality_score",
        "export_ready",
        "recommendation_reason",
        "trait_text",
    ]

    SAFE_LOOKALIKE_COLUMNS = [
        "seed_cohort_index",
        "lookalike_cohort_index",
        "seed_cluster_id",
        "lookalike_cluster_id",
        "seed_location_name",
        "seed_primary_poi_type",
        "lookalike_location_name",
        "lookalike_primary_poi_type",
        "similarity_score",
        "recommendation_reason",
    ]

    SAFE_AGGREGATE_COLUMNS = {
        "total_maid_volume",
        "noisy_maid_volume",
        "safe_maid_volume",
        "total_observations",
        "observation_count",
        "observations_count",
        "safe_observation_count",
    }

    BLOCKED_TOKENS = [
        "raw_maid",
        "device_id",
        "email",
        "phone",
        "latitude",
        "longitude",
        "raw_observation",
        "hashed",
        "lat",
        "lng",
        "user_id",
        "client_id",
    ]

    def __init__(self):
        self.config = load_production_config()
        self.validator = SafeSchemaValidator()
        self.audit_logger = AuditLogger(self.config.audit_log_path)

    def run(
        self,
        top_cohorts: pd.DataFrame,
        output_dir: str | Path,
        lookalikes: Optional[pd.DataFrame] = None,
        run_id: Optional[str] = None,
        audience_namespace: str = "punk_audience",
        approval_required: bool = True,
        min_management_quality: float = 0.25,
        max_export_cohorts: Optional[int] = None,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = run_id or f"safe_export_run_{uuid.uuid4().hex[:12]}"

        self._validate_runtime_config(
            min_management_quality=min_management_quality,
            max_export_cohorts=max_export_cohorts,
        )

        self.audit_logger.log(
            "safe_export_started",
            {
                "run_id": run_id,
                "top_cohort_rows": int(len(top_cohorts)),
                "lookalike_rows": int(0 if lookalikes is None else len(lookalikes)),
                "approval_required": bool(approval_required),
                "min_management_quality": float(min_management_quality),
            },
        )

        self._validate_input_dataframe(top_cohorts, "safe_export_top_cohorts")
        if lookalikes is not None:
            self._validate_input_dataframe(lookalikes, "safe_export_lookalikes")

        export_cohorts = self._prepare_export_cohorts(
            top_cohorts=top_cohorts,
            run_id=run_id,
            audience_namespace=audience_namespace,
            approval_required=approval_required,
            min_management_quality=min_management_quality,
            max_export_cohorts=max_export_cohorts,
        )

        export_lookalikes = self._prepare_export_lookalikes(
            lookalikes=lookalikes,
            export_cohorts=export_cohorts,
        )

        export_payload = self._build_export_payload(
            export_cohorts=export_cohorts,
            export_lookalikes=export_lookalikes,
            run_id=run_id,
            approval_required=approval_required,
        )

        approval_request = self._build_approval_request(
            export_cohorts=export_cohorts,
            export_lookalikes=export_lookalikes,
            run_id=run_id,
            approval_required=approval_required,
        )

        cohorts_path = output_dir / "safe_export_cohorts.csv"
        lookalikes_path = output_dir / "safe_export_lookalikes.csv"
        payload_path = output_dir / "safe_export_payload.json"
        approval_path = output_dir / "export_approval_request.json"
        manifest_path = output_dir / "safe_export_manifest.json"

        self._validate_no_blocked_columns(export_cohorts.columns, "safe_export_cohorts_output")
        self._validate_no_blocked_columns(export_lookalikes.columns, "safe_export_lookalikes_output")

        export_cohorts.to_csv(cohorts_path, index=False)
        export_lookalikes.to_csv(lookalikes_path, index=False)
        self._write_json(payload_path, export_payload)
        self._write_json(approval_path, approval_request)

        checksums = {
            "safe_export_cohorts": self._sha256_file(cohorts_path),
            "safe_export_lookalikes": self._sha256_file(lookalikes_path),
            "safe_export_payload": self._sha256_file(payload_path),
            "export_approval_request": self._sha256_file(approval_path),
        }

        manifest = {
            "module": "Safe Export",
            "status": "completed",
            "run_id": run_id,
            "approval_required": bool(approval_required),
            "approval_status": "pending_approval" if approval_required else "not_required",
            "downstream_export_enabled": False if approval_required else True,
            "export_blocked_until_approved": bool(approval_required),
            "exported_cohorts": int(len(export_cohorts)),
            "exported_lookalike_pairs": int(len(export_lookalikes)),
            "min_management_quality": float(min_management_quality),
            "raw_maids_exported": False,
            "hashed_identifiers_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
            "outputs": {
                "safe_export_cohorts": str(cohorts_path),
                "safe_export_lookalikes": str(lookalikes_path),
                "safe_export_payload": str(payload_path),
                "export_approval_request": str(approval_path),
                "safe_export_manifest": str(manifest_path),
            },
            "checksums_sha256": checksums,
        }

        self._write_json(manifest_path, manifest)

        self.audit_logger.log(
            "safe_export_completed",
            {
                "run_id": run_id,
                "exported_cohorts": int(len(export_cohorts)),
                "exported_lookalike_pairs": int(len(export_lookalikes)),
                "approval_status": manifest["approval_status"],
                "manifest": str(manifest_path),
            },
        )

        return manifest

    def run_from_artifacts(
        self,
        top_cohorts_path: str | Path,
        output_dir: str | Path,
        lookalikes_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        top_cohorts = pd.read_csv(top_cohorts_path)
        lookalikes = pd.read_csv(lookalikes_path) if lookalikes_path else None

        return self.run(
            top_cohorts=top_cohorts,
            lookalikes=lookalikes,
            output_dir=output_dir,
            **kwargs,
        )

    def _validate_runtime_config(
        self,
        *,
        min_management_quality: float,
        max_export_cohorts: Optional[int],
    ) -> None:
        if not 0 <= min_management_quality <= 1:
            raise ValueError("min_management_quality must be between 0 and 1.")

        if max_export_cohorts is not None and max_export_cohorts <= 0:
            raise ValueError("max_export_cohorts must be greater than 0.")

    def _validate_input_dataframe(self, df: pd.DataFrame, context: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{context} must be a pandas DataFrame.")

        if df.empty:
            raise ValueError(f"{context} cannot be empty.")

        self._validate_no_blocked_columns(df.columns, context)
        self.validator.validate_no_secret_values(df, context=context)

    def _validate_no_blocked_columns(self, columns, context: str) -> None:
        for col in columns:
            lower = str(col).lower()

            if lower in self.SAFE_AGGREGATE_COLUMNS:
                continue

            for token in self.BLOCKED_TOKENS:
                if lower == token or lower.startswith(f"{token}_") or lower.endswith(f"_{token}"):
                    raise ValueError(f"{context} contains blocked sensitive column: {col}")

            if lower in {
                "maid",
                "maids",
                "raw_maid",
                "device_id",
                "email",
                "phone",
                "lat",
                "lng",
                "latitude",
                "longitude",
                "user_id",
                "client_id",
            }:
                raise ValueError(f"{context} contains blocked sensitive column: {col}")

    def _prepare_export_cohorts(
        self,
        *,
        top_cohorts: pd.DataFrame,
        run_id: str,
        audience_namespace: str,
        approval_required: bool,
        min_management_quality: float,
        max_export_cohorts: Optional[int],
    ) -> pd.DataFrame:
        df = top_cohorts.copy()

        safe_cols = [col for col in self.SAFE_COHORT_COLUMNS if col in df.columns]
        df = df[safe_cols].copy()

        required_defaults = {
            "cohort_index": 0,
            "cluster_id": 0,
            "cluster_size": 1,
            "location_name": "all_locations",
            "primary_poi_type": "all_poi_types",
            "created_day_part": "all_day_parts",
            "lookback_bucket": "all_lookbacks",
            "sessions": 1,
            "total_maid_volume": 1,
            "noisy_maid_volume": 1,
            "quality_score": 0.5,
            "cluster_coherence_score": 0.5,
            "management_quality_score": 0.5,
            "export_ready": False,
            "recommendation_reason": "safe aggregated cohort",
            "trait_text": "",
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        text_cols = [
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "recommendation_reason",
            "trait_text",
        ]

        for col in text_cols:
            df[col] = (
                df[col]
                .replace([np.nan, None], required_defaults[col])
                .astype(str)
                .str.strip()
                .replace({"": required_defaults[col], "nan": required_defaults[col], "None": required_defaults[col]})
            )

        numeric_cols = [
            "cohort_index",
            "cluster_id",
            "cluster_size",
            "sessions",
            "total_maid_volume",
            "noisy_maid_volume",
            "quality_score",
            "cluster_coherence_score",
            "management_quality_score",
        ]

        for col in numeric_cols:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(required_defaults[col])
            )

        df["quality_score"] = df["quality_score"].clip(0, 1)
        df["cluster_coherence_score"] = df["cluster_coherence_score"].clip(0, 1)
        df["management_quality_score"] = df["management_quality_score"].clip(0, 1)

        df["export_ready"] = self._bool_series(df["export_ready"])

        df = df[
            (df["export_ready"])
            & (df["management_quality_score"] >= min_management_quality)
        ].copy()

        if df.empty:
            raise ValueError("No export-ready cohorts found after safety and quality filtering.")

        df = df.sort_values(
            ["management_quality_score", "cluster_coherence_score", "noisy_maid_volume"],
            ascending=False,
        )

        if max_export_cohorts is not None:
            df = df.head(max_export_cohorts)

        df = df.reset_index(drop=True)

        df["export_cohort_id"] = df.apply(
            lambda row: self._safe_export_id(row=row, namespace=audience_namespace),
            axis=1,
        )

        df["audience_name"] = df.apply(self._build_audience_name, axis=1)
        df["export_status"] = "pending_approval" if approval_required else "ready_for_downstream"
        df["privacy_mode"] = "aggregated_dp_safe"
        df["allowed_destination"] = "approved_downstream_only"
        df["data_safety_status"] = "safe_aggregated_no_raw_identifiers"

        output_cols = [
            "export_cohort_id",
            "audience_name",
            "cohort_index",
            "cluster_id",
            "cluster_size",
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "sessions",
            "total_maid_volume",
            "noisy_maid_volume",
            "quality_score",
            "cluster_coherence_score",
            "management_quality_score",
            "recommendation_reason",
            "privacy_mode",
            "allowed_destination",
            "data_safety_status",
            "export_status",
            "trait_text",
        ]

        return df[output_cols]

    def _prepare_export_lookalikes(
        self,
        *,
        lookalikes: Optional[pd.DataFrame],
        export_cohorts: pd.DataFrame,
    ) -> pd.DataFrame:
        output_cols = [
            "seed_export_cohort_id",
            "lookalike_export_cohort_id",
            "seed_cohort_index",
            "lookalike_cohort_index",
            "seed_cluster_id",
            "lookalike_cluster_id",
            "seed_location_name",
            "seed_primary_poi_type",
            "lookalike_location_name",
            "lookalike_primary_poi_type",
            "similarity_score",
            "recommendation_reason",
            "data_safety_status",
        ]

        if lookalikes is None or lookalikes.empty:
            return pd.DataFrame(columns=output_cols)

        df = lookalikes.copy()
        safe_cols = [col for col in self.SAFE_LOOKALIKE_COLUMNS if col in df.columns]
        df = df[safe_cols].copy()

        cohort_id_map = {
            int(row["cohort_index"]): row["export_cohort_id"]
            for _, row in export_cohorts.iterrows()
        }

        for col in ["seed_cohort_index", "lookalike_cohort_index", "seed_cluster_id", "lookalike_cluster_id"]:
            if col not in df.columns:
                df[col] = -1
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

        for col in [
            "seed_location_name",
            "seed_primary_poi_type",
            "lookalike_location_name",
            "lookalike_primary_poi_type",
            "recommendation_reason",
        ]:
            if col not in df.columns:
                df[col] = "not_available"
            df[col] = df[col].replace([np.nan, None], "not_available").astype(str)

        if "similarity_score" not in df.columns:
            df["similarity_score"] = 0.0

        df["similarity_score"] = (
            pd.to_numeric(df["similarity_score"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-1, 1)
        )

        df["seed_export_cohort_id"] = df["seed_cohort_index"].map(cohort_id_map)
        df["lookalike_export_cohort_id"] = df["lookalike_cohort_index"].map(cohort_id_map)

        df = df.dropna(subset=["seed_export_cohort_id", "lookalike_export_cohort_id"]).copy()

        df["data_safety_status"] = "safe_aggregated_no_raw_identifiers"

        return df[output_cols].reset_index(drop=True)

    def _build_export_payload(
        self,
        *,
        export_cohorts: pd.DataFrame,
        export_lookalikes: pd.DataFrame,
        run_id: str,
        approval_required: bool,
    ) -> Dict[str, Any]:
        return {
            "package_type": "safe_audience_export",
            "run_id": run_id,
            "approval_required": bool(approval_required),
            "approval_status": "pending_approval" if approval_required else "not_required",
            "downstream_export_enabled": False if approval_required else True,
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "individual_user_data_exported": False,
                "aggregation_level": "cohort",
            },
            "cohorts": self._json_safe(export_cohorts.to_dict(orient="records")),
            "lookalike_pairs": self._json_safe(export_lookalikes.to_dict(orient="records")),
        }

    def _build_approval_request(
        self,
        *,
        export_cohorts: pd.DataFrame,
        export_lookalikes: pd.DataFrame,
        run_id: str,
        approval_required: bool,
    ) -> Dict[str, Any]:
        return {
            "approval_request_type": "safe_audience_export",
            "run_id": run_id,
            "approval_required": bool(approval_required),
            "approval_status": "pending_approval" if approval_required else "not_required",
            "export_blocked_until_approved": bool(approval_required),
            "review_required_before_downstream_delivery": bool(approval_required),
            "exported_cohorts": int(len(export_cohorts)),
            "exported_lookalike_pairs": int(len(export_lookalikes)),
            "review_checklist": [
                "Confirm no raw MAIDs, hashed identifiers, emails, phones, lat/lng, or individual rows are present.",
                "Confirm all audiences are aggregated cohort-level outputs.",
                "Confirm destination platform and legal approval before activation.",
                "Confirm export_status is pending_approval before any downstream delivery.",
            ],
        }

    def _safe_export_id(self, *, row: pd.Series, namespace: str) -> str:
        raw = "|".join(
            [
                str(namespace),
                str(int(row.get("cohort_index", 0))),
                str(int(row.get("cluster_id", 0))),
                str(row.get("location_name", "")),
                str(row.get("primary_poi_type", "")),
                str(row.get("created_day_part", "")),
                str(row.get("lookback_bucket", "")),
            ]
        )

        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{namespace}_{digest}"

    def _build_audience_name(self, row: pd.Series) -> str:
        parts = [
            "Punk Audience",
            str(row.get("primary_poi_type", "cohort")).replace("_", " ").title(),
            str(row.get("created_day_part", "all")).title(),
            str(row.get("location_name", "all locations")).title(),
            f"Cluster {int(row.get('cluster_id', 0))}",
        ]

        name = " - ".join(parts)
        return " ".join(name.split())[:120]

    def _bool_series(self, series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.write_text(json.dumps(self._json_safe(data), indent=2, allow_nan=False))

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._json_safe(v) for v in value]

        if isinstance(value, tuple):
            return [self._json_safe(v) for v in value]

        if isinstance(value, np.ndarray):
            return value.tolist()

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
