from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.core.approval_workflow import ApprovalWorkflow
from app.core.audit_logger import AuditLogger
from app.core.privacy_budget_ledger import PrivacyBudgetLedger
from app.core.production_config import load_production_config
from app.core.schema_validator import SafeSchemaValidator


class SyntheticEngineAgent:
    """
    Production-hardened SyntheticEngineAgent.

    Supported engines:
    - dp_aggregate: built-in production provider using k-anonymous aggregated cohorts + DP noise.
    - sdv_dpgc / dpgc: SDV Enterprise DPGCSynthesizer provider.
    - external_dp_service: future Docker/cloud DP service provider.
    - auto / gaussian / ctgan: local/dev only when fallback is explicitly allowed.

    Production rules:
    - No silent fallback.
    - Raw identifiers are rejected before cleaning.
    - Production inputs must already pass k-anonymity.
    - DP engines record epsilon in privacy ledger.
    - Output always requires pending approval before export.
    """

    SAFE_COLUMNS = [
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
        "sessions",
        "total_maid_volume",
        "noisy_maid_volume",
        "safe_maid_volume",
        "quality_score",
        "privacy_status",
        "cluster_id",
        "trait_text",
    ]

    BLOCKED_COLUMNS = [
        "maid",
        "maids",
        "raw_maid",
        "raw_maids",
        "device_id",
        "email",
        "phone",
        "lat",
        "lng",
        "latitude",
        "longitude",
        "observation",
        "observations",
    ]

    DP_ENGINES = {"DPAggregateCohortSynthesizer", "DPGCSynthesizer"}

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.config = load_production_config()
        self.validator = SafeSchemaValidator()
        self.audit_logger = AuditLogger(self.config.audit_log_path)
        self.ledger = PrivacyBudgetLedger(
            self.config.privacy_ledger_path,
            max_epsilon_per_run=self.config.max_epsilon_per_run,
        )
        self.approval_workflow = ApprovalWorkflow(self.config.approval_dir)

    def generate(
        self,
        cohorts: pd.DataFrame,
        output_dir: str | Path,
        synthetic_rows: int = 1000,
        engine_requested: Optional[str] = None,
        epsilon: Optional[float] = None,
        production_mode: Optional[bool] = None,
        allow_fallback: Optional[bool] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = run_id or f"synthetic_run_{uuid.uuid4().hex[:12]}"
        engine_requested = (engine_requested or self.config.synthetic_engine).strip().lower()
        epsilon = float(epsilon if epsilon is not None else self.config.synthetic_epsilon)
        production_mode = self.config.production_mode if production_mode is None else bool(production_mode)
        allow_fallback = (
            self.config.allow_synthetic_fallback
            if allow_fallback is None
            else bool(allow_fallback)
        )
        k_min = int(self.config.k_min)

        self._validate_runtime_config(
            engine_requested=engine_requested,
            epsilon=epsilon,
            production_mode=production_mode,
            allow_fallback=allow_fallback,
            synthetic_rows=synthetic_rows,
        )

        self.audit_logger.log(
            "synthetic_generation_started",
            {
                "run_id": run_id,
                "engine_requested": engine_requested,
                "epsilon": epsilon,
                "k_min": k_min,
                "production_mode": production_mode,
                "allow_fallback": allow_fallback,
                "synthetic_rows": synthetic_rows,
            },
        )

        # Fail closed before cleaning/dropping anything.
        self.validator.validate_no_blocked_columns(cohorts.columns, context="synthetic_raw_input")
        self.validator.validate_no_secret_values(cohorts, context="synthetic_raw_input")

        safe_input = self._prepare_safe_input(cohorts)
        self.validator.validate_safe_cohort_dataframe(safe_input, context="synthetic_safe_input")

        if production_mode:
            self._enforce_k_min_input(safe_input, k_min=k_min)

        engine_used = None
        fallback_reason = None
        dpgc_available = False
        synthetic_df: Optional[pd.DataFrame] = None

        if engine_requested == "dp_aggregate":
            synthetic_df = self._try_dp_aggregate(
                safe_input,
                rows=synthetic_rows,
                epsilon=epsilon,
                k_min=k_min,
            )
            engine_used = "DPAggregateCohortSynthesizer"

        elif engine_requested in {"sdv_dpgc", "dpgc", "dpgcsynthesizer"}:
            try:
                synthetic_df = self._try_dpgc(safe_input, synthetic_rows, epsilon)
                engine_used = "DPGCSynthesizer"
                dpgc_available = True
            except Exception as exc:
                fallback_reason = f"DPGCSynthesizer unavailable or failed: {exc}"

        elif engine_requested == "external_dp_service":
            try:
                synthetic_df = self._try_external_dp_service(safe_input, synthetic_rows, epsilon, k_min)
                engine_used = "ExternalDPSyntheticService"
            except Exception as exc:
                fallback_reason = f"External DP service unavailable or failed: {exc}"

        elif not production_mode and allow_fallback and engine_requested == "auto":
            try:
                synthetic_df = self._try_dpgc(safe_input, synthetic_rows, epsilon)
                engine_used = "DPGCSynthesizer"
                dpgc_available = True
            except Exception as exc:
                fallback_reason = f"DPGCSynthesizer unavailable or failed: {exc}"

        # Local/dev optional engines only.
        if synthetic_df is None and not production_mode and allow_fallback:
            if engine_requested in {"auto", "gaussian", "gaussiancopula", "gaussiancopulasynthesizer"}:
                try:
                    synthetic_df = self._try_gaussian_copula(safe_input, synthetic_rows)
                    engine_used = "GaussianCopulaSynthesizer"
                except Exception as exc:
                    fallback_reason = f"GaussianCopulaSynthesizer unavailable or failed: {exc}"

        if synthetic_df is None and not production_mode and allow_fallback:
            if engine_requested in {"auto", "ctgan", "ctgansynthesizer"}:
                try:
                    synthetic_df = self._try_ctgan(safe_input, synthetic_rows)
                    engine_used = "CTGANSynthesizer"
                except Exception as exc:
                    fallback_reason = f"CTGANSynthesizer unavailable or failed: {exc}"

        if synthetic_df is None and not production_mode and allow_fallback:
            synthetic_df = self._local_dev_sampler(safe_input, synthetic_rows)
            engine_used = "local_dev_aggregated_sampler"
            if not fallback_reason:
                fallback_reason = "Local/dev sampler explicitly allowed."

        if synthetic_df is None:
            self.audit_logger.log(
                "synthetic_generation_failed_closed",
                {
                    "run_id": run_id,
                    "engine_requested": engine_requested,
                    "production_mode": production_mode,
                    "allow_fallback": allow_fallback,
                    "reason": fallback_reason,
                },
            )
            raise RuntimeError(
                "Synthetic generation failed closed. "
                "Production requires an explicit working DP provider. "
                f"Reason: {fallback_reason}"
            )

        synthetic_df = self._finalize_synthetic_output(synthetic_df, synthetic_rows)
        self.validator.validate_safe_cohort_dataframe(synthetic_df, context="synthetic_output")

        synthetic_csv = output_dir / "synthetic_safe_seed_profiles.csv"
        manifest_json = output_dir / "synthetic_manifest.json"
        schema_json = output_dir / "synthetic_safe_input_schema.json"

        synthetic_df.to_csv(synthetic_csv, index=False)

        safe_input_schema = {
            "columns": {col: str(dtype) for col, dtype in safe_input.dtypes.items()},
            "row_count": int(len(safe_input)),
            "production_safe": True,
            "raw_identifiers_present": False,
            "raw_coordinates_present": False,
            "individual_rows_present": False,
        }
        self._write_json(schema_json, safe_input_schema)

        privacy_budget_record = None
        if engine_used in self.DP_ENGINES:
            privacy_budget_record = self.ledger.record_spend(
                run_id=run_id,
                module="synthetic_generation",
                epsilon=epsilon,
                engine=engine_used,
                metadata={
                    "rows_generated": int(len(synthetic_df)),
                    "input_rows": int(len(safe_input)),
                    "k_min": k_min,
                },
            )

        approval_request = self.approval_workflow.create_request(
            run_id=run_id,
            module="synthetic_generation",
            output_dir=output_dir,
            artifacts=[str(synthetic_csv), str(manifest_json), str(schema_json)],
            summary={
                "engine_used": engine_used,
                "rows_generated": int(len(synthetic_df)),
                "production_mode": production_mode,
                "raw_identifiers_exported": False,
            },
        )

        manifest = {
            "module": "Synthetic Data Generation",
            "status": "completed",
            "run_id": run_id,
            "engine_requested": engine_requested,
            "engine_used": engine_used,
            "production_mode": production_mode,
            "allow_fallback": allow_fallback,
            "dpgc_available": dpgc_available,
            "fallback_reason": fallback_reason,
            "rows_generated": int(len(synthetic_df)),
            "epsilon": epsilon,
            "k_min": k_min,
            "privacy_budget_recorded": privacy_budget_record is not None,
            "privacy_budget_record": privacy_budget_record,
            "approval_status": approval_request["status"],
            "input_was_aggregated": True,
            "input_rows": int(len(safe_input)),
            "raw_maids_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
            "outputs": {
                "synthetic_csv": str(synthetic_csv),
                "synthetic_manifest": str(manifest_json),
                "safe_input_schema": str(schema_json),
                "approval_request": str(output_dir / "approval_request.json"),
            },
        }

        self._write_json(manifest_json, manifest)

        self.audit_logger.log(
            "synthetic_generation_completed",
            {
                "run_id": run_id,
                "engine_used": engine_used,
                "rows_generated": int(len(synthetic_df)),
                "approval_status": approval_request["status"],
                "synthetic_csv": str(synthetic_csv),
                "manifest": str(manifest_json),
            },
        )

        return manifest

    def _validate_runtime_config(
        self,
        *,
        engine_requested: str,
        epsilon: float,
        production_mode: bool,
        allow_fallback: bool,
        synthetic_rows: int,
    ) -> None:
        if synthetic_rows <= 0:
            raise ValueError("synthetic_rows must be greater than 0.")

        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

        if production_mode and allow_fallback:
            raise ValueError("Production mode cannot allow fallback synthetic engines.")

        production_engines = {
            "dp_aggregate",
            "sdv_dpgc",
            "dpgc",
            "dpgcsynthesizer",
            "external_dp_service",
        }

        if production_mode and engine_requested not in production_engines:
            raise ValueError(
                "Production mode requires explicit engine: "
                "dp_aggregate, sdv_dpgc, or external_dp_service."
            )

    def _prepare_safe_input(self, cohorts: pd.DataFrame) -> pd.DataFrame:
        df = cohorts.copy()

        safe_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            blocked = any(token in col_lower for token in self.BLOCKED_COLUMNS)

            if col in {"total_maid_volume", "noisy_maid_volume", "safe_maid_volume"}:
                blocked = False

            if col in self.SAFE_COLUMNS and not blocked:
                safe_cols.append(col)

        df = df[safe_cols].copy()

        text_defaults = {
            "location_name": "all_locations",
            "primary_poi_type": "all_poi_types",
            "created_day_part": "all_day_parts",
            "lookback_bucket": "all_lookbacks",
            "privacy_status": "safe_aggregated",
            "trait_text": "",
        }

        numeric_defaults = {
            "sessions": 1,
            "total_maid_volume": 1,
            "noisy_maid_volume": 1,
            "quality_score": 0.5,
            "cluster_id": -1,
        }

        for col, default in text_defaults.items():
            if col not in df.columns:
                df[col] = default

            df[col] = df[col].replace([np.nan, None], default)
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
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

        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default

            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(default)
            )

        max_quality = pd.to_numeric(df["quality_score"], errors="coerce").max()
        if pd.notna(max_quality) and max_quality > 1:
            df["quality_score"] = (df["quality_score"] / 100.0).clip(0, 1)
        else:
            df["quality_score"] = df["quality_score"].clip(0, 1)

        def build_trait_text(row: pd.Series) -> str:
            return " | ".join(
                [
                    f"location {row.get('location_name', 'all_locations')}",
                    f"poi {row.get('primary_poi_type', 'all_poi_types')}",
                    f"daypart {row.get('created_day_part', 'all_day_parts')}",
                    f"lookback {row.get('lookback_bucket', 'all_lookbacks')}",
                    f"quality {round(float(row.get('quality_score', 0.5)), 3)}",
                    f"sessions {int(float(row.get('sessions', 0)))}",
                ]
            )

        df["trait_text"] = df["trait_text"].astype(str).str.strip()
        weak_trait_mask = df["trait_text"].isin(["", "0", "0.0", "nan", "None", "none"])
        df.loc[weak_trait_mask, "trait_text"] = df[weak_trait_mask].apply(build_trait_text, axis=1)

        ordered_cols = [
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "sessions",
            "total_maid_volume",
            "noisy_maid_volume",
            "quality_score",
            "privacy_status",
            "cluster_id",
            "trait_text",
        ]

        return df[[c for c in ordered_cols if c in df.columns]].copy()

    def _enforce_k_min_input(self, df: pd.DataFrame, k_min: int) -> None:
        if k_min <= 0:
            raise ValueError("k_min must be greater than 0.")

        volume_col = None
        for candidate in ["noisy_maid_volume", "total_maid_volume", "safe_maid_volume"]:
            if candidate in df.columns:
                volume_col = candidate
                break

        if volume_col is None:
            raise ValueError("Production synthetic input must include aggregate volume column.")

        volumes = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        failed = int((volumes < k_min).sum())

        if failed > 0:
            raise ValueError(
                f"Production synthetic input contains {failed} cohorts below k_min={k_min}. "
                "Run PrivacyLayerAgent first or remove unsafe cohorts."
            )

    def _try_dp_aggregate(self, df: pd.DataFrame, rows: int, epsilon: float, k_min: int) -> pd.DataFrame:
        """
        Built-in production DP provider for aggregated cohorts.

        This does not generate individual users. It creates synthetic seed profiles
        from already k-anonymous aggregate cohorts and applies Laplace DP noise
        to aggregate count fields.
        """
        rng = np.random.default_rng(self.random_state)

        weight_col = "noisy_maid_volume" if "noisy_maid_volume" in df.columns else "total_maid_volume"
        weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(1).clip(lower=1)

        sampled = df.sample(
            n=rows,
            replace=True,
            weights=weights,
            random_state=self.random_state,
        ).reset_index(drop=True)

        noise_scale = 1.0 / epsilon

        for col in ["sessions", "total_maid_volume", "noisy_maid_volume"]:
            if col in sampled.columns:
                base = pd.to_numeric(sampled[col], errors="coerce").fillna(k_min).astype(float)
                noise = rng.laplace(loc=0.0, scale=noise_scale, size=len(sampled))

                lower = 1 if col == "sessions" else k_min
                sampled[col] = (base + noise).clip(lower=lower).round().astype(int)

        sampled["quality_score"] = (
            pd.to_numeric(sampled.get("quality_score", 0.5), errors="coerce")
            .fillna(0.5)
            .astype(float)
            .add(rng.normal(0, 0.015, len(sampled)))
            .clip(0, 1)
        )

        sampled["privacy_status"] = "passed"
        sampled["dp_provider"] = "dp_aggregate"

        return sampled

    def _try_dpgc(self, df: pd.DataFrame, rows: int, epsilon: float) -> pd.DataFrame:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import DPGCSynthesizer

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        try:
            synth = DPGCSynthesizer(metadata, epsilon=epsilon)
        except TypeError:
            synth = DPGCSynthesizer(metadata)

        synth.fit(df)
        return synth.sample(num_rows=rows)

    def _try_external_dp_service(
        self,
        df: pd.DataFrame,
        rows: int,
        epsilon: float,
        k_min: int,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "external_dp_service provider is reserved for Docker/cloud DP service integration."
        )

    def _try_gaussian_copula(self, df: pd.DataFrame, rows: int) -> pd.DataFrame:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        synth = GaussianCopulaSynthesizer(metadata)
        synth.fit(df)
        return synth.sample(num_rows=rows)

    def _try_ctgan(self, df: pd.DataFrame, rows: int) -> pd.DataFrame:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        synth = CTGANSynthesizer(metadata, epochs=20)
        synth.fit(df)
        return synth.sample(num_rows=rows)

    def _local_dev_sampler(self, df: pd.DataFrame, rows: int) -> pd.DataFrame:
        weight_col = "noisy_maid_volume" if "noisy_maid_volume" in df.columns else "total_maid_volume"
        weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(1).clip(lower=1)

        sampled = df.sample(
            n=rows,
            replace=True,
            weights=weights,
            random_state=self.random_state,
        ).reset_index(drop=True)

        rng = np.random.default_rng(self.random_state)

        sampled["quality_score"] = (
            pd.to_numeric(sampled.get("quality_score", 0.5), errors="coerce")
            .fillna(0.5)
            .astype(float)
            .add(rng.normal(0, 0.02, len(sampled)))
            .clip(0, 1)
        )

        return sampled

    def _finalize_synthetic_output(self, df: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
        df = df.copy().head(expected_rows).reset_index(drop=True)

        text_defaults = {
            "location_name": "all_locations",
            "primary_poi_type": "all_poi_types",
            "created_day_part": "all_day_parts",
            "lookback_bucket": "all_lookbacks",
            "privacy_status": "passed",
        }

        for col, default in text_defaults.items():
            if col not in df.columns:
                df[col] = default

            df[col] = df[col].replace([np.nan, None], default)
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
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

        int_defaults = {
            "sessions": 1,
            "total_maid_volume": 1,
            "noisy_maid_volume": 1,
            "cluster_id": -1,
        }

        for col, default in int_defaults.items():
            if col not in df.columns:
                df[col] = default

            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(default)
                .clip(lower=0)
                .round()
                .astype(int)
            )

        if "quality_score" not in df.columns:
            df["quality_score"] = 0.5

        df["quality_score"] = (
            pd.to_numeric(df["quality_score"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.5)
            .astype(float)
        )

        max_quality = df["quality_score"].max()
        if pd.notna(max_quality) and max_quality > 1:
            df["quality_score"] = df["quality_score"] / 100.0

        df["quality_score"] = df["quality_score"].clip(0, 1)

        def build_trait_text(row: pd.Series) -> str:
            return " | ".join(
                [
                    f"location {row.get('location_name', 'all_locations')}",
                    f"poi {row.get('primary_poi_type', 'all_poi_types')}",
                    f"daypart {row.get('created_day_part', 'all_day_parts')}",
                    f"lookback {row.get('lookback_bucket', 'all_lookbacks')}",
                    f"quality {round(float(row.get('quality_score', 0.5)), 3)}",
                    f"sessions {int(row.get('sessions', 0))}",
                ]
            )

        df["trait_text"] = df.apply(build_trait_text, axis=1)

        if "synthetic_seed_id" in df.columns:
            df = df.drop(columns=["synthetic_seed_id"])

        df.insert(0, "synthetic_seed_id", [f"synthetic_seed_{i+1:06d}" for i in range(len(df))])

        df["synthetic_profile"] = True
        df["privacy_mode"] = "aggregated_synthetic_only"
        df["approval_status"] = "pending_approval"

        keep_cols = []
        for col in df.columns:
            col_lower = str(col).lower()

            if col in {"total_maid_volume", "noisy_maid_volume", "safe_maid_volume"}:
                keep_cols.append(col)
                continue

            blocked = any(token in col_lower for token in self.BLOCKED_COLUMNS)
            if not blocked:
                keep_cols.append(col)

        return df[keep_cols]

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
