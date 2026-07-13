import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np
import pandas as pd

from app.core.production_guardrails import (
    require_local_file_storage_allowed,
)


PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")


UNSAFE_KEYWORDS = [
    "email",
    "phone",
    "device",
    "client_id",
    "hash",
    "raw",
    "lat",
    "lon",
    "latitude",
    "longitude",
]

# Aggregate MAID volume is allowed because it is cohort-level, not a raw MAID/id.
ALLOWED_AGGREGATE_COLUMNS = {
    "total_maid_volume",
    "noisy_maid_volume",
    "safe_maid_volume",
}


def processed_path_for_job(job_id: str) -> Path:
    return PROCESSED_DIR / f"{job_id}_clean_features.csv"


def is_safe_column(column_name: str) -> bool:
    """
    Blocks columns that may expose individual-level information.

    Cohort-level aggregate MAID volume columns are allowed because they do not
    contain raw MAIDs or per-device identifiers.
    """
    col = str(column_name).lower().strip()

    if col in ALLOWED_AGGREGATE_COLUMNS:
        return True

    if col in {"maid", "maids", "maid_id", "maid_ids", "raw_maids", "raw_maid"}:
        return False

    return not any(keyword in col for keyword in UNSAFE_KEYWORDS)


def unsafe_columns(df: pd.DataFrame) -> List[str]:
    return [str(col) for col in df.columns if not is_safe_column(str(col))]


def validate_no_unsafe_columns(df: pd.DataFrame, context: str) -> None:
    bad = unsafe_columns(df)
    if bad:
        raise ValueError(f"Unsafe synthetic {context} columns blocked: {bad}")


def safe_trait_columns(df: pd.DataFrame) -> List[str]:
    """
    Selects only safe columns for synthetic generation.
    """
    blocked_exact = {
        "privacy_status",
        "k_min",
    }

    return [
        col for col in df.columns
        if is_safe_column(col) and col not in blocked_exact
    ]


def build_segment_label(row: Dict[str, Any]) -> str:
    """
    Creates readable synthetic segment label.
    """
    parts = []

    for key in ["city", "interest", "visit_time", "affinity", "age_group", "location_name", "primary_poi_type"]:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            parts.append(str(value))

    return "_".join(parts).lower().replace(" ", "_") or "synthetic_segment"


def fallback_synthetic_from_aggregates(
    df: pd.DataFrame,
    num_rows: int,
) -> pd.DataFrame:
    """
    Privacy-safe aggregate sampler.

    This does not create individual real users. It samples from privacy-safe
    aggregated cohort rows and creates synthetic seed profiles.
    """
    validate_no_unsafe_columns(df, context="source")

    trait_cols = safe_trait_columns(df)

    if not trait_cols:
        raise ValueError("No safe trait columns available for synthetic generation.")

    weights = None

    if "noisy_count" in df.columns:
        weights = pd.to_numeric(df["noisy_count"], errors="coerce").fillna(1).astype(float).values
    elif "cohort_size" in df.columns:
        weights = pd.to_numeric(df["cohort_size"], errors="coerce").fillna(1).astype(float).values
    elif "noisy_maid_volume" in df.columns:
        weights = pd.to_numeric(df["noisy_maid_volume"], errors="coerce").fillna(1).astype(float).values
    elif "total_maid_volume" in df.columns:
        weights = pd.to_numeric(df["total_maid_volume"], errors="coerce").fillna(1).astype(float).values

    if weights is None or weights.sum() <= 0:
        probabilities = None
    else:
        probabilities = weights / weights.sum()

    synthetic_rows = []

    for index in range(num_rows):
        sampled_idx = np.random.choice(df.index.values, p=probabilities)
        source_row = df.loc[sampled_idx].to_dict()

        safe_row = {}

        for col in trait_cols:
            safe_row[col] = source_row.get(col)

        safe_row["synthetic_profile_id"] = f"synth_{uuid4().hex[:12]}"
        safe_row["synthetic_segment"] = build_segment_label(safe_row)
        safe_row["synthetic_source"] = "aggregated_privacy_safe_features"
        safe_row["synthetic_rank"] = index + 1
        safe_row["privacy_mode"] = "aggregated_synthetic_only"
        safe_row["approval_status"] = "pending_approval"

        synthetic_rows.append(safe_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    validate_no_unsafe_columns(synthetic_df, context="output")
    return synthetic_df


def try_sdv_synthetic(df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
    """
    Dev-only SDV GaussianCopula path.

    This is not allowed in production mode because it is not the approved
    production DP synthetic provider.
    """
    validate_no_unsafe_columns(df, context="source")

    safe_cols = safe_trait_columns(df)
    safe_df = df[safe_cols].copy()

    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(safe_df)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(safe_df)

    synthetic_df = synthesizer.sample(num_rows=num_rows)

    synthetic_df["synthetic_profile_id"] = [
        f"synth_{uuid4().hex[:12]}" for _ in range(len(synthetic_df))
    ]
    synthetic_df["synthetic_source"] = "sdv_gaussian_copula_dev_only"
    synthetic_df["synthetic_rank"] = range(1, len(synthetic_df) + 1)
    synthetic_df["privacy_mode"] = "dev_synthetic_only"
    synthetic_df["approval_status"] = "pending_approval"

    validate_no_unsafe_columns(synthetic_df, context="output")
    return synthetic_df


def _validate_generation_request(
    *,
    num_rows: int,
    use_sdv: bool,
    production_mode: bool,
    allow_fallback: bool,
) -> None:
    if int(num_rows) <= 0:
        raise ValueError("num_rows must be greater than 0.")

    if production_mode and allow_fallback:
        raise ValueError("Production synthetic generation cannot allow fallback engines.")

    if production_mode and use_sdv:
        raise ValueError(
            "Legacy SDV Gaussian synthetic generation is dev-only. "
            "Production must use aggregate-safe generation or SyntheticEngineAgent dp_aggregate."
        )


def generate_synthetic_for_job(
    job_id: str,
    num_rows: int = 1000,
    use_sdv: bool = False,
    production_mode: bool = True,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Legacy synthetic generation pipeline, now hardened.

    Production behavior:
    - no SDV Gaussian path
    - no silent fallback
    - no unsafe source/output columns
    - aggregate synthetic seed profiles only
    """
    require_local_file_storage_allowed(
        "legacy synthetic generation"
    )

    _validate_generation_request(
        num_rows=num_rows,
        use_sdv=use_sdv,
        production_mode=production_mode,
        allow_fallback=allow_fallback,
    )

    processed_path = processed_path_for_job(job_id)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    if df.empty:
        raise ValueError("Processed feature table is empty. Cannot generate synthetic data.")

    validate_no_unsafe_columns(df, context="source")

    backend = "aggregated_sampler"
    fallback_reason = None

    if use_sdv:
        try:
            synthetic_df = try_sdv_synthetic(df, num_rows=num_rows)
            backend = "sdv_gaussian_copula_dev_only"
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Synthetic generation failed closed. Reason: {exc}") from exc
            fallback_reason = str(exc)
            backend = "aggregated_sampler_explicit_dev_fallback"
            synthetic_df = fallback_synthetic_from_aggregates(df, num_rows=num_rows)
    else:
        synthetic_df = fallback_synthetic_from_aggregates(df, num_rows=num_rows)

    validate_no_unsafe_columns(synthetic_df, context="output")

    synthetic_path = SYNTHETIC_DIR / f"{job_id}_synthetic.csv"
    manifest_path = SYNTHETIC_DIR / f"{job_id}_synthetic_manifest.json"

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(synthetic_path, index=False)

    manifest = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_processed_path": str(processed_path),
        "synthetic_path": str(synthetic_path),
        "num_rows_requested": int(num_rows),
        "num_rows_generated": int(len(synthetic_df)),
        "backend": backend,
        "use_sdv_requested": bool(use_sdv),
        "production_mode": bool(production_mode),
        "allow_fallback": bool(allow_fallback),
        "fallback_reason": fallback_reason,
        "privacy_mode": "synthetic_from_aggregated_features",
        "contains_raw_pii": False,
        "contains_raw_maids": False,
        "contains_hashed_identifiers": False,
        "contains_individual_user_data": False,
        "safe_for_export_seed": True,
        "requires_manual_approval_before_upload": True,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    return {
        "status": "completed",
        "message": "Synthetic dataset generated",
        "job_id": job_id,
        "backend": backend,
        "synthetic_path": str(synthetic_path),
        "manifest_path": str(manifest_path),
        "num_rows_generated": int(len(synthetic_df)),
        "privacy_mode": "synthetic_from_aggregated_features",
        "contains_raw_pii": False,
        "contains_raw_maids": False,
        "contains_hashed_identifiers": False,
        "contains_individual_user_data": False,
        "safe_for_export_seed": True,
        "requires_manual_approval_before_upload": True,
        "production_mode": bool(production_mode),
        "allow_fallback": bool(allow_fallback),
        "fallback_reason": fallback_reason,
    }
