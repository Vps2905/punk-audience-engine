import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


UNSAFE_KEYWORDS = [
    "email",
    "phone",
    "device",
    "maid",
    "client_id",
    "hash",
    "raw",
    "lat",
    "lon",
    "latitude",
    "longitude"
]


def processed_path_for_job(job_id: str) -> Path:
    return PROCESSED_DIR / f"{job_id}_clean_features.csv"


def is_safe_column(column_name: str) -> bool:
    """
    Blocks columns that may expose individual-level information.
    """
    col = str(column_name).lower()
    return not any(keyword in col for keyword in UNSAFE_KEYWORDS)


def safe_trait_columns(df: pd.DataFrame) -> List[str]:
    """
    Selects only safe columns for synthetic generation.
    """
    blocked_exact = {
        "cohort_size",
        "noisy_count",
        "privacy_status",
        "k_min"
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

    for key in ["city", "interest", "visit_time", "affinity", "age_group"]:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            parts.append(str(value))

    return "_".join(parts).lower().replace(" ", "_") or "synthetic_segment"


def fallback_synthetic_from_aggregates(
    df: pd.DataFrame,
    num_rows: int
) -> pd.DataFrame:
    """
    Privacy-safe fallback synthetic generator.

    It does not create individual real users.
    It samples from aggregated cohort rows and creates fake seed profiles.
    """
    trait_cols = safe_trait_columns(df)

    if not trait_cols:
        raise ValueError("No safe trait columns available for synthetic generation.")

    weights = None

    if "noisy_count" in df.columns:
        weights = df["noisy_count"].fillna(1).astype(float).values
    elif "cohort_size" in df.columns:
        weights = df["cohort_size"].fillna(1).astype(float).values

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

        if "noisy_count" in source_row:
            safe_row["source_noisy_count"] = source_row.get("noisy_count")

        synthetic_rows.append(safe_row)

    return pd.DataFrame(synthetic_rows)


def try_sdv_synthetic(df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
    """
    Tries to generate synthetic data using SDV if installed.

    If SDV is unavailable or fails, caller will use fallback.
    """
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
    synthetic_df["synthetic_source"] = "sdv_gaussian_copula"
    synthetic_df["synthetic_rank"] = range(1, len(synthetic_df) + 1)

    return synthetic_df


def generate_synthetic_for_job(
    job_id: str,
    num_rows: int = 1000,
    use_sdv: bool = True
) -> Dict[str, Any]:
    """
    Main synthetic generation pipeline.

    1. Load processed privacy-safe features
    2. Try SDV if requested
    3. If SDV fails, use safe aggregated fallback
    4. Save synthetic CSV
    5. Save manifest
    """
    processed_path = processed_path_for_job(job_id)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    if df.empty:
        raise ValueError("Processed feature table is empty. Cannot generate synthetic data.")

    backend = "fallback_aggregated_sampler"
    fallback_reason = None

    if use_sdv:
        try:
            synthetic_df = try_sdv_synthetic(df, num_rows=num_rows)
            backend = "sdv_gaussian_copula"
        except Exception as e:
            fallback_reason = str(e)
            synthetic_df = fallback_synthetic_from_aggregates(df, num_rows=num_rows)
    else:
        synthetic_df = fallback_synthetic_from_aggregates(df, num_rows=num_rows)

    synthetic_path = SYNTHETIC_DIR / f"{job_id}_synthetic.csv"
    manifest_path = SYNTHETIC_DIR / f"{job_id}_synthetic_manifest.json"

    synthetic_df.to_csv(synthetic_path, index=False)

    manifest = {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_processed_path": str(processed_path),
        "synthetic_path": str(synthetic_path),
        "num_rows_requested": num_rows,
        "num_rows_generated": int(len(synthetic_df)),
        "backend": backend,
        "fallback_reason": fallback_reason,
        "privacy_mode": "synthetic_from_aggregated_features",
        "contains_raw_pii": False,
        "contains_individual_user_data": False,
        "safe_for_export_seed": True
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

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
        "safe_for_export_seed": True,
        "fallback_reason": fallback_reason
    }
