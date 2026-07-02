from typing import List
import pandas as pd

from app.utils.hashing import sha256_hash
from app.utils.dp_noise import add_laplace_noise


PII_COLUMNS = ["email", "phone", "device_id", "client_id", "maid"]
DEFAULT_K_ANONYMITY = 1000


def create_age_group(age) -> str:
    try:
        age = int(age)
    except Exception:
        return "unknown"

    if age < 18:
        return "under_18"
    if age <= 24:
        return "18_24"
    if age <= 34:
        return "25_34"
    if age <= 44:
        return "35_44"
    if age <= 54:
        return "45_54"
    if age <= 64:
        return "55_64"
    return "65_plus"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    if "age" in df.columns:
        df["age_group"] = df["age"].apply(create_age_group)
        df = df.drop(columns=["age"])

    return df


def hash_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in PII_COLUMNS:
        if col in df.columns:
            df[f"{col}_hash"] = df[col].apply(sha256_hash)
            df = df.drop(columns=[col])

    return df


def build_trait_columns(df: pd.DataFrame) -> List[str]:
    blocked_keywords = ["hash", "raw", "exact", "lat", "lon", "latitude", "longitude"]
    trait_cols = []

    for col in df.columns:
        if any(keyword in col for keyword in blocked_keywords):
            continue
        trait_cols.append(col)

    return trait_cols


def aggregate_with_privacy(
    df: pd.DataFrame,
    k_min: int = DEFAULT_K_ANONYMITY,
    epsilon: float = 1.0
) -> pd.DataFrame:
    trait_cols = build_trait_columns(df)

    if not trait_cols:
        raise ValueError("No safe trait columns found for aggregation.")

    grouped = (
        df.groupby(trait_cols, dropna=False)
        .size()
        .reset_index(name="cohort_size")
    )

    grouped = grouped[grouped["cohort_size"] >= k_min].copy()

    grouped["noisy_count"] = grouped["cohort_size"].apply(
        lambda count: add_laplace_noise(count, epsilon)
    )

    grouped["privacy_status"] = "passed"
    grouped["k_min"] = k_min

    return grouped


def apply_privacy_pipeline(
    df: pd.DataFrame,
    k_min: int = DEFAULT_K_ANONYMITY,
    epsilon: float = 1.0
) -> pd.DataFrame:
    """
    Main function used by ingestion_service.

    Flow:
    1. normalize column names
    2. create age_group
    3. hash PII
    4. aggregate records
    5. remove small groups
    6. add DP noise
    """
    normalized = normalize_columns(df)
    hashed = hash_pii_columns(normalized)
    protected = aggregate_with_privacy(hashed, k_min=k_min, epsilon=epsilon)

    return protected
