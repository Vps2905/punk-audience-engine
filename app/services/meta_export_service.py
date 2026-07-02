import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4

import pandas as pd

from app.services.cohort_service import get_cohort


EXPORT_DIR = Path("data/exports")
SYNTHETIC_DIR = Path("data/synthetic")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


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


EXPORT_STATUS = {}


def is_safe_column(column_name: str) -> bool:
    col = str(column_name).lower()
    return not any(keyword in col for keyword in UNSAFE_KEYWORDS)


def clean_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unsafe columns from synthetic seed export.
    """
    safe_cols = [col for col in df.columns if is_safe_column(col)]
    cleaned = df[safe_cols].copy()

    return cleaned


def synthetic_path_for_job(job_id: str) -> Path:
    return SYNTHETIC_DIR / f"{job_id}_synthetic.csv"


def build_meta_trait_payload(cohort: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds aggregated trait payload for Meta-safe audience planning.
    """
    return {
        "cohort_id": cohort.get("cohort_id"),
        "name": cohort.get("name"),
        "job_id": cohort.get("job_id"),
        "privacy_mode": cohort.get("privacy_mode", "aggregated_only"),
        "quality": cohort.get("quality", {}),
        "aggregated_traits": cohort.get("aggregated_traits", {}),
        "source_query": cohort.get("source_query"),
        "export_note": (
            "This payload contains aggregated audience traits only. "
            "It does not contain raw personal identifiers or individual user data."
        )
    }


def generate_meta_safe_export(
    cohort_id: str,
    seed_limit: int = 1000,
    approval_status: str = "pending_approval"
) -> Dict[str, Any]:
    """
    Generates Meta-safe export package.

    Output files:
    - aggregated traits JSON
    - synthetic seed CSV
    - export manifest JSON
    """
    cohort = get_cohort(cohort_id)

    if cohort.get("status") == "not_found":
        return cohort

    job_id = cohort["job_id"]
    export_id = f"export_{uuid4().hex[:12]}"

    cohort_export_dir = EXPORT_DIR / export_id
    cohort_export_dir.mkdir(parents=True, exist_ok=True)

    traits_payload = build_meta_trait_payload(cohort)

    traits_path = cohort_export_dir / f"{cohort_id}_aggregated_traits.json"
    seed_path = cohort_export_dir / f"{cohort_id}_synthetic_seed.csv"
    manifest_path = cohort_export_dir / f"{cohort_id}_export_manifest.json"

    with open(traits_path, "w", encoding="utf-8") as f:
        json.dump(traits_payload, f, indent=2)

    synthetic_source_path = synthetic_path_for_job(job_id)
    synthetic_rows_exported = 0
    synthetic_available = synthetic_source_path.exists()

    if synthetic_available:
        synthetic_df = pd.read_csv(synthetic_source_path)
        synthetic_df = clean_for_export(synthetic_df)
        synthetic_df = synthetic_df.head(seed_limit)
        synthetic_df.to_csv(seed_path, index=False)
        synthetic_rows_exported = int(len(synthetic_df))
    else:
        # Still create an empty seed file so export package is complete.
        pd.DataFrame().to_csv(seed_path, index=False)

    manifest = {
        "export_id": export_id,
        "cohort_id": cohort_id,
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "approval_status": approval_status,
        "export_type": "meta_safe_seed_package",
        "meta_destination": "Meta Custom Audience / Advantage+ seed preparation",
        "files": {
            "aggregated_traits": str(traits_path),
            "synthetic_seed_csv": str(seed_path),
            "manifest": str(manifest_path)
        },
        "privacy_guarantees": {
            "contains_raw_email": False,
            "contains_raw_phone": False,
            "contains_device_id": False,
            "contains_individual_user_data": False,
            "aggregated_traits_only": True,
            "synthetic_seed_profiles": True,
            "requires_manual_approval_before_upload": True
        },
        "synthetic_seed": {
            "source_path": str(synthetic_source_path),
            "available": synthetic_available,
            "rows_exported": synthetic_rows_exported,
            "seed_limit": seed_limit
        },
        "quality": cohort.get("quality", {}),
        "notes": [
            "This v1 export prepares a safe seed package but does not push directly to Meta.",
            "Manual approval is required before any external ad platform upload.",
            "Only aggregated traits and synthetic seed profiles are exported."
        ]
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    EXPORT_STATUS[export_id] = {
        "status": "created",
        "approval_status": approval_status,
        "cohort_id": cohort_id,
        "job_id": job_id,
        "manifest_path": str(manifest_path)
    }

    return {
        "status": "completed",
        "message": "Meta-safe export package created",
        "export_id": export_id,
        "cohort_id": cohort_id,
        "job_id": job_id,
        "approval_status": approval_status,
        "export_type": "meta_safe_seed_package",
        "aggregated_traits_path": str(traits_path),
        "synthetic_seed_path": str(seed_path),
        "manifest_path": str(manifest_path),
        "synthetic_rows_exported": synthetic_rows_exported,
        "privacy_guarantees": manifest["privacy_guarantees"]
    }


def get_export_status(export_id: str) -> Dict[str, Any]:
    """
    Returns export status.
    """
    if export_id in EXPORT_STATUS:
        return EXPORT_STATUS[export_id]

    # If server restarted, recover status by scanning manifest files.
    for manifest_path in EXPORT_DIR.glob(f"{export_id}/*_export_manifest.json"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        return {
            "status": "created",
            "approval_status": manifest.get("approval_status"),
            "cohort_id": manifest.get("cohort_id"),
            "job_id": manifest.get("job_id"),
            "manifest_path": str(manifest_path)
        }

    return {
        "status": "not_found",
        "message": f"Export not found: {export_id}"
    }
