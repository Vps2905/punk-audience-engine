import shutil
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

import pandas as pd
from fastapi import UploadFile

from app.services.privacy_service import apply_privacy_pipeline
from app.services.lineage_service import write_lineage


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

JOB_STATUS = {}


def ingest_csv(file: UploadFile, k_min: int = 1000, epsilon: float = 1.0) -> Dict[str, Any]:
    job_id = f"job_{uuid4().hex[:12]}"

    JOB_STATUS[job_id] = {
        "status": "started",
        "message": "Ingestion started"
    }

    safe_filename = file.filename.replace("/", "_").replace(" ", "_")
    raw_path = RAW_DIR / f"{job_id}_{safe_filename}"

    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(raw_path)

        clean_df = apply_privacy_pipeline(
            df=df,
            k_min=k_min,
            epsilon=epsilon
        )

        processed_path = PROCESSED_DIR / f"{job_id}_clean_features.csv"
        clean_df.to_csv(processed_path, index=False)

        lineage = {
            "source": str(raw_path),
            "input_rows": int(len(df)),
            "output_rows": int(len(clean_df)),
            "privacy_config": {
                "k_min": k_min,
                "epsilon": epsilon
            },
            "transformations": [
                "normalize_columns",
                "hash_pii_columns",
                "age_to_age_group",
                "aggregate_traits",
                "k_anonymity_filter",
                "laplace_dp_noise"
            ],
            "outputs": {
                "processed_features": str(processed_path)
            }
        }

        lineage_path = write_lineage(job_id, lineage)

        JOB_STATUS[job_id] = {
            "status": "completed",
            "message": "Ingestion and privacy processing completed",
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "lineage_path": str(lineage_path),
            "input_rows": int(len(df)),
            "output_rows": int(len(clean_df))
        }

        return {
            "job_id": job_id,
            **JOB_STATUS[job_id]
        }

    except Exception as e:
        JOB_STATUS[job_id] = {
            "status": "failed",
            "message": str(e),
            "raw_path": str(raw_path)
        }

        return {
            "job_id": job_id,
            **JOB_STATUS[job_id]
        }


def get_job_status(job_id: str) -> Dict[str, Any]:
    return JOB_STATUS.get(
        job_id,
        {
            "status": "not_found",
            "message": "Job ID not found"
        }
    )
