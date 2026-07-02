from typing import Dict, Any
from fastapi import UploadFile

from app.services.ingestion_service import ingest_csv
from app.services.synthetic_service import generate_synthetic_for_job
from app.services.embedding_service import embed_processed_job
from app.services.clustering_service import cluster_job_vectors
from app.services.cohort_service import create_cohort
from app.services.meta_export_service import generate_meta_safe_export


def clean_prompt_to_query(prompt: str) -> str:
    """
    Converts user prompt into an audience query.

    Example:
    'Create fitness evening audience and prepare Meta export'
    becomes:
    'fitness evening'
    """
    text = prompt.lower()

    removable_words = [
        "create",
        "make",
        "build",
        "generate",
        "audience",
        "cohort",
        "segment",
        "prepare",
        "export",
        "meta",
        "safe",
        "for",
        "to",
        "and",
        "please"
    ]

    for word in removable_words:
        text = text.replace(word, " ")

    return " ".join(text.split()) or prompt


def build_cohort_name(prompt: str) -> str:
    """
    Creates readable cohort name from prompt.
    """
    query = clean_prompt_to_query(prompt)
    return f"{query.title()} Audience"


def run_full_audience_pipeline(
    file: UploadFile,
    prompt: str,
    k_min: int = 1000,
    epsilon: float = 1.0,
    synthetic_rows: int = 1000,
    seed_limit: int = 1000
) -> Dict[str, Any]:
    """
    One-click full pipeline.

    This is the product-facing orchestration:
    CSV + prompt -> privacy-safe audience + Meta export.
    """
    ingest_result = ingest_csv(
        file=file,
        k_min=k_min,
        epsilon=epsilon
    )

    if ingest_result.get("status") != "completed":
        return {
            "status": "failed",
            "stage": "ingestion",
            "message": "Ingestion failed",
            "details": ingest_result
        }

    job_id = ingest_result["job_id"]

    synthetic_result = generate_synthetic_for_job(
        job_id=job_id,
        num_rows=synthetic_rows,
        use_sdv=True
    )

    embedding_result = embed_processed_job(job_id)

    cluster_result = cluster_job_vectors(
        job_id=job_id,
        n_clusters=3
    )

    query = clean_prompt_to_query(prompt)
    cohort_name = build_cohort_name(prompt)

    cohort_result = create_cohort(
        job_id=job_id,
        name=cohort_name,
        query=query,
        filters=None,
        top_k=20
    )

    if cohort_result.get("status") != "completed":
        return {
            "status": "failed",
            "stage": "cohort",
            "message": "Cohort creation failed",
            "job_id": job_id,
            "details": cohort_result
        }

    cohort_id = cohort_result["cohort_id"]

    export_result = generate_meta_safe_export(
        cohort_id=cohort_id,
        seed_limit=seed_limit,
        approval_status="pending_approval"
    )

    return {
        "status": "completed",
        "message": "Audience intelligence pipeline completed",
        "input_prompt": prompt,
        "audience_query": query,
        "job_id": job_id,
        "cohort_id": cohort_id,
        "export_id": export_result.get("export_id"),
        "audience_summary": {
            "name": cohort_result.get("name"),
            "quality": cohort_result.get("quality"),
            "aggregated_traits": cohort_result.get("aggregated_traits"),
            "record_count": cohort_result.get("record_count"),
            "privacy_mode": cohort_result.get("privacy_mode")
        },
        "meta_export": {
            "approval_status": export_result.get("approval_status"),
            "export_type": export_result.get("export_type"),
            "synthetic_rows_exported": export_result.get("synthetic_rows_exported"),
            "manifest_path": export_result.get("manifest_path"),
            "synthetic_seed_path": export_result.get("synthetic_seed_path"),
            "aggregated_traits_path": export_result.get("aggregated_traits_path"),
            "privacy_guarantees": export_result.get("privacy_guarantees")
        },
        "internal_pipeline": {
            "ingestion": {
                "input_rows": ingest_result.get("input_rows"),
                "output_rows": ingest_result.get("output_rows"),
                "processed_path": ingest_result.get("processed_path"),
                "lineage_path": ingest_result.get("lineage_path")
            },
            "synthetic": {
                "backend": synthetic_result.get("backend"),
                "rows_generated": synthetic_result.get("num_rows_generated"),
                "safe_for_export_seed": synthetic_result.get("safe_for_export_seed")
            },
            "embeddings": {
                "backend": embedding_result.get("embedding_backend"),
                "dimension": embedding_result.get("dimension"),
                "rows_embedded": embedding_result.get("rows_embedded")
            },
            "clustering": {
                "actual_clusters": cluster_result.get("actual_clusters"),
                "cluster_path": cluster_result.get("cluster_path")
            }
        }
    }
