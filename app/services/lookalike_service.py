import json
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime

from app.services.cohort_service import get_cohort, remove_unsafe_fields, calculate_quality_score, extract_aggregated_traits
from app.services.embedding_service import search_similar_audiences
from app.core.production_guardrails import (
    require_local_file_storage_allowed,
)


LOOKALIKE_DIR = Path("data/cohorts")


def build_lookalike_query(cohort: Dict[str, Any]) -> str:
    """
    Builds query text from cohort traits.
    """
    if cohort.get("source_query"):
        return cohort["source_query"]

    traits = cohort.get("aggregated_traits", {})
    parts = []

    for field, value_counts in traits.items():
        for value in value_counts.keys():
            parts.append(f"{field} {value}")

    return " ".join(parts)


def create_lookalike(
    cohort_id: str,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Creates lookalike audience from existing cohort.
    """
    require_local_file_storage_allowed(
        "legacy lookalike creation"
    )

    source_cohort = get_cohort(cohort_id)

    if source_cohort.get("status") == "not_found":
        return source_cohort

    job_id = source_cohort["job_id"]
    lookalike_query = build_lookalike_query(source_cohort)

    search_result = search_similar_audiences(
        job_id=job_id,
        query=lookalike_query,
        top_k=top_k
    )

    safe_records = [
        remove_unsafe_fields(record)
        for record in search_result["results"]
    ]

    quality = calculate_quality_score(safe_records)
    aggregated_traits = extract_aggregated_traits(safe_records)

    lookalike_id = f"lookalike_{uuid4().hex[:12]}"

    LOOKALIKE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOOKALIKE_DIR / f"{lookalike_id}.json"

    lookalike_doc = {
        "lookalike_id": lookalike_id,
        "source_cohort_id": cohort_id,
        "job_id": job_id,
        "lookalike_query": lookalike_query,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "privacy_mode": "aggregated_only",
        "quality": quality,
        "aggregated_traits": aggregated_traits,
        "records": safe_records
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookalike_doc, f, indent=2)

    return {
        "status": "completed",
        "message": "Lookalike cohort created",
        "lookalike_id": lookalike_id,
        "source_cohort_id": cohort_id,
        "lookalike_path": str(output_path),
        "privacy_mode": "aggregated_only",
        "quality": quality,
        "aggregated_traits": aggregated_traits,
        "record_count": len(safe_records)
    }
