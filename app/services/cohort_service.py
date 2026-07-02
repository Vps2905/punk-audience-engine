import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime

import pandas as pd

from app.services.embedding_service import search_similar_audiences
from app.services.vector_store_service import load_vector_store


COHORT_DIR = Path("data/cohorts")
COHORT_DIR.mkdir(parents=True, exist_ok=True)


def cohort_path(cohort_id: str) -> Path:
    return COHORT_DIR / f"{cohort_id}.json"


def remove_unsafe_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures cohort output does not expose individual-level data.
    We only allow aggregated/safe fields.
    """
    blocked_keywords = [
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

    safe_record = {}

    for key, value in record.items():
        key_lower = str(key).lower()

        if any(blocked in key_lower for blocked in blocked_keywords):
            continue

        safe_record[key] = value

    return safe_record


def calculate_quality_score(
    cohort_records: List[Dict[str, Any]],
    min_safe_size: int = 1000
) -> Dict[str, Any]:
    """
    Calculates simple cohort quality.

    Score parts:
    - Size score: bigger audience is better
    - Coherence score: similarity score / trait consistency
    - Privacy score: all rows passed privacy
    """
    if not cohort_records:
        return {
            "quality_score": 0,
            "size_score": 0,
            "coherence_score": 0,
            "privacy_score": 0,
            "total_size": 0,
            "status": "empty"
        }

    total_size = 0
    similarity_scores = []
    privacy_passed = True

    for record in cohort_records:
        count_value = record.get("noisy_count", record.get("cohort_size", 0))

        try:
            total_size += int(count_value)
        except Exception:
            pass

        if "similarity_score" in record:
            similarity_scores.append(float(record["similarity_score"]))

        if record.get("privacy_status") != "passed":
            privacy_passed = False

    # Size score max 40
    size_score = min((total_size / min_safe_size) * 40, 40)

    # Coherence score max 40
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        coherence_score = max(min(avg_similarity * 40, 40), 0)
    else:
        coherence_score = 25

    # Privacy score max 20
    privacy_score = 20 if privacy_passed else 0

    quality_score = round(size_score + coherence_score + privacy_score, 2)

    if quality_score >= 75:
        status = "strong"
    elif quality_score >= 50:
        status = "usable"
    else:
        status = "needs_more_data"

    return {
        "quality_score": quality_score,
        "size_score": round(size_score, 2),
        "coherence_score": round(coherence_score, 2),
        "privacy_score": round(privacy_score, 2),
        "total_size": total_size,
        "status": status
    }


def extract_aggregated_traits(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a summary of common cohort traits.
    """
    trait_fields = [
        "city",
        "interest",
        "visit_time",
        "affinity",
        "age_group",
        "cluster_id"
    ]

    traits = {}

    for field in trait_fields:
        values = []

        for record in records:
            if field in record and record[field] is not None:
                values.append(str(record[field]))

        if values:
            counts = pd.Series(values).value_counts().head(5).to_dict()
            traits[field] = counts

    return traits


def create_cohort(
    job_id: str,
    name: str,
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Creates a privacy-safe cohort from either:
    - semantic query using vector search
    - metadata filters
    """
    cohort_id = f"cohort_{uuid4().hex[:12]}"
    matched_records: List[Dict[str, Any]] = []

    if query:
        search_result = search_similar_audiences(
            job_id=job_id,
            query=query,
            top_k=top_k
        )
        matched_records = search_result["results"]
    else:
        store = load_vector_store(job_id)
        matched_records = store["metadata"]

    if filters:
        filtered = []

        for record in matched_records:
            keep = True

            for key, expected_value in filters.items():
                actual_value = str(record.get(key, "")).lower()
                expected_value = str(expected_value).lower()

                if actual_value != expected_value:
                    keep = False
                    break

            if keep:
                filtered.append(record)

        matched_records = filtered

    safe_records = [remove_unsafe_fields(record) for record in matched_records]

    quality = calculate_quality_score(safe_records)
    aggregated_traits = extract_aggregated_traits(safe_records)

    cohort_doc = {
        "cohort_id": cohort_id,
        "name": name,
        "job_id": job_id,
        "source_query": query,
        "filters": filters or {},
        "created_at": datetime.utcnow().isoformat() + "Z",
        "privacy_mode": "aggregated_only",
        "total_records": len(safe_records),
        "quality": quality,
        "aggregated_traits": aggregated_traits,
        "records": safe_records
    }

    path = cohort_path(cohort_id)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cohort_doc, f, indent=2)

    return {
        "status": "completed",
        "message": "Cohort created",
        "cohort_id": cohort_id,
        "cohort_path": str(path),
        "name": name,
        "job_id": job_id,
        "privacy_mode": "aggregated_only",
        "quality": quality,
        "aggregated_traits": aggregated_traits,
        "record_count": len(safe_records)
    }


def get_cohort(cohort_id: str) -> Dict[str, Any]:
    """
    Reads saved cohort.
    """
    path = cohort_path(cohort_id)

    if not path.exists():
        return {
            "status": "not_found",
            "message": f"Cohort not found: {cohort_id}"
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
