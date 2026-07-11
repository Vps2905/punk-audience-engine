import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.core.production_guardrails import require_local_file_storage_allowed


VECTOR_DIR = Path("data/vectors")


def _ensure_local_vector_storage_allowed(feature_name: str) -> None:
    require_local_file_storage_allowed(feature_name)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def get_vector_paths(job_id: str) -> Dict[str, Path]:
    """
    Returns all file paths for one embedding job.
    """
    _ensure_local_vector_storage_allowed("local vector store paths")
    return {
        "vectors": VECTOR_DIR / f"{job_id}_vectors.npy",
        "metadata": VECTOR_DIR / f"{job_id}_metadata.json",
        "model": VECTOR_DIR / f"{job_id}_model.json",
    }


def save_vector_store(
    job_id: str,
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    model_info: Dict[str, Any]
) -> Dict[str, str]:
    """
    Saves vectors and metadata locally.
    """
    paths = get_vector_paths(job_id)

    np.save(paths["vectors"], vectors)

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(paths["model"], "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    return {
        "vectors_path": str(paths["vectors"]),
        "metadata_path": str(paths["metadata"]),
        "model_path": str(paths["model"]),
    }


def load_vector_store(job_id: str) -> Dict[str, Any]:
    """
    Loads vectors and metadata for a job.
    """
    paths = get_vector_paths(job_id)

    if not paths["vectors"].exists():
        raise FileNotFoundError(f"Vector file not found for job_id={job_id}")

    vectors = np.load(paths["vectors"])

    with open(paths["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(paths["model"], "r", encoding="utf-8") as f:
        model_info = json.load(f)

    return {
        "vectors": vectors,
        "metadata": metadata,
        "model_info": model_info,
    }


def similarity_search(
    job_id: str,
    query_vector: np.ndarray,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Compares query vector with stored audience vectors using cosine similarity.
    """
    store = load_vector_store(job_id)
    vectors = store["vectors"]
    metadata = store["metadata"]

    if vectors.size == 0:
        return []

    query_vector = query_vector.reshape(1, -1)
    scores = cosine_similarity(query_vector, vectors)[0]

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        item = metadata[int(idx)].copy()
        item["similarity_score"] = float(scores[int(idx)])
        results.append(item)

    return results


def save_cluster_output(job_id: str, clustered_df: pd.DataFrame) -> str:
    """
    Saves clustering output.
    """
    _ensure_local_vector_storage_allowed("local cluster output")
    output_path = VECTOR_DIR / f"{job_id}_clusters.csv"
    clustered_df.to_csv(output_path, index=False)
    return str(output_path)
