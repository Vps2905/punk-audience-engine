from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.cluster import KMeans

from app.services.vector_store_service import load_vector_store, save_cluster_output


def cluster_job_vectors(job_id: str, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Clusters stored audience vectors.

    For small datasets, automatically reduces cluster count.
    """
    store = load_vector_store(job_id)

    vectors = store["vectors"]
    metadata = store["metadata"]

    if len(metadata) == 0:
        raise ValueError("No metadata records available for clustering.")

    safe_cluster_count = min(n_clusters, len(metadata))

    if safe_cluster_count <= 1:
        cluster_labels = [0 for _ in metadata]
    else:
        model = KMeans(
            n_clusters=safe_cluster_count,
            random_state=42,
            n_init="auto"
        )
        cluster_labels = model.fit_predict(vectors).tolist()

    clustered_records = []

    for idx, record in enumerate(metadata):
        item = record.copy()
        item["cluster_id"] = int(cluster_labels[idx])
        clustered_records.append(item)

    clustered_df = pd.DataFrame(clustered_records)
    cluster_path = save_cluster_output(job_id, clustered_df)

    cluster_summary = (
        clustered_df.groupby("cluster_id")
        .size()
        .reset_index(name="cluster_size")
        .to_dict(orient="records")
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Clustering completed",
        "requested_clusters": n_clusters,
        "actual_clusters": safe_cluster_count,
        "cluster_path": cluster_path,
        "cluster_summary": cluster_summary,
    }
