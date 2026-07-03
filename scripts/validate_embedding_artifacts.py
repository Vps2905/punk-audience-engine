from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BLOCKED_COLUMNS = [
    "maid",
    "raw_maid",
    "device_id",
    "email",
    "phone",
    "lat",
    "lng",
    "latitude",
    "longitude",
    "raw_observation",
    "hashed",
]

SAFE_AGGREGATE_COLUMNS = {
    "total_maid_volume",
    "noisy_maid_volume",
    "safe_maid_volume",
    "total_observations",
    "observation_count",
    "observations_count",
    "safe_observation_count",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_embedding_artifacts.py <embedding_output_dir>")
        return 2

    base = Path(sys.argv[1])

    vector_path = base / "cohort_vectors.npy"
    metadata_path = base / "cohort_metadata.csv"
    preview_path = base / "vector_preview.csv"
    manifest_path = base / "embedding_manifest.json"
    similarity_path = base / "similarity_search_demo.json"

    for path in [vector_path, metadata_path, preview_path, manifest_path, similarity_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    vectors = np.load(vector_path)
    metadata = pd.read_csv(metadata_path)
    preview = pd.read_csv(preview_path)
    manifest = json.loads(manifest_path.read_text())
    similarity = json.loads(similarity_path.read_text())

    assert manifest["status"] == "completed"
    assert manifest["vector_count"] == vectors.shape[0]
    assert manifest["vector_dimension"] == vectors.shape[1]
    assert manifest["metadata_rows"] == len(metadata)
    assert len(metadata) == vectors.shape[0]
    assert vectors.ndim == 2
    assert vectors.shape[0] > 0
    assert vectors.shape[1] > 0
    assert np.isfinite(vectors).all(), "Vectors contain NaN or infinite values"

    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(norms > 0), "One or more vectors has zero norm"

    assert manifest["raw_maids_exported"] is False
    assert manifest["raw_observations_exported"] is False
    assert manifest["raw_lat_lng_exported"] is False
    assert manifest["raw_email_exported"] is False
    assert manifest["raw_phone_exported"] is False
    assert manifest["individual_user_data_exported"] is False

    for col in metadata.columns:
        lower = col.lower()

        if lower in SAFE_AGGREGATE_COLUMNS:
            continue

        for blocked in BLOCKED_COLUMNS:
            assert blocked not in lower, f"Blocked metadata column leaked: {col}"

    assert "trait_text" in metadata.columns
    assert metadata["trait_text"].astype(str).str.len().gt(0).all()

    assert len(preview) > 0
    assert "vector_norm" in preview.columns
    assert preview["vector_norm"].between(0.99, 1.01).all()

    assert similarity["raw_identifiers_returned"] is False
    assert len(similarity["results"]) > 0

    print("Embedding artifact validation passed ✅")
    print("Vectors:", vectors.shape[0])
    print("Dimension:", vectors.shape[1])
    print("Provider:", manifest["embedding_provider"])
    print("Similarity results:", len(similarity["results"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
