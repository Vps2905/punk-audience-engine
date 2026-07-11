import pandas as pd

from app.services import all_safe_cohort_embedding_service as service_module
from app.services.all_safe_cohort_embedding_service import AllSafeCohortEmbeddingService


def test_all_safe_embeddings_use_postgres_store_without_local_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")

    captured = {}

    def fake_embed_records(job_id, records):
        captured["job_id"] = job_id
        captured["records"] = records
        return {
            "job_id": job_id,
            "status": "completed",
            "rows_embedded": len(records),
            "embedding_backend": "sklearn_hashing",
            "model_name": "sklearn_hashing_vectorizer",
            "dimension": 384,
            "vectors_path": f"postgres://audience_vectors?job_id={job_id}",
            "metadata_path": f"postgres://audience_vectors.metadata_json?job_id={job_id}",
            "model_path": f"postgres://audience_vector_models?job_id={job_id}",
        }

    monkeypatch.setattr(service_module, "embed_records", fake_embed_records)

    df = pd.DataFrame(
        [
            {
                "location_name": "Montreal",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "quality_score": 0.91,
                "privacy_status": "safe",
            }
        ]
    )

    manifest = AllSafeCohortEmbeddingService().build_index(
        safe_cohorts=df,
        output_dir=tmp_path,
        job_id="v2_all_safe_test",
    )

    assert captured["job_id"] == "v2_all_safe_test"
    assert captured["records"][0]["vector_index"] == 0
    assert captured["records"][0]["embedding_text"]
    assert manifest["embedding_store"] == "postgres"
    assert manifest["output_vectors"].startswith("postgres://")
    assert manifest["vector_count"] == 1
    assert manifest["vector_dimension"] == 384
    assert not (tmp_path / "all_safe_cohort_vectors.npy").exists()
    assert (tmp_path / "all_safe_cohort_embedding_manifest.json").exists()
