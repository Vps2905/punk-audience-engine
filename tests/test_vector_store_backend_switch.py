import numpy as np
import pandas as pd

from app.services import vector_store_service


def test_postgres_backend_used_for_save(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")

    called = {}

    def fake_save(job_id, vectors, metadata, model_info):
        called["job_id"] = job_id
        return {
            "vectors_path": "postgres://audience_vectors?job_id=job1",
            "metadata_path": "postgres://metadata",
            "model_path": "postgres://model",
        }

    monkeypatch.setattr(vector_store_service, "save_postgres_vector_store", fake_save)

    result = vector_store_service.save_vector_store(
        job_id="job1",
        vectors=np.asarray([[1.0, 0.0]]),
        metadata=[{"trait_text": "restaurant montreal"}],
        model_info={"backend": "test", "dimension": 2},
    )

    assert called["job_id"] == "job1"
    assert result["vectors_path"].startswith("postgres://")


def test_postgres_backend_used_for_similarity(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")

    called = {}

    def fake_search(job_id, query_vector, top_k):
        called["job_id"] = job_id
        return [{"trait_text": "restaurant montreal", "similarity_score": 0.99}]

    monkeypatch.setattr(vector_store_service, "postgres_similarity_search", fake_search)

    result = vector_store_service.similarity_search(
        job_id="job1",
        query_vector=np.asarray([1.0, 0.0]),
        top_k=1,
    )

    assert called["job_id"] == "job1"
    assert result[0]["similarity_score"] == 0.99


def test_postgres_backend_used_for_cluster_output(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")

    called = {}

    def fake_save_cluster(job_id, clustered_df):
        called["rows"] = len(clustered_df)
        return "postgres://audience_vector_clusters?job_id=job1"

    monkeypatch.setattr(vector_store_service, "save_postgres_cluster_output", fake_save_cluster)

    result = vector_store_service.save_cluster_output(
        job_id="job1",
        clustered_df=pd.DataFrame([{"vector_index": 0, "cluster_id": 1}]),
    )

    assert called["rows"] == 1
    assert result.startswith("postgres://")
