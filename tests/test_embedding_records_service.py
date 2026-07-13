import numpy as np

from app.services import embedding_service


def test_embed_records_uses_stateless_hashing_backend(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("EMBEDDING_HASHING_FEATURES", "384")

    captured = {}

    def fake_save_vector_store(job_id, vectors, metadata, model_info):
        captured["job_id"] = job_id
        captured["vectors"] = vectors
        captured["metadata"] = metadata
        captured["model_info"] = model_info
        return {
            "vectors_path": "postgres://audience_vectors?job_id=records_job",
            "metadata_path": "postgres://audience_vectors.metadata_json?job_id=records_job",
            "model_path": "postgres://audience_vector_models?job_id=records_job",
        }

    monkeypatch.setattr(embedding_service, "save_vector_store", fake_save_vector_store)

    result = embedding_service.embed_records(
        job_id="records_job",
        records=[
            {
                "location_name": "Montreal",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "quality_score": 0.91,
            }
        ],
    )

    assert result["status"] == "completed"
    assert result["embedding_backend"] == "sklearn_hashing"
    assert result["dimension"] == 384
    assert captured["vectors"].shape == (1, 384)
    assert captured["model_info"]["backend"] == "sklearn_hashing"
    assert "vectorizer_path" not in captured["model_info"]


def test_encode_query_supports_hashing_backend(monkeypatch):
    def fake_load_vector_store(job_id):
        return {
            "vectors": np.asarray([[1.0, 0.0]]),
            "metadata": [{"trait_text": "coffee shop montreal"}],
            "model_info": {
                "backend": "sklearn_hashing",
                "model_name": "sklearn_hashing_vectorizer",
                "dimension": 384,
            },
        }

    monkeypatch.setattr(embedding_service, "load_vector_store", fake_load_vector_store)

    vector = embedding_service.encode_query_for_job("records_job", "coffee shop montreal")

    assert vector.shape == (384,)
    assert np.linalg.norm(vector) > 0


def test_postgres_query_encoding_loads_model_only(
    monkeypatch,
):
    monkeypatch.setenv(
        "VECTOR_BACKEND",
        "postgres_array",
    )

    calls = {
        "model": 0,
        "full_store": 0,
    }

    def fake_model_loader(job_id):
        calls["model"] += 1

        assert job_id == "records_job"

        return {
            "backend": "sklearn_hashing",
            "model_name": (
                "sklearn_hashing_vectorizer"
            ),
            "dimension": 384,
            "vector_dimension": 384,
        }

    def forbidden_full_store(job_id):
        calls["full_store"] += 1

        raise AssertionError(
            "Postgres query encoding must not load "
            "the complete vector store."
        )

    monkeypatch.setattr(
        embedding_service,
        "load_postgres_vector_model_info",
        fake_model_loader,
    )

    monkeypatch.setattr(
        embedding_service,
        "load_vector_store",
        forbidden_full_store,
    )

    vector = embedding_service.encode_query_for_job(
        "records_job",
        "coffee shop visitors in Montreal",
    )

    assert vector.shape == (384,)
    assert calls["model"] == 1
    assert calls["full_store"] == 0
