import numpy as np
import pandas as pd

from app.agents import audience_intelligence_orchestrator_agent as orchestrator_module
from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
)


def test_main_embedding_flow_uses_postgres_384_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")

    selected = pd.DataFrame(
        [
            {
                "location_name": "Montreal",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "quality_score": 0.91,
                "privacy_status": "passed",
            },
            {
                "location_name": "Toronto",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "quality_score": 0.82,
                "privacy_status": "passed",
            },
        ]
    )

    captured = {}

    def fake_embed_records(*, job_id, records):
        captured["embedding_job_id"] = job_id
        captured["records"] = records

        return {
            "status": "completed",
            "job_id": job_id,
            "rows_embedded": 2,
            "embedding_backend": "sklearn_hashing",
            "model_name": "sklearn_hashing_vectorizer",
            "dimension": 384,
            "vectors_path": f"postgres://audience_vectors?job_id={job_id}",
            "metadata_path": (
                f"postgres://audience_vectors.metadata_json?job_id={job_id}"
            ),
            "model_path": f"postgres://audience_vector_models?job_id={job_id}",
        }

    def fake_load_vector_store(job_id):
        captured["loaded_job_id"] = job_id

        metadata = selected.reset_index(drop=True).to_dict(orient="records")
        for index, row in enumerate(metadata):
            row["vector_index"] = index
            row["trait_text"] = (
                f"{row['location_name']} "
                f"{row['primary_poi_type']} "
                f"{row['created_day_part']}"
            )

        return {
            "vectors": np.ones((2, 384), dtype=float),
            "metadata": metadata,
            "model_info": {
                "backend": "sklearn_hashing",
                "dimension": 384,
            },
        }

    class FakeCohortManagementAgent:
        def run(self, **kwargs):
            captured["cohort_metadata"] = kwargs["metadata"]
            captured["cohort_vectors"] = kwargs["vectors"]
            captured["cohort_run_id"] = kwargs["run_id"]

            return {
                "status": "completed",
                "managed_cohorts": 2,
                "cluster_count": 2,
                "export_ready_cohorts": 2,
                "outputs": {
                    "top_cohorts": str(tmp_path / "top_cohorts.csv"),
                    "lookalike_cohorts": str(tmp_path / "lookalikes.csv"),
                },
            }

    class ForbiddenLegacyEmbeddingAgent:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "Legacy TF-IDF embedding agent must not run with Postgres backend."
            )

    monkeypatch.setattr(
        orchestrator_module,
        "embed_records",
        fake_embed_records,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_vector_store",
        fake_load_vector_store,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "CohortManagementAgent",
        FakeCohortManagementAgent,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "EmbeddingFeatureStoreAgent",
        ForbiddenLegacyEmbeddingAgent,
    )

    embedding_result, cohort_result = (
        AudienceIntelligenceOrchestratorAgent()
        ._run_embedding_and_cohort_management(
            selected_cohorts=selected,
            embedding_dir=tmp_path / "03_embeddings",
            cohort_dir=tmp_path / "04_cohort_management",
            run_id="production_run_123",
            min_export_quality=0.25,
        )
    )

    assert embedding_result["status"] == "completed"
    assert embedding_result["embedding_provider"] == "sklearn_hashing"
    assert embedding_result["vector_backend"] == "postgres_array"
    assert embedding_result["vector_count"] == 2
    assert embedding_result["vector_dimension"] == 384
    assert embedding_result["outputs"]["cohort_vectors"].startswith(
        "postgres://audience_vectors"
    )

    assert captured["embedding_job_id"] == "production_run_123_embedding"
    assert captured["loaded_job_id"] == "production_run_123_embedding"
    assert captured["cohort_vectors"].shape == (2, 384)
    assert len(captured["cohort_metadata"]) == 2
    assert cohort_result["status"] == "completed"

    assert not (tmp_path / "03_embeddings" / "cohort_vectors.npy").exists()
