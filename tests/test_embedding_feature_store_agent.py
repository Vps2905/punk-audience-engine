from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents.embedding_feature_store_agent import EmbeddingFeatureStoreAgent


def sample_safe_cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "0_7d",
                "sessions": 10,
                "total_maid_volume": 2000,
                "noisy_maid_volume": 2001,
                "total_observations": 12,
                "quality_score": 0.91,
                "privacy_status": "passed",
                "trait_text": "location montreal | poi restaurant | daypart evening | quality 0.91",
            },
            {
                "location_name": "toronto",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "lookback_bucket": "8_30d",
                "sessions": 8,
                "total_maid_volume": 1500,
                "noisy_maid_volume": 1501,
                "total_observations": 9,
                "quality_score": 0.82,
                "privacy_status": "passed",
                "trait_text": "location toronto | poi gym | daypart morning | quality 0.82",
            },
            {
                "location_name": "new york",
                "primary_poi_type": "food",
                "created_day_part": "evening",
                "lookback_bucket": "0_7d",
                "sessions": 15,
                "total_maid_volume": 3000,
                "noisy_maid_volume": 3002,
                "total_observations": 17,
                "quality_score": 0.95,
                "privacy_status": "passed",
                "trait_text": "location new york | poi food | daypart evening | quality 0.95",
            },
        ]
    )


def test_embedding_agent_builds_safe_vector_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    agent = EmbeddingFeatureStoreAgent()

    result = agent.build(
        cohorts=sample_safe_cohorts(),
        output_dir=tmp_path / "embeddings",
        embedding_provider="sklearn_tfidf",
        run_id="embedding_test_success",
        max_features=64,
    )

    assert result["status"] == "completed"
    assert result["embedding_provider"] == "sklearn_tfidf"
    assert result["vector_count"] == 3
    assert result["vector_dimension"] > 0
    assert result["raw_maids_exported"] is False
    assert result["raw_observations_exported"] is False
    assert result["raw_lat_lng_exported"] is False
    assert result["individual_user_data_exported"] is False

    vector_path = Path(result["outputs"]["cohort_vectors"])
    metadata_path = Path(result["outputs"]["cohort_metadata"])
    manifest_path = Path(result["outputs"]["embedding_manifest"])
    similarity_path = Path(result["outputs"]["similarity_search_demo"])

    assert vector_path.exists()
    assert metadata_path.exists()
    assert manifest_path.exists()
    assert similarity_path.exists()

    vectors = np.load(vector_path)
    metadata = pd.read_csv(metadata_path)
    manifest = json.loads(manifest_path.read_text())

    assert vectors.shape[0] == len(metadata) == 3
    assert vectors.shape[1] == manifest["vector_dimension"]
    assert np.isfinite(vectors).all()

    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(norms > 0)

    blocked_cols = ["maid", "raw_maid", "device_id", "email", "phone", "lat", "lng", "latitude", "longitude"]
    for col in metadata.columns:
        lower = col.lower()
        if lower in {"total_maid_volume", "noisy_maid_volume", "total_observations"}:
            continue
        for blocked in blocked_cols:
            assert blocked not in lower


def test_embedding_agent_blocks_sensitive_columns(tmp_path: Path):
    df = sample_safe_cohorts()
    df["device_id"] = ["a", "b", "c"]

    agent = EmbeddingFeatureStoreAgent()

    with pytest.raises(ValueError) as exc:
        agent.build(
            cohorts=df,
            output_dir=tmp_path,
            embedding_provider="sklearn_tfidf",
            run_id="embedding_sensitive_block",
        )

    assert "blocked sensitive column" in str(exc.value).lower()


def test_embedding_agent_rejects_invalid_provider(tmp_path: Path):
    agent = EmbeddingFeatureStoreAgent()

    with pytest.raises(ValueError) as exc:
        agent.build(
            cohorts=sample_safe_cohorts(),
            output_dir=tmp_path,
            embedding_provider="bad_provider",
            run_id="embedding_bad_provider",
        )

    assert "embedding_provider" in str(exc.value)


def test_embedding_agent_similarity_search_returns_safe_results(tmp_path: Path):
    agent = EmbeddingFeatureStoreAgent()

    result = agent.build(
        cohorts=sample_safe_cohorts(),
        output_dir=tmp_path / "embeddings",
        embedding_provider="sklearn_tfidf",
        run_id="embedding_similarity",
        max_features=64,
    )

    vectors = np.load(result["outputs"]["cohort_vectors"])
    metadata = pd.read_csv(result["outputs"]["cohort_metadata"])

    search = agent.search_similar(
        query_text="restaurant evening food audience",
        metadata=metadata,
        vectors=vectors,
        top_k=2,
    )

    assert search["raw_identifiers_returned"] is False
    assert search["top_k"] == 2
    assert len(search["results"]) == 2
    assert "similarity_score" in search["results"][0]


def test_embedding_agent_external_service_fails_until_configured(tmp_path: Path):
    agent = EmbeddingFeatureStoreAgent()

    with pytest.raises(NotImplementedError) as exc:
        agent.build(
            cohorts=sample_safe_cohorts(),
            output_dir=tmp_path,
            embedding_provider="external_embedding_service",
            run_id="embedding_external_missing",
        )

    assert "reserved" in str(exc.value).lower()
