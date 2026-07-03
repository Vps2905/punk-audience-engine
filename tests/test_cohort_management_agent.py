from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents.cohort_management_agent import CohortManagementAgent


def sample_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "0_7d",
                "sessions": 10,
                "total_maid_volume": 3000,
                "noisy_maid_volume": 3001,
                "total_observations": 12,
                "quality_score": 0.91,
                "privacy_status": "passed",
                "trait_text": "location montreal | poi restaurant | daypart evening",
            },
            {
                "location_name": "toronto",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "8_30d",
                "sessions": 9,
                "total_maid_volume": 2500,
                "noisy_maid_volume": 2502,
                "total_observations": 10,
                "quality_score": 0.86,
                "privacy_status": "passed",
                "trait_text": "location toronto | poi restaurant | daypart evening",
            },
            {
                "location_name": "new york",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "lookback_bucket": "0_7d",
                "sessions": 8,
                "total_maid_volume": 2000,
                "noisy_maid_volume": 2001,
                "total_observations": 9,
                "quality_score": 0.79,
                "privacy_status": "passed",
                "trait_text": "location new york | poi gym | daypart morning",
            },
            {
                "location_name": "san francisco",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "lookback_bucket": "8_30d",
                "sessions": 7,
                "total_maid_volume": 1800,
                "noisy_maid_volume": 1801,
                "total_observations": 8,
                "quality_score": 0.74,
                "privacy_status": "passed",
                "trait_text": "location san francisco | poi gym | daypart morning",
            },
            {
                "location_name": "montreal",
                "primary_poi_type": "food",
                "created_day_part": "afternoon",
                "lookback_bucket": "0_7d",
                "sessions": 6,
                "total_maid_volume": 1600,
                "noisy_maid_volume": 1601,
                "total_observations": 7,
                "quality_score": 0.69,
                "privacy_status": "passed",
                "trait_text": "location montreal | poi food | daypart afternoon",
            },
            {
                "location_name": "toronto",
                "primary_poi_type": "food",
                "created_day_part": "afternoon",
                "lookback_bucket": "8_30d",
                "sessions": 5,
                "total_maid_volume": 1300,
                "noisy_maid_volume": 1301,
                "total_observations": 6,
                "quality_score": 0.61,
                "privacy_status": "passed",
                "trait_text": "location toronto | poi food | daypart afternoon",
            },
        ]
    )


def sample_vectors() -> np.ndarray:
    vectors = np.array(
        [
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
            [0.8, 0.2, 0.1, 0.0],
            [0.7, 0.3, 0.1, 0.0],
        ],
        dtype="float32",
    )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def test_cohort_management_builds_clusters_and_lookalikes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    agent = CohortManagementAgent()

    result = agent.run(
        metadata=sample_metadata(),
        vectors=sample_vectors(),
        output_dir=tmp_path / "cohort_management",
        run_id="cohort_management_success",
        min_clusters=2,
        max_clusters=3,
        top_n=4,
        lookalike_top_k=2,
        min_export_quality=0.25,
    )

    assert result["status"] == "completed"
    assert result["managed_cohorts"] == 6
    assert result["cluster_count"] >= 2
    assert result["top_cohorts"] == 4
    assert result["lookalike_pairs"] > 0
    assert result["raw_maids_exported"] is False
    assert result["individual_user_data_exported"] is False

    clusters_path = Path(result["outputs"]["cohort_clusters"])
    top_path = Path(result["outputs"]["top_cohorts"])
    lookalikes_path = Path(result["outputs"]["lookalike_cohorts"])
    report_path = Path(result["outputs"]["cohort_quality_report"])
    manifest_path = Path(result["outputs"]["cohort_management_manifest"])

    assert clusters_path.exists()
    assert top_path.exists()
    assert lookalikes_path.exists()
    assert report_path.exists()
    assert manifest_path.exists()

    clusters = pd.read_csv(clusters_path)
    top = pd.read_csv(top_path)
    lookalikes = pd.read_csv(lookalikes_path)
    manifest = json.loads(manifest_path.read_text())

    assert len(clusters) == 6
    assert len(top) == 4
    assert len(lookalikes) > 0
    assert clusters["management_quality_score"].between(0, 1).all()
    assert clusters["cluster_coherence_score"].between(0, 1).all()
    assert manifest["cluster_count"] == clusters["cluster_id"].nunique()


def test_cohort_management_blocks_sensitive_metadata(tmp_path: Path):
    metadata = sample_metadata()
    metadata["email"] = ["a@test.com"] * len(metadata)

    agent = CohortManagementAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            metadata=metadata,
            vectors=sample_vectors(),
            output_dir=tmp_path,
            run_id="cohort_management_sensitive_block",
        )

    assert "blocked sensitive column" in str(exc.value).lower()


def test_cohort_management_rejects_vector_row_mismatch(tmp_path: Path):
    agent = CohortManagementAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            metadata=sample_metadata(),
            vectors=sample_vectors()[:3],
            output_dir=tmp_path,
            run_id="cohort_management_vector_mismatch",
        )

    assert "row count" in str(exc.value).lower()


def test_cohort_management_run_from_artifacts(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    vectors_path = tmp_path / "vectors.npy"

    sample_metadata().to_csv(metadata_path, index=False)
    np.save(vectors_path, sample_vectors())

    agent = CohortManagementAgent()

    result = agent.run_from_artifacts(
        metadata_path=metadata_path,
        vectors_path=vectors_path,
        output_dir=tmp_path / "cohort_management",
        run_id="cohort_management_from_artifacts",
        min_clusters=2,
        max_clusters=3,
        top_n=3,
        lookalike_top_k=2,
    )

    assert result["status"] == "completed"
    assert result["managed_cohorts"] == 6
    assert result["top_cohorts"] == 3


def test_cohort_management_rejects_invalid_quality_threshold(tmp_path: Path):
    agent = CohortManagementAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            metadata=sample_metadata(),
            vectors=sample_vectors(),
            output_dir=tmp_path,
            run_id="cohort_management_bad_threshold",
            min_export_quality=2.0,
        )

    assert "min_export_quality" in str(exc.value)
