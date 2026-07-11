from pathlib import Path

import numpy as np
import pandas as pd

from app.agents.cohort_management_agent import CohortManagementAgent
from app.agents.safe_export_agent import SafeExportAgent


def test_cohort_management_can_skip_local_artifacts(tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "location_name": "Montreal",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "quality_score": 0.9,
                "total_maid_volume": 10000,
                "noisy_maid_volume": 9800,
                "privacy_status": "passed",
            },
            {
                "location_name": "Toronto",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "quality_score": 0.8,
                "total_maid_volume": 9000,
                "noisy_maid_volume": 8800,
                "privacy_status": "passed",
            },
        ]
    )

    vectors = np.zeros((2, 384), dtype=float)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0

    output_dir = tmp_path / "cohort_output"

    result = CohortManagementAgent().run(
        metadata=metadata,
        vectors=vectors,
        output_dir=output_dir,
        run_id="memory_cohort_run",
        min_clusters=2,
        max_clusters=2,
        persist_artifacts=False,
    )

    assert result["status"] == "completed"
    assert result["storage_backend"] == "run_history_jsonb"
    assert len(result["records"]["top_cohorts"]) == 2
    assert not output_dir.exists()
    assert result["outputs"]["top_cohorts"].startswith("postgres://")


def test_safe_export_can_build_db_package_without_local_files(tmp_path):
    top_cohorts = pd.DataFrame(
        [
            {
                "cohort_index": 0,
                "cluster_id": 0,
                "cluster_size": 1,
                "location_name": "Montreal",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "lookback_bucket": "30d",
                "sessions": 100,
                "total_maid_volume": 10000,
                "noisy_maid_volume": 9800,
                "quality_score": 0.9,
                "cluster_coherence_score": 0.9,
                "management_quality_score": 0.9,
                "export_ready": True,
                "recommendation_reason": "safe aggregate",
                "trait_text": "Montreal coffee shop evening",
            }
        ]
    )

    output_dir = tmp_path / "safe_export"

    result = SafeExportAgent().run(
        top_cohorts=top_cohorts,
        lookalikes=pd.DataFrame(),
        output_dir=output_dir,
        run_id="memory_safe_export_run",
        approval_required=True,
        persist_artifacts=False,
    )

    assert result["status"] == "completed"
    assert result["storage_backend"] == "run_history_jsonb"
    assert result["approval_status"] == "pending_approval"
    assert result["downstream_export_enabled"] is False
    assert len(result["package"]["cohorts"]) == 1
    assert result["package"]["payload"]
    assert result["package"]["approval_request"]
    assert not output_dir.exists()
    assert result["outputs"]["safe_export_manifest"].startswith("postgres://")
