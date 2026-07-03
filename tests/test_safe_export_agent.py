from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agents.safe_export_agent import SafeExportAgent


def sample_top_cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cohort_index": 1,
                "cluster_id": 0,
                "cluster_size": 3,
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "0_7d",
                "sessions": 10,
                "total_maid_volume": 3000,
                "noisy_maid_volume": 3001,
                "quality_score": 0.91,
                "cluster_coherence_score": 0.88,
                "management_quality_score": 0.89,
                "export_ready": True,
                "recommendation_reason": "cluster 0 | quality 0.89",
                "trait_text": "location montreal | poi restaurant | daypart evening",
            },
            {
                "cohort_index": 2,
                "cluster_id": 0,
                "cluster_size": 3,
                "location_name": "toronto",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "8_30d",
                "sessions": 8,
                "total_maid_volume": 2200,
                "noisy_maid_volume": 2201,
                "quality_score": 0.82,
                "cluster_coherence_score": 0.83,
                "management_quality_score": 0.80,
                "export_ready": True,
                "recommendation_reason": "cluster 0 | quality 0.80",
                "trait_text": "location toronto | poi restaurant | daypart evening",
            },
            {
                "cohort_index": 3,
                "cluster_id": 1,
                "cluster_size": 2,
                "location_name": "new york",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "lookback_bucket": "0_7d",
                "sessions": 5,
                "total_maid_volume": 1200,
                "noisy_maid_volume": 1201,
                "quality_score": 0.55,
                "cluster_coherence_score": 0.70,
                "management_quality_score": 0.20,
                "export_ready": False,
                "recommendation_reason": "below export threshold",
                "trait_text": "location new york | poi gym | daypart morning",
            },
        ]
    )


def sample_lookalikes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed_cohort_index": 1,
                "lookalike_cohort_index": 2,
                "seed_cluster_id": 0,
                "lookalike_cluster_id": 0,
                "seed_location_name": "montreal",
                "seed_primary_poi_type": "restaurant",
                "lookalike_location_name": "toronto",
                "lookalike_primary_poi_type": "restaurant",
                "similarity_score": 0.93,
                "recommendation_reason": "similar restaurant evening cohort",
            },
            {
                "seed_cohort_index": 1,
                "lookalike_cohort_index": 3,
                "seed_cluster_id": 0,
                "lookalike_cluster_id": 1,
                "seed_location_name": "montreal",
                "seed_primary_poi_type": "restaurant",
                "lookalike_location_name": "new york",
                "lookalike_primary_poi_type": "gym",
                "similarity_score": 0.21,
                "recommendation_reason": "not exported because lookalike cohort not export-ready",
            },
        ]
    )


def test_safe_export_builds_approval_gated_package(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    agent = SafeExportAgent()

    result = agent.run(
        top_cohorts=sample_top_cohorts(),
        lookalikes=sample_lookalikes(),
        output_dir=tmp_path / "safe_export",
        run_id="safe_export_success",
        audience_namespace="punk_audience",
        approval_required=True,
        min_management_quality=0.25,
    )

    assert result["status"] == "completed"
    assert result["approval_status"] == "pending_approval"
    assert result["downstream_export_enabled"] is False
    assert result["export_blocked_until_approved"] is True
    assert result["exported_cohorts"] == 2
    assert result["raw_maids_exported"] is False
    assert result["hashed_identifiers_exported"] is False
    assert result["individual_user_data_exported"] is False

    cohorts_path = Path(result["outputs"]["safe_export_cohorts"])
    lookalikes_path = Path(result["outputs"]["safe_export_lookalikes"])
    payload_path = Path(result["outputs"]["safe_export_payload"])
    approval_path = Path(result["outputs"]["export_approval_request"])
    manifest_path = Path(result["outputs"]["safe_export_manifest"])

    assert cohorts_path.exists()
    assert lookalikes_path.exists()
    assert payload_path.exists()
    assert approval_path.exists()
    assert manifest_path.exists()

    cohorts = pd.read_csv(cohorts_path)
    lookalikes = pd.read_csv(lookalikes_path)
    payload = json.loads(payload_path.read_text())
    approval = json.loads(approval_path.read_text())

    assert len(cohorts) == 2
    assert cohorts["export_status"].eq("pending_approval").all()
    assert cohorts["privacy_mode"].eq("aggregated_dp_safe").all()
    assert cohorts["data_safety_status"].eq("safe_aggregated_no_raw_identifiers").all()
    assert cohorts["export_cohort_id"].astype(str).str.startswith("punk_audience_").all()

    assert len(lookalikes) == 1
    assert payload["downstream_export_enabled"] is False
    assert approval["export_blocked_until_approved"] is True


def test_safe_export_blocks_sensitive_columns(tmp_path: Path):
    df = sample_top_cohorts()
    df["raw_maid"] = ["abc"] * len(df)

    agent = SafeExportAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            top_cohorts=df,
            output_dir=tmp_path,
            run_id="safe_export_sensitive_block",
        )

    assert "blocked sensitive column" in str(exc.value).lower()


def test_safe_export_rejects_when_no_export_ready_cohorts(tmp_path: Path):
    df = sample_top_cohorts()
    df["export_ready"] = False

    agent = SafeExportAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            top_cohorts=df,
            output_dir=tmp_path,
            run_id="safe_export_no_ready",
        )

    assert "no export-ready cohorts" in str(exc.value).lower()


def test_safe_export_run_from_artifacts(tmp_path: Path):
    top_path = tmp_path / "top_cohorts.csv"
    lookalikes_path = tmp_path / "lookalikes.csv"

    sample_top_cohorts().to_csv(top_path, index=False)
    sample_lookalikes().to_csv(lookalikes_path, index=False)

    agent = SafeExportAgent()

    result = agent.run_from_artifacts(
        top_cohorts_path=top_path,
        lookalikes_path=lookalikes_path,
        output_dir=tmp_path / "safe_export",
        run_id="safe_export_from_artifacts",
        audience_namespace="punk_audience",
        approval_required=True,
    )

    assert result["status"] == "completed"
    assert result["exported_cohorts"] == 2
    assert result["approval_status"] == "pending_approval"


def test_safe_export_rejects_invalid_quality_threshold(tmp_path: Path):
    agent = SafeExportAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            top_cohorts=sample_top_cohorts(),
            output_dir=tmp_path,
            run_id="safe_export_bad_quality",
            min_management_quality=2.0,
        )

    assert "min_management_quality" in str(exc.value)
