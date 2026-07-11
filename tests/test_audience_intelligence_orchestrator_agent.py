from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def sample_safe_cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "31_90d",
                "sessions": 13,
                "total_maid_volume": 97501,
                "noisy_maid_volume": 97501,
                "total_observations": 20,
                "privacy_status": "passed",
                "quality_score": 0.43,
                "trait_text": "location montreal | poi restaurant | daypart evening | lookback 31_90d",
            },
            {
                "location_name": "montreal qc",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "31_90d",
                "sessions": 3,
                "total_maid_volume": 82243,
                "noisy_maid_volume": 82243,
                "total_observations": 10,
                "privacy_status": "passed",
                "quality_score": 0.35,
                "trait_text": "location montreal qc | poi restaurant | daypart evening | lookback 31_90d",
            },
            {
                "location_name": "san francisco",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "0_7d",
                "sessions": 5,
                "total_maid_volume": 120000,
                "noisy_maid_volume": 120001,
                "total_observations": 15,
                "privacy_status": "passed",
                "quality_score": 0.50,
                "trait_text": "location san francisco | poi restaurant | daypart evening | lookback 0_7d",
            },
            {
                "location_name": "san francisco",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "lookback_bucket": "0_7d",
                "sessions": 8,
                "total_maid_volume": 150000,
                "noisy_maid_volume": 150001,
                "total_observations": 16,
                "privacy_status": "passed",
                "quality_score": 0.70,
                "trait_text": "location san francisco | poi gym | daypart morning | lookback 0_7d",
            },
        ]
    )


def test_orchestrator_runs_full_prompt_pipeline_from_safe_artifact(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("SYNTHETIC_ENGINE", "dp_aggregate")
    monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "false")
    monkeypatch.setenv("K_ANONYMITY_MIN", "1000")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("PRIVACY_LEDGER_PATH", str(tmp_path / "privacy_budget.jsonl"))

    # This test validates the legacy local artifact flow.
    # Prevent production backend environment variables from leaking in
    # from the developer shell.
    monkeypatch.setenv("VECTOR_BACKEND", "local")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "local")
    monkeypatch.setenv("ALLOW_LOCAL_FILE_STORAGE", "true")

    safe_path = tmp_path / "safe_cohorts.csv"
    sample_safe_cohorts().to_csv(safe_path, index=False)

    agent = AudienceIntelligenceOrchestratorAgent()

    result = agent.run(
        prompt="Build me a high-quality restaurant evening audience for Montreal and San Francisco",
        output_root=tmp_path / "prompt_runs",
        source="safe_artifact",
        safe_cohort_path=safe_path,
        k_min=1000,
        epsilon=1.0,
        synthetic_rows=50,
        max_export_cohorts=10,
        min_export_quality=0.25,
        approval_required=True,
    )

    assert result["status"] == "completed"
    assert result["source_mode"] == "existing_safe_artifact"
    assert result["prompt_selected_cohorts"] >= 2

    assert result["synthetic"]["status"] == "completed"
    assert result["embedding"]["status"] == "completed"
    assert result["cohort_management"]["status"] == "completed"
    assert result["safe_export"]["status"] == "completed"

    assert result["safe_export"]["approval_status"] == "pending_approval"
    assert result["safe_export"]["downstream_export_enabled"] is False
    assert result["privacy_guarantees"]["raw_maids_exported"] is False
    assert result["privacy_guarantees"]["individual_user_data_exported"] is False

    run_dir = Path(result["run_dir"])
    assert (run_dir / "final_prompt_summary.json").exists()
    assert (run_dir / "02_synthetic" / "synthetic_manifest.json").exists()
    assert (run_dir / "03_embeddings" / "embedding_manifest.json").exists()
    assert (run_dir / "04_cohort_management" / "cohort_management_manifest.json").exists()
    assert (run_dir / "05_safe_export" / "safe_export_manifest.json").exists()


def test_orchestrator_prompt_filter_detects_exact_match():
    agent = AudienceIntelligenceOrchestratorAgent()
    cohorts = sample_safe_cohorts()

    selected, report = agent._select_cohorts_for_prompt(
        prompt="restaurant evening audience for Montreal",
        cohorts=cohorts,
    )

    assert len(selected) >= 2
    assert report["filter_mode"] == "location+poi+daypart"
    assert "restaurant" in report["poi_terms_detected"]
    assert "evening" in report["dayparts_detected"]


def test_orchestrator_source_requires_safe_path(tmp_path: Path):
    agent = AudienceIntelligenceOrchestratorAgent()

    try:
        agent.run(
            prompt="restaurant audience",
            output_root=tmp_path,
            source="safe_artifact",
            safe_cohort_path=None,
        )
    except ValueError as exc:
        assert "safe-cohort-path" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing safe-cohort-path")
