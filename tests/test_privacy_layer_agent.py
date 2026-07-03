from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agents.privacy_layer_agent import PrivacyLayerAgent
from app.agents.synthetic_engine_agent import SyntheticEngineAgent


def sample_raw_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "maid": "user-a",
                "location_name": "Montreal",
                "primary_poi_type": "Restaurant",
                "created_at": "2026-07-01T18:00:00Z",
                "maid_count": 600,
                "lat": 45.5017,
                "lng": -73.5673,
            },
            {
                "maid": "user-b",
                "location_name": "Montreal",
                "primary_poi_type": "Restaurant",
                "created_at": "2026-07-01T19:00:00Z",
                "maid_count": 700,
                "lat": 45.5018,
                "lng": -73.5674,
            },
            {
                "maid": "user-c",
                "location_name": "Toronto",
                "primary_poi_type": "Gym",
                "created_at": "2026-07-01T07:00:00Z",
                "maid_count": 300,
                "lat": 43.6532,
                "lng": -79.3832,
            },
        ]
    )


def test_privacy_layer_generates_safe_feature_table(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRIVACY_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    agent = PrivacyLayerAgent()

    result = agent.run(
        data=sample_raw_observations(),
        output_dir=tmp_path / "privacy",
        k_min=1000,
        epsilon=1.0,
        run_id="privacy_test_success",
        hash_secret="test_secret",
    )

    assert result["status"] == "completed"
    assert result["output_safe_cohorts"] == 1
    assert result["blocked_cohorts"] == 1
    assert result["differential_privacy"]["privacy_budget_recorded"] is True
    assert result["export_safety"]["raw_maids_exported"] is False
    assert result["export_safety"]["raw_lat_lng_exported"] is False
    assert result["export_safety"]["individual_user_data_exported"] is False

    feature_path = Path(result["outputs"]["clean_feature_table"])
    privacy_report_path = Path(result["outputs"]["privacy_report"])
    lineage_path = Path(result["outputs"]["lineage_report"])
    schema_path = Path(result["outputs"]["input_schema"])

    assert feature_path.exists()
    assert privacy_report_path.exists()
    assert lineage_path.exists()
    assert schema_path.exists()

    df = pd.read_csv(feature_path)
    assert len(df) == 1
    assert "maid" not in df.columns
    assert "lat" not in df.columns
    assert "lng" not in df.columns
    assert "hashed" not in " ".join(df.columns).lower()
    assert df["privacy_status"].iloc[0] == "passed"
    assert df["noisy_maid_volume"].iloc[0] >= 1000
    assert df["quality_score"].between(0, 1).all()

    report = json.loads(privacy_report_path.read_text())
    assert report["hashing"]["raw_identifiers_exported"] is False
    assert report["hashing"]["hashed_identifiers_exported"] is False


def test_privacy_layer_blocks_secret_like_values(tmp_path: Path):
    df = sample_raw_observations()
    df.loc[0, "location_name"] = "postgresql://user:pass@host/db"

    agent = PrivacyLayerAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            data=df,
            output_dir=tmp_path,
            k_min=1000,
            epsilon=1.0,
            run_id="privacy_secret_block",
            hash_secret="test_secret",
        )

    assert "secret-like value" in str(exc.value).lower()


def test_privacy_layer_rejects_invalid_epsilon(tmp_path: Path):
    agent = PrivacyLayerAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            data=sample_raw_observations(),
            output_dir=tmp_path,
            k_min=1000,
            epsilon=0,
            run_id="privacy_bad_epsilon",
            hash_secret="test_secret",
        )

    assert "epsilon" in str(exc.value).lower()


def test_privacy_layer_rejects_when_no_cohorts_pass_k(tmp_path: Path):
    agent = PrivacyLayerAgent()

    with pytest.raises(ValueError) as exc:
        agent.run(
            data=sample_raw_observations(),
            output_dir=tmp_path,
            k_min=10000,
            epsilon=1.0,
            run_id="privacy_no_k_pass",
            hash_secret="test_secret",
        )

    assert "no cohorts passed" in str(exc.value).lower()


def test_privacy_output_can_feed_synthetic_agent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRIVACY_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("APPROVAL_DIR", str(tmp_path / "approvals"))
    monkeypatch.setenv("K_ANONYMITY_MIN", "1000")

    privacy_agent = PrivacyLayerAgent()

    privacy_result = privacy_agent.run(
        data=sample_raw_observations(),
        output_dir=tmp_path / "privacy",
        k_min=1000,
        epsilon=1.0,
        run_id="privacy_to_synthetic",
        hash_secret="test_secret",
    )

    safe_cohorts = pd.read_csv(privacy_result["outputs"]["clean_feature_table"])

    synthetic_agent = SyntheticEngineAgent()

    synthetic_result = synthetic_agent.generate(
        cohorts=safe_cohorts,
        output_dir=tmp_path / "synthetic",
        synthetic_rows=20,
        engine_requested="dp_aggregate",
        epsilon=1.0,
        production_mode=True,
        allow_fallback=False,
        run_id="privacy_to_synthetic",
    )

    assert synthetic_result["status"] == "completed"
    assert synthetic_result["engine_used"] == "DPAggregateCohortSynthesizer"
    assert synthetic_result["privacy_budget_recorded"] is True

    synthetic_df = pd.read_csv(synthetic_result["outputs"]["synthetic_csv"])
    assert len(synthetic_df) == 20
    assert "maid" not in synthetic_df.columns
    assert "lat" not in synthetic_df.columns
    assert "lng" not in synthetic_df.columns
    assert synthetic_df["approval_status"].iloc[0] == "pending_approval"
