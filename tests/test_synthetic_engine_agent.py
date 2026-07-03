from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agents.synthetic_engine_agent import SyntheticEngineAgent


def sample_safe_cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "8_30d",
                "sessions": 10,
                "total_maid_volume": 5000,
                "noisy_maid_volume": 5050,
                "quality_score": 0.91,
                "privacy_status": "passed",
                "cluster_id": 1,
                "trait_text": "montreal restaurant evening audience",
            },
            {
                "location_name": "new york",
                "primary_poi_type": "gym",
                "created_day_part": "morning",
                "lookback_bucket": "0_7d",
                "sessions": 8,
                "total_maid_volume": 3200,
                "noisy_maid_volume": 3180,
                "quality_score": 0.87,
                "privacy_status": "passed",
                "cluster_id": 2,
                "trait_text": "new york gym morning audience",
            },
        ]
    )


def test_dev_mode_generates_safe_synthetic_outputs(tmp_path: Path):
    agent = SyntheticEngineAgent()

    result = agent.generate(
        cohorts=sample_safe_cohorts(),
        output_dir=tmp_path,
        synthetic_rows=25,
        engine_requested="auto",
        epsilon=1.0,
        production_mode=False,
        allow_fallback=True,
        run_id="test_synthetic_dev",
    )

    csv_path = Path(result["outputs"]["synthetic_csv"])
    manifest_path = Path(result["outputs"]["synthetic_manifest"])
    approval_path = Path(result["outputs"]["approval_request"])

    assert result["status"] == "completed"
    assert result["rows_generated"] == 25
    assert result["approval_status"] == "pending_approval"
    assert result["raw_maids_exported"] is False
    assert result["raw_observations_exported"] is False
    assert result["raw_lat_lng_exported"] is False
    assert result["raw_email_exported"] is False
    assert result["raw_phone_exported"] is False
    assert result["individual_user_data_exported"] is False

    assert csv_path.exists()
    assert manifest_path.exists()
    assert approval_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 25

    blocked_tokens = [
        "raw_maid",
        "device_id",
        "email",
        "phone",
        "latitude",
        "longitude",
        "observation",
        "observations",
    ]

    for col in df.columns:
        lower = col.lower()
        for token in blocked_tokens:
            assert token not in lower, f"Blocked column leaked: {col}"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["approval_status"] == "pending_approval"


def test_production_mode_fails_closed_without_dpgc(tmp_path: Path):
    agent = SyntheticEngineAgent()

    with pytest.raises(RuntimeError) as exc:
        agent.generate(
            cohorts=sample_safe_cohorts(),
            output_dir=tmp_path,
            synthetic_rows=25,
            engine_requested="dpgc",
            epsilon=1.0,
            production_mode=True,
            allow_fallback=False,
            run_id="test_synthetic_prod",
        )

    assert "failed closed" in str(exc.value).lower()
    assert "DPGCSynthesizer" in str(exc.value)


def test_production_mode_rejects_fallback(tmp_path: Path):
    agent = SyntheticEngineAgent()

    with pytest.raises(ValueError) as exc:
        agent.generate(
            cohorts=sample_safe_cohorts(),
            output_dir=tmp_path,
            synthetic_rows=25,
            engine_requested="dpgc",
            epsilon=1.0,
            production_mode=True,
            allow_fallback=True,
            run_id="test_invalid_config",
        )

    assert "cannot allow fallback" in str(exc.value).lower()


def test_invalid_epsilon_is_rejected(tmp_path: Path):
    agent = SyntheticEngineAgent()

    with pytest.raises(ValueError) as exc:
        agent.generate(
            cohorts=sample_safe_cohorts(),
            output_dir=tmp_path,
            synthetic_rows=25,
            engine_requested="auto",
            epsilon=0,
            production_mode=False,
            allow_fallback=True,
            run_id="test_bad_epsilon",
        )

    assert "epsilon" in str(exc.value).lower()


def test_sensitive_input_column_is_blocked(tmp_path: Path):
    df = sample_safe_cohorts()
    df["email"] = ["user@example.com", "test@example.com"]

    agent = SyntheticEngineAgent()

    with pytest.raises(ValueError) as exc:
        agent.generate(
            cohorts=df,
            output_dir=tmp_path,
            synthetic_rows=25,
            engine_requested="auto",
            epsilon=1.0,
            production_mode=False,
            allow_fallback=True,
            run_id="test_sensitive_input",
        )

    assert "blocked sensitive column" in str(exc.value).lower()
