from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agents.synthetic_engine_agent import SyntheticEngineAgent


def production_safe_cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "8_30d",
                "sessions": 20,
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
                "sessions": 18,
                "total_maid_volume": 3200,
                "noisy_maid_volume": 3180,
                "quality_score": 0.87,
                "privacy_status": "passed",
                "cluster_id": 2,
                "trait_text": "new york gym morning audience",
            },
        ]
    )


def test_dp_aggregate_production_success_records_budget_and_approval(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRIVACY_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("APPROVAL_DIR", str(tmp_path / "approvals"))
    monkeypatch.setenv("K_ANONYMITY_MIN", "1000")

    agent = SyntheticEngineAgent()

    result = agent.generate(
        cohorts=production_safe_cohorts(),
        output_dir=tmp_path / "synthetic",
        synthetic_rows=50,
        engine_requested="dp_aggregate",
        epsilon=1.0,
        production_mode=True,
        allow_fallback=False,
        run_id="dp_aggregate_success",
    )

    assert result["status"] == "completed"
    assert result["engine_used"] == "DPAggregateCohortSynthesizer"
    assert result["production_mode"] is True
    assert result["allow_fallback"] is False
    assert result["privacy_budget_recorded"] is True
    assert result["approval_status"] == "pending_approval"

    csv_path = Path(result["outputs"]["synthetic_csv"])
    manifest_path = Path(result["outputs"]["synthetic_manifest"])
    schema_path = Path(result["outputs"]["safe_input_schema"])
    approval_path = Path(result["outputs"]["approval_request"])

    assert csv_path.exists()
    assert manifest_path.exists()
    assert schema_path.exists()
    assert approval_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 50
    assert df["quality_score"].between(0, 1).all()
    assert not df["trait_text"].str.contains("sdv-id", case=False, na=False).any()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["privacy_budget_recorded"] is True
    assert manifest["privacy_budget_record"]["epsilon_spent"] == 1.0
    assert manifest["raw_maids_exported"] is False
    assert manifest["individual_user_data_exported"] is False

    ledger_path = tmp_path / "ledger.jsonl"
    assert ledger_path.exists()
    assert "dp_aggregate_success" in ledger_path.read_text()


def test_mocked_sdv_dpgc_production_success_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRIVACY_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("APPROVAL_DIR", str(tmp_path / "approvals"))
    monkeypatch.setenv("K_ANONYMITY_MIN", "1000")

    agent = SyntheticEngineAgent()

    def fake_dpgc(df: pd.DataFrame, rows: int, epsilon: float) -> pd.DataFrame:
        sampled = df.sample(n=rows, replace=True, random_state=42).reset_index(drop=True)
        sampled["trait_text"] = "sdv-id-should-be-rebuilt"
        return sampled

    monkeypatch.setattr(agent, "_try_dpgc", fake_dpgc)

    result = agent.generate(
        cohorts=production_safe_cohorts(),
        output_dir=tmp_path / "synthetic_dpgc",
        synthetic_rows=25,
        engine_requested="sdv_dpgc",
        epsilon=1.0,
        production_mode=True,
        allow_fallback=False,
        run_id="mocked_dpgc_success",
    )

    assert result["status"] == "completed"
    assert result["engine_used"] == "DPGCSynthesizer"
    assert result["privacy_budget_recorded"] is True

    df = pd.read_csv(result["outputs"]["synthetic_csv"])
    assert len(df) == 25
    assert not df["trait_text"].str.contains("sdv-id", case=False, na=False).any()


def test_production_rejects_cohorts_below_k_min(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("K_ANONYMITY_MIN", "1000")

    agent = SyntheticEngineAgent()

    bad = production_safe_cohorts()
    bad.loc[0, "noisy_maid_volume"] = 10

    with pytest.raises(ValueError) as exc:
        agent.generate(
            cohorts=bad,
            output_dir=tmp_path,
            synthetic_rows=10,
            engine_requested="dp_aggregate",
            epsilon=1.0,
            production_mode=True,
            allow_fallback=False,
            run_id="below_k_min",
        )

    assert "below k_min" in str(exc.value)


def test_external_dp_service_fails_closed_until_configured(tmp_path: Path):
    agent = SyntheticEngineAgent()

    with pytest.raises(RuntimeError) as exc:
        agent.generate(
            cohorts=production_safe_cohorts(),
            output_dir=tmp_path,
            synthetic_rows=10,
            engine_requested="external_dp_service",
            epsilon=1.0,
            production_mode=True,
            allow_fallback=False,
            run_id="external_service_missing",
        )

    assert "failed closed" in str(exc.value).lower()
