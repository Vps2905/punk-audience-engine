from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import synthetic_service as service


def _prepare_processed(monkeypatch, tmp_path: Path, job_id: str, df: pd.DataFrame) -> None:
    processed_dir = tmp_path / "processed"
    synthetic_dir = tmp_path / "synthetic"
    processed_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(service, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(service, "SYNTHETIC_DIR", synthetic_dir)

    df.to_csv(processed_dir / f"{job_id}_clean_features.csv", index=False)


def _safe_processed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["montreal", "toronto"],
            "interest": ["coffee", "gym"],
            "visit_time": ["evening", "morning"],
            "cohort_size": [5000, 3000],
            "noisy_count": [5020, 3010],
            "privacy_status": ["passed", "passed"],
            "k_min": [1000, 1000],
        }
    )


def test_legacy_synthetic_service_defaults_to_production_aggregate_sampler(monkeypatch, tmp_path):
    job_id = "legacy_prod_safe"
    _prepare_processed(monkeypatch, tmp_path, job_id, _safe_processed_df())

    result = service.generate_synthetic_for_job(
        job_id=job_id,
        num_rows=20,
    )

    assert result["status"] == "completed"
    assert result["backend"] == "aggregated_sampler"
    assert result["production_mode"] is True
    assert result["allow_fallback"] is False
    assert result["contains_raw_maids"] is False
    assert result["contains_individual_user_data"] is False
    assert result["requires_manual_approval_before_upload"] is True

    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["production_mode"] is True
    assert manifest["allow_fallback"] is False
    assert manifest["use_sdv_requested"] is False

    synthetic = pd.read_csv(result["synthetic_path"])
    assert len(synthetic) == 20
    assert "synthetic_profile_id" in synthetic.columns
    assert "privacy_mode" in synthetic.columns

    for col in synthetic.columns:
        lower = col.lower()
        assert "email" not in lower
        assert "phone" not in lower
        assert "device" not in lower
        assert "raw" not in lower


def test_legacy_synthetic_service_rejects_sdv_in_production(monkeypatch, tmp_path):
    job_id = "legacy_prod_rejects_sdv"
    _prepare_processed(monkeypatch, tmp_path, job_id, _safe_processed_df())

    with pytest.raises(ValueError) as exc:
        service.generate_synthetic_for_job(
            job_id=job_id,
            num_rows=10,
            use_sdv=True,
            production_mode=True,
            allow_fallback=False,
        )

    assert "dev-only" in str(exc.value)


def test_legacy_synthetic_service_fails_closed_without_explicit_dev_fallback(monkeypatch, tmp_path):
    job_id = "legacy_dev_failed_closed"
    _prepare_processed(monkeypatch, tmp_path, job_id, _safe_processed_df())

    def broken_sdv(df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        raise RuntimeError("sdv unavailable")

    monkeypatch.setattr(service, "try_sdv_synthetic", broken_sdv)

    with pytest.raises(RuntimeError) as exc:
        service.generate_synthetic_for_job(
            job_id=job_id,
            num_rows=10,
            use_sdv=True,
            production_mode=False,
            allow_fallback=False,
        )

    assert "failed closed" in str(exc.value).lower()


def test_legacy_synthetic_service_blocks_unsafe_source_columns(monkeypatch, tmp_path):
    job_id = "legacy_blocks_unsafe"
    df = _safe_processed_df()
    df["device_id"] = ["abc", "def"]
    _prepare_processed(monkeypatch, tmp_path, job_id, df)

    with pytest.raises(ValueError) as exc:
        service.generate_synthetic_for_job(
            job_id=job_id,
            num_rows=10,
        )

    assert "unsafe synthetic source columns blocked" in str(exc.value).lower()
    assert "device_id" in str(exc.value)


def test_legacy_synthetic_route_returns_400_for_blocked_sdv(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    job_id = "route_blocks_sdv"
    _prepare_processed(monkeypatch, tmp_path, job_id, _safe_processed_df())

    client = TestClient(app)

    response = client.post(
        f"/synthetic/generate/{job_id}?num_rows=10&use_sdv=true&production_mode=true",
        headers={
            "x-audience-api-key": "test-key",
            "x-audience-tenant-id": "tenant-a",
        },
    )

    assert response.status_code == 400
    assert "dev-only" in response.json()["detail"]
