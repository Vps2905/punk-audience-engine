from fastapi.testclient import TestClient

from app.main import app


def _headers():
    return {"x-audience-api-key": "test-key"}


def test_audience_intelligence_synthetic_api_records_job_and_lineage(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    db_url = f"sqlite:///{tmp_path / 'synthetic_api.db'}"
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", db_url)

    from app.api import audience_intelligence_synthetic as synthetic_api

    def fake_generate_synthetic_for_job(job_id: str, num_rows: int, use_sdv: bool):
        return {
            "message": "Synthetic dataset generated",
            "job_id": job_id,
            "backend": "test_synthetic_backend",
            "synthetic_path": f"data/synthetic/{job_id}_synthetic.csv",
            "manifest_path": f"data/synthetic/{job_id}_synthetic_manifest.json",
            "num_rows_generated": num_rows,
            "privacy_mode": "synthetic_from_aggregated_features",
            "safe_for_export_seed": True,
        }

    monkeypatch.setattr(
        synthetic_api,
        "generate_synthetic_for_job",
        fake_generate_synthetic_for_job,
    )

    client = TestClient(app)

    response = client.post(
        "/api/audience-intelligence/synthetic/generate/parent_job_1?num_rows=25&use_sdv=true&actor=tester",
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["parent_job_id"] == "parent_job_1"
    assert payload["num_rows_generated"] == 25
    assert payload["result"]["safe_for_export_seed"] is True
    assert payload["synthetic_job_id"].startswith("ingest_")

    lineage_response = client.get(
        f"/api/audience-intelligence/synthetic/{payload['synthetic_job_id']}/lineage",
        headers=_headers(),
    )

    assert lineage_response.status_code == 200
    lineage_payload = lineage_response.json()

    assert lineage_payload["job"]["status"] == "ok"
    assert lineage_payload["job"]["job"]["status"] == "completed"
    assert lineage_payload["lineage"]["has_privacy_step"] is True
    assert lineage_payload["lineage"]["has_output"] is True
    assert "synthetic_generation" in lineage_payload["lineage"]["privacy_controls_detected"]


def test_audience_intelligence_synthetic_api_returns_400_on_generation_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    db_url = f"sqlite:///{tmp_path / 'synthetic_bad.db'}"
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", db_url)

    from app.api import audience_intelligence_synthetic as synthetic_api

    def fake_generate_synthetic_for_job(job_id: str, num_rows: int, use_sdv: bool):
        raise ValueError("Processed feature table is empty. Cannot generate synthetic data.")

    monkeypatch.setattr(
        synthetic_api,
        "generate_synthetic_for_job",
        fake_generate_synthetic_for_job,
    )

    client = TestClient(app)

    response = client.post(
        "/api/audience-intelligence/synthetic/generate/missing_parent_job?num_rows=25",
        headers=_headers(),
    )

    assert response.status_code == 400
    assert "Processed feature table is empty" in response.json()["detail"]
