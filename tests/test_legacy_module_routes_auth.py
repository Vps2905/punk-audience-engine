from fastapi.testclient import TestClient

from app.main import app


def test_legacy_ingest_status_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    client = TestClient(app)

    response = client.get("/status/some_job")

    assert response.status_code in {401, 403}


def test_legacy_synthetic_generate_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    client = TestClient(app)

    response = client.post("/synthetic/generate/some_job")

    assert response.status_code in {401, 403}


def test_legacy_full_pipeline_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    client = TestClient(app)

    response = client.post("/audience/generate")

    assert response.status_code in {401, 403, 422}


def test_legacy_synthetic_generate_allows_valid_api_key(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    from app.api import synthetic_routes

    def fake_generate_synthetic_for_job(job_id: str, num_rows: int, use_sdv: bool, **kwargs):
        return {
            "message": "Synthetic dataset generated",
            "job_id": job_id,
            "num_rows_generated": num_rows,
            "safe_for_export_seed": True,
        }

    monkeypatch.setattr(
        synthetic_routes,
        "generate_synthetic_for_job",
        fake_generate_synthetic_for_job,
    )

    client = TestClient(app)

    response = client.post(
        "/synthetic/generate/some_job?num_rows=10&use_sdv=true",
        headers={"x-audience-api-key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "some_job"
    assert response.json()["num_rows_generated"] == 10
