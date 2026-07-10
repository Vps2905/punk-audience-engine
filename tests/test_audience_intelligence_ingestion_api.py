from fastapi.testclient import TestClient

from app.main import app


def _headers():
    return {"x-audience-api-key": "test-key"}


def _events():
    return [
        {
            "entity_id": "user_1",
            "created_at": "2026-07-10T10:00:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
        },
        {
            "entity_id": "user_1",
            "created_at": "2026-07-10T10:30:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
        },
        {
            "entity_id": "user_2",
            "created_at": "2026-07-10T11:00:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
        },
        {
            "entity_id": "user_3",
            "created_at": "2026-07-10T11:30:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
        },
    ]


def test_audience_intelligence_ingest_api_returns_safe_features(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")

    client = TestClient(app)

    response = client.post(
        "/api/audience-intelligence/ingest",
        headers=_headers(),
        json={
            "run_id": "run_api_1",
            "actor": "api_test",
            "events": _events(),
            "config": {
                "source_type": "api",
                "source_ref": "unit_test",
                "min_cohort_size": 2,
                "epsilon": 1.0,
                "delta": 0.00001,
                "hash_salt": "test_salt",
                "random_seed": 42,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["input_rows"] == 4
    assert payload["bounded_rows"] == 3
    assert payload["safe_feature_rows_count"] == 1
    assert "job_id" in payload
    assert "entity_id" not in payload["safe_feature_rows"][0]
    assert "contribution_bounding" in payload["privacy_controls"]
    assert "lineage_logging" in payload["privacy_controls"]


def test_audience_intelligence_ingest_status_api(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    db_url = f"sqlite:///{tmp_path / 'api_status.db'}"
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", db_url)

    client = TestClient(app)

    ingest_response = client.post(
        "/api/audience-intelligence/ingest",
        headers=_headers(),
        json={
            "events": _events(),
            "config": {
                "min_cohort_size": 2,
                "hash_salt": "test_salt",
                "random_seed": 42,
            },
        },
    )

    assert ingest_response.status_code == 200
    job_id = ingest_response.json()["job_id"]

    status_response = client.get(
        f"/api/audience-intelligence/ingest/{job_id}/status",
        headers=_headers(),
    )

    assert status_response.status_code == 200
    payload = status_response.json()

    assert payload["job"]["status"] == "ok"
    assert payload["job"]["job"]["job_id"] == job_id
    assert payload["lineage"]["has_source"] is True
    assert payload["lineage"]["has_privacy_step"] is True
    assert payload["lineage"]["has_output"] is True


def test_audience_intelligence_ingest_api_rejects_bad_config(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", f"sqlite:///{tmp_path / 'api_bad.db'}")
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", f"sqlite:///{tmp_path / 'api_bad.db'}")

    client = TestClient(app)

    response = client.post(
        "/api/audience-intelligence/ingest",
        headers=_headers(),
        json={
            "events": _events(),
            "config": {
                "min_cohort_size": 0,
            },
        },
    )

    assert response.status_code == 400
    assert "min_cohort_size must be >= 1" in response.json()["detail"]
