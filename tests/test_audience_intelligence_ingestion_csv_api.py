from fastapi.testclient import TestClient

from app.main import app


def _headers():
    return {"x-audience-api-key": "test-key"}


def _csv_bytes():
    return (
        "entity_id,created_at,location_name,primary_poi_type,created_day_part\n"
        "user_1,2026-07-10T10:00:00Z,montreal,cafe,evening\n"
        "user_1,2026-07-10T10:30:00Z,montreal,cafe,evening\n"
        "user_2,2026-07-10T11:00:00Z,montreal,cafe,evening\n"
        "user_3,2026-07-10T11:30:00Z,montreal,cafe,evening\n"
    ).encode("utf-8")


def test_audience_intelligence_csv_ingest_returns_safe_features(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    db_url = f"sqlite:///{tmp_path / 'csv_api.db'}"
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", db_url)

    client = TestClient(app)

    response = client.post(
        "/api/audience-intelligence/ingest/csv?min_cohort_size=2&epsilon=1.0&hash_salt=test_salt&random_seed=42&run_id=csv_run_1&actor=csv_test",
        headers=_headers(),
        files={"file": ("sample.csv", _csv_bytes(), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["input_rows"] == 4
    assert payload["bounded_rows"] == 3
    assert payload["aggregated_rows"] == 1
    assert payload["safe_feature_rows_count"] == 1
    assert "entity_id" not in payload["safe_feature_rows"][0]
    assert "hmac_sha256_tokenization" in payload["privacy_controls"]
    assert "k_anonymity" in payload["privacy_controls"]


def test_audience_intelligence_csv_ingest_rejects_non_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    db_url = f"sqlite:///{tmp_path / 'csv_bad.db'}"
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", db_url)

    client = TestClient(app)

    response = client.post(
        "/api/audience-intelligence/ingest/csv",
        headers=_headers(),
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]


def test_audience_intelligence_csv_ingest_blocks_missing_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    db_url = f"sqlite:///{tmp_path / 'csv_missing.db'}"
    monkeypatch.setenv("AUDIENCE_INGESTION_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_LINEAGE_DATABASE_URL", db_url)

    client = TestClient(app)

    bad_csv = (
        "created_at,location_name,primary_poi_type,created_day_part\n"
        "2026-07-10T10:00:00Z,montreal,cafe,evening\n"
    ).encode("utf-8")

    response = client.post(
        "/api/audience-intelligence/ingest/csv?min_cohort_size=1",
        headers=_headers(),
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "blocked"
    assert "entity_id" in payload["missing_columns"]
    assert payload["safe_feature_rows"] == []
