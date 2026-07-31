from pathlib import Path

import pytest

from app.services.ingestion_lineage_service import IngestionLineageService
from app.services.privacy_ingestion_pipeline_service import (
    PrivacyIngestionConfig,
    PrivacyIngestionPipelineService,
)


def _rows():
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


def test_privacy_ingestion_pipeline_outputs_safe_feature_rows(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'pipeline.db'}"
    service = PrivacyIngestionPipelineService(database_url=db_url)

    result = service.process_events(
        _rows(),
        config=PrivacyIngestionConfig(
            source_type="csv",
            source_ref="sample.csv",
            min_cohort_size=2,
            epsilon=1.0,
            delta=1e-5,
            hash_salt="test_salt",
            random_seed=42,
        ),
        run_id="run_1",
        actor="tester",
    )

    assert result["status"] == "completed"
    assert result["input_rows"] == 4
    assert result["bounded_rows"] == 3
    assert result["aggregated_rows"] == 1
    assert result["safe_feature_rows_count"] == 1

    safe_row = result["safe_feature_rows"][0]

    assert safe_row["bounded_count"] == 3
    assert "dp_noisy_count" in safe_row
    assert safe_row["dp_epsilon"] == 1.0
    assert safe_row["dp_delta"] == 1e-5
    assert "entity_id" not in safe_row

    assert "contribution_bounding" in result["privacy_controls"]
    assert "differential_privacy_noise" in result["privacy_controls"]

    lineage = IngestionLineageService(database_url=db_url).build_chain_summary(
        job_id=result["job_id"]
    )

    assert lineage["has_source"] is True
    assert lineage["has_privacy_step"] is True
    assert lineage["has_output"] is True
    assert "contribution_bounding" in lineage["privacy_controls_detected"]
    assert "differential_privacy" in lineage["privacy_controls_detected"]
    assert "k_anonymity" in lineage["privacy_controls_detected"]


def test_privacy_ingestion_pipeline_blocks_small_cohort(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'pipeline.db'}"
    service = PrivacyIngestionPipelineService(database_url=db_url)

    result = service.process_events(
        _rows(),
        config=PrivacyIngestionConfig(
            min_cohort_size=10,
            hash_salt="test_salt",
            random_seed=42,
        ),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "All cohorts failed k-anonymity threshold."
    assert result["safe_feature_rows"] == []


def test_privacy_ingestion_pipeline_blocks_missing_required_columns(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'pipeline.db'}"
    service = PrivacyIngestionPipelineService(database_url=db_url)

    result = service.process_events(
        [
            {
                "created_at": "2026-07-10T10:00:00Z",
                "location_name": "montreal",
                "primary_poi_type": "cafe",
                "created_day_part": "evening",
            }
        ],
        config=PrivacyIngestionConfig(min_cohort_size=1),
    )

    assert result["status"] == "blocked"
    assert "entity_id" in result["missing_columns"]
    assert result["safe_feature_rows"] == []


def test_privacy_ingestion_pipeline_validates_config(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'pipeline.db'}"
    service = PrivacyIngestionPipelineService(database_url=db_url)

    try:
        service.process_events(
            _rows(),
            config=PrivacyIngestionConfig(min_cohort_size=0),
        )
    except ValueError as exc:
        assert "min_cohort_size must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected config validation failure")


def test_production_tokenization_rejects_legacy_public_salt(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.delenv("AUDIENCE_TOKENIZATION_HMAC_KEY", raising=False)
    service = PrivacyIngestionPipelineService(
        database_url=f"sqlite:///{tmp_path / 'production-key.db'}"
    )

    with pytest.raises(RuntimeError, match="managed HMAC key"):
        service.process_events(
            _rows(),
            config=PrivacyIngestionConfig(
                min_cohort_size=2,
                hash_salt="legacy-public-salt",
            ),
        )
