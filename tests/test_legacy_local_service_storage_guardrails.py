import pytest

from app.services import cohort_service
from app.services import ingestion_service
from app.services import lineage_service
from app.services import lookalike_service
from app.services import synthetic_service


def _blocked(_feature_name):
    raise RuntimeError("local storage blocked")


def test_ingestion_blocks_before_creating_directories(
    tmp_path,
    monkeypatch,
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"

    monkeypatch.setattr(ingestion_service, "RAW_DIR", raw)
    monkeypatch.setattr(
        ingestion_service,
        "PROCESSED_DIR",
        processed,
    )
    monkeypatch.setattr(
        ingestion_service,
        "require_local_file_storage_allowed",
        _blocked,
    )

    with pytest.raises(RuntimeError, match="local storage blocked"):
        ingestion_service.ingest_csv(None)

    assert not raw.exists()
    assert not processed.exists()


def test_synthetic_blocks_before_local_file_access(
    tmp_path,
    monkeypatch,
):
    processed = tmp_path / "processed"
    synthetic = tmp_path / "synthetic"

    monkeypatch.setattr(
        synthetic_service,
        "PROCESSED_DIR",
        processed,
    )
    monkeypatch.setattr(
        synthetic_service,
        "SYNTHETIC_DIR",
        synthetic,
    )
    monkeypatch.setattr(
        synthetic_service,
        "require_local_file_storage_allowed",
        _blocked,
    )

    with pytest.raises(RuntimeError, match="local storage blocked"):
        synthetic_service.generate_synthetic_for_job("job_1")

    assert not processed.exists()
    assert not synthetic.exists()


def test_cohort_creation_blocks_before_directory_creation(
    tmp_path,
    monkeypatch,
):
    cohort_dir = tmp_path / "cohorts"

    monkeypatch.setattr(
        cohort_service,
        "COHORT_DIR",
        cohort_dir,
    )
    monkeypatch.setattr(
        cohort_service,
        "require_local_file_storage_allowed",
        _blocked,
    )

    with pytest.raises(RuntimeError, match="local storage blocked"):
        cohort_service.create_cohort(
            job_id="job_1",
            name="Blocked cohort",
        )

    assert not cohort_dir.exists()


def test_cohort_lookup_blocks_before_local_read(
    tmp_path,
    monkeypatch,
):
    cohort_dir = tmp_path / "cohorts"

    monkeypatch.setattr(
        cohort_service,
        "COHORT_DIR",
        cohort_dir,
    )
    monkeypatch.setattr(
        cohort_service,
        "require_local_file_storage_allowed",
        _blocked,
    )

    with pytest.raises(RuntimeError, match="local storage blocked"):
        cohort_service.get_cohort("cohort_1")

    assert not cohort_dir.exists()


def test_lookalike_blocks_before_directory_creation(
    tmp_path,
    monkeypatch,
):
    lookalike_dir = tmp_path / "cohorts"

    monkeypatch.setattr(
        lookalike_service,
        "LOOKALIKE_DIR",
        lookalike_dir,
    )
    monkeypatch.setattr(
        lookalike_service,
        "require_local_file_storage_allowed",
        _blocked,
    )

    with pytest.raises(RuntimeError, match="local storage blocked"):
        lookalike_service.create_lookalike("cohort_1")

    assert not lookalike_dir.exists()


def test_lineage_blocks_before_directory_creation(
    tmp_path,
    monkeypatch,
):
    lineage_dir = tmp_path / "lineage"

    monkeypatch.setattr(
        lineage_service,
        "LINEAGE_DIR",
        lineage_dir,
    )
    monkeypatch.setattr(
        lineage_service,
        "require_local_file_storage_allowed",
        _blocked,
    )

    with pytest.raises(RuntimeError, match="local storage blocked"):
        lineage_service.write_lineage(
            "job_1",
            {"source": "test"},
        )

    assert not lineage_dir.exists()
