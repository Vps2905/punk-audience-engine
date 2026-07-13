from pathlib import Path

from app import config


def _set_test_directories(monkeypatch, root: Path) -> list[Path]:
    directories = {
        "RAW_DIR": root / "raw",
        "PROCESSED_DIR": root / "processed",
        "SYNTHETIC_DIR": root / "synthetic",
        "VECTOR_DIR": root / "vectors",
        "COHORT_DIR": root / "cohorts",
        "EXPORT_DIR": root / "exports",
        "LINEAGE_DIR": root / "lineage",
    }

    for name, path in directories.items():
        monkeypatch.setattr(config.settings, name, path)

    return list(directories.values())


def test_production_startup_does_not_create_local_data_dirs(
    tmp_path,
    monkeypatch,
):
    directories = _set_test_directories(
        monkeypatch,
        tmp_path / "production-data",
    )

    monkeypatch.setattr(
        config.production_guardrails,
        "local_file_storage_allowed",
        lambda: False,
    )

    created = config.ensure_data_dirs()

    assert created == []
    assert all(not path.exists() for path in directories)


def test_local_mode_creates_required_data_dirs(
    tmp_path,
    monkeypatch,
):
    directories = _set_test_directories(
        monkeypatch,
        tmp_path / "local-data",
    )

    monkeypatch.setattr(
        config.production_guardrails,
        "local_file_storage_allowed",
        lambda: True,
    )

    created = config.ensure_data_dirs()

    assert set(created) == {str(path) for path in directories}
    assert all(path.is_dir() for path in directories)
