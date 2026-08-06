from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import ClassVar

import pytest

from scripts import index_historical_audience_features as script


def _args() -> Namespace:
    return Namespace(
        tenant_id="tenant-a",
        purpose="internal-evaluation",
        rights_policy_id="historical-internal-use",
        privacy_policy_version="privacy-policy-v1",
        job_id="reviewed-snapshot-id",
        source_ref="postgres://legacy_safe_audience_vectors",
        data_use_mode="historical_preview",
    )


class FakePreflight:
    result: ClassVar[dict] = {}
    calls: ClassVar[list[dict]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.result)


def _configure_feature_databases(monkeypatch):
    monkeypatch.setenv(
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL",
        "configured-writer-target",
    )
    monkeypatch.setenv(
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL",
        "configured-migration-target",
    )


def test_index_requires_explicit_feature_database(monkeypatch):
    monkeypatch.setenv("ECHO_DATABASE_URL", "configured-source")
    monkeypatch.setenv(
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL",
        "configured-migration-target",
    )
    monkeypatch.delenv(
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="AUDIENCE_FEATURE_WRITER_DATABASE_URL",
    ):
        script.index_historical_features(_args())


def test_index_requires_explicit_migration_database(monkeypatch):
    monkeypatch.setenv("ECHO_DATABASE_URL", "configured-source")
    monkeypatch.setenv(
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL",
        "configured-writer-target",
    )
    monkeypatch.delenv(
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="AUDIENCE_FEATURE_MIGRATION_DATABASE_URL",
    ):
        script.index_historical_features(_args())


def test_index_rejects_same_database_before_snapshot_read(monkeypatch):
    monkeypatch.setenv("ECHO_DATABASE_URL", "configured-source")
    _configure_feature_databases(monkeypatch)
    FakePreflight.calls = []
    FakePreflight.result = {
        "status": "phase2_schema_ready",
        "same_database_as_source": True,
    }
    monkeypatch.setattr(
        script,
        "Phase2DatabasePreflightService",
        FakePreflight,
    )
    monkeypatch.setattr(
        script,
        "LegacyPostgresSafeVectorReader",
        lambda **_kwargs: pytest.fail("snapshot must not be read"),
    )

    with pytest.raises(RuntimeError, match="separate"):
        script.index_historical_features(_args())


def test_index_rejects_unverified_feature_schema(monkeypatch):
    monkeypatch.setenv("ECHO_DATABASE_URL", "configured-source")
    _configure_feature_databases(monkeypatch)
    FakePreflight.calls = []
    FakePreflight.result = {
        "status": "ready_for_approved_migration",
        "same_database_as_source": False,
    }
    monkeypatch.setattr(
        script,
        "Phase2DatabasePreflightService",
        FakePreflight,
    )
    monkeypatch.setattr(
        script,
        "LegacyPostgresSafeVectorReader",
        lambda **_kwargs: pytest.fail("snapshot must not be read"),
    )

    with pytest.raises(RuntimeError, match="preflight-verified"):
        script.index_historical_features(_args())


def test_index_reads_only_explicit_reviewed_snapshot_after_preflight(
    monkeypatch,
):
    monkeypatch.setenv("ECHO_DATABASE_URL", "configured-source")
    _configure_feature_databases(monkeypatch)
    FakePreflight.calls = []
    FakePreflight.result = {
        "status": "phase2_schema_ready",
        "same_database_as_source": False,
    }
    monkeypatch.setattr(
        script,
        "Phase2DatabasePreflightService",
        FakePreflight,
    )

    calls = {}
    snapshot = SimpleNamespace(
        job_id="reviewed-snapshot-id",
        latest_source_timestamp=datetime(
            2026,
            7,
            8,
            tzinfo=timezone.utc,
        ),
    )

    class FakeReader:
        def __init__(self, *, database_url):
            calls["source_database_url"] = database_url

        def load_snapshot(self, *, job_id):
            calls["job_id"] = job_id
            return snapshot

    class FakeAdapter:
        def adapt(self, received_snapshot, **kwargs):
            calls["snapshot"] = received_snapshot
            calls["adapt_kwargs"] = kwargs
            return "safe-feature-set"

    class FakeStore:
        def __init__(self, *, database_url):
            calls["feature_database_url"] = database_url

        def save_feature_set(self, feature_set):
            calls["feature_set"] = feature_set
            return {"status": "saved"}

    monkeypatch.setattr(
        script,
        "LegacyPostgresSafeVectorReader",
        FakeReader,
    )
    monkeypatch.setattr(
        script,
        "PostgresCanonicalFeatureAdapterService",
        FakeAdapter,
    )
    monkeypatch.setattr(
        script,
        "PgvectorAudienceFeatureStore",
        FakeStore,
    )

    receipt = script.index_historical_features(_args())

    assert FakePreflight.calls == [
        {
            "source_database_url": "configured-source",
            "feature_database_url": "configured-migration-target",
            "punk_owned_target_confirmed": True,
        }
    ]
    assert calls["job_id"] == "reviewed-snapshot-id"
    assert calls["source_database_url"] == "configured-source"
    assert calls["feature_database_url"] == "configured-writer-target"
    assert calls["feature_set"] == "safe-feature-set"
    assert receipt["status"] == "saved"
    assert receipt["activation_blocked"] is True
