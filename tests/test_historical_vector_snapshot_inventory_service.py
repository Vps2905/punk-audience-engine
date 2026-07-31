from __future__ import annotations

from datetime import datetime, timezone

from app.services.historical_vector_snapshot_inventory_service import (
    HistoricalVectorSnapshotInventoryService,
)


def _row(
    job_id: str,
    *,
    count: int,
    position_time: int,
    dimension: int = 384,
    declared_count: int | None = None,
    incompatible_embeddings: int = 0,
) -> dict:
    return {
        "job_id": job_id,
        "declared_vector_count": (
            count if declared_count is None else declared_count
        ),
        "vector_dimension": dimension,
        "updated_at": datetime(
            2026,
            7,
            position_time,
            tzinfo=timezone.utc,
        ),
        "stored_vector_count": count,
        "distinct_vector_indices": count,
        "min_vector_index": 0 if count else None,
        "max_vector_index": count - 1 if count else None,
        "incompatible_embedding_count": incompatible_embeddings,
    }


class FakeMappings:
    def __init__(self, *, one=None, rows=None):
        self._one = one
        self._rows = rows or []

    def mappings(self):
        return self

    def one(self):
        return self._one

    def all(self):
        return self._rows


class FakeTransaction:
    def __init__(self):
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.transaction = FakeTransaction()
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def begin(self):
        return self.transaction

    def exec_driver_sql(self, statement):
        self.calls.append(str(statement))

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append(sql)
        if "transaction_read_only" in sql:
            return FakeMappings(one={"transaction_read_only": "on"})
        if "to_regclass" in sql:
            return FakeMappings(
                one={
                    "models_present": True,
                    "vectors_present": True,
                    "source_events_present": True,
                }
            )
        if "MAX(created_at)" in sql:
            return FakeMappings(
                one={
                    "source_latest_at": datetime(
                        2026,
                        7,
                        8,
                        tzinfo=timezone.utc,
                    )
                }
            )
        if "WITH vector_stats AS" in sql:
            assert params["expected_dimension"] == 384
            assert params["limit"] == 100
            return FakeMappings(rows=self.rows)
        raise AssertionError(f"Unexpected inventory SQL: {sql}")


class FakeDialect:
    name = "postgresql"


class FakeEngine:
    dialect = FakeDialect()

    def __init__(self, rows):
        self.connection = FakeConnection(rows)

    def connect(self):
        return self.connection


def test_inventory_explains_latest_small_snapshot_without_auto_selecting():
    engine = FakeEngine(
        [
            _row("latest_small", count=4, position_time=24),
            _row("older_complete", count=86, position_time=22),
        ]
    )

    report = HistoricalVectorSnapshotInventoryService(
        engine=engine
    ).inventory()

    assert report["status"] == "inventory_ready"
    assert report["latest_snapshot_job_id"] == "latest_small"
    assert report["latest_compatible_snapshot_job_id"] == "latest_small"
    assert (
        report["largest_compatible_snapshot_job_id"]
        == "older_complete"
    )
    assert report["automatic_selection_performed"] is False
    assert report["selection_requires_operator_review"] is True


def test_inventory_marks_count_dimension_and_index_failures():
    service = HistoricalVectorSnapshotInventoryService(
        engine=FakeEngine([])
    )
    bad_row = _row(
        "incomplete",
        count=4,
        position_time=24,
        dimension=3,
        declared_count=86,
        incompatible_embeddings=4,
    )
    bad_row["distinct_vector_indices"] = 3
    bad_row["max_vector_index"] = 8

    report = service._build_report(
        [bad_row],
        source_latest_at=None,
        limit=100,
    )

    snapshot = report["snapshots"][0]
    assert snapshot["compatible_complete_snapshot"] is False
    assert snapshot["issues"] == [
        "declared_stored_count_mismatch",
        "model_dimension_incompatible",
        "embedding_dimension_incompatible",
        "vector_indices_not_contiguous",
    ]
    assert report["compatible_complete_snapshot_count"] == 0


def test_inventory_uses_verified_read_only_transaction_and_rolls_back():
    engine = FakeEngine(
        [_row("snapshot", count=86, position_time=22)]
    )

    report = HistoricalVectorSnapshotInventoryService(
        engine=engine
    ).inventory()

    assert engine.connection.calls[0] == "SET TRANSACTION READ ONLY"
    assert engine.connection.transaction.rolled_back is True
    assert report["read_only_transaction_verified"] is True


def test_inventory_query_returns_no_embeddings_traits_or_metadata():
    engine = FakeEngine(
        [_row("snapshot", count=86, position_time=22)]
    )

    report = HistoricalVectorSnapshotInventoryService(
        engine=engine
    ).inventory()
    sql = "\n".join(engine.connection.calls)
    rendered = str(report)

    assert "trait_text" not in sql
    assert "metadata_json" not in sql
    assert "SELECT embedding" not in sql
    assert "raw_identifiers_read" in rendered
    assert report["raw_identifiers_read"] is False
    assert report["embeddings_returned"] is False
    assert report["trait_or_metadata_rows_returned"] is False


def test_inventory_missing_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    report = HistoricalVectorSnapshotInventoryService().inventory()

    assert report["status"] == "blocked"
    assert (
        report["reason_code"]
        == "historical_source_database_not_configured"
    )
    assert report["snapshots"] == []
    assert report["credentials_exposed"] is False
