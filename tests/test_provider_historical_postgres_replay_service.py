from datetime import datetime, timezone
import random

import pytest

from app.models.provider_historical_replay_contracts import (
    HistoricalReplayConfig,
)
from app.services.provider_historical_postgres_replay_service import (
    ProviderHistoricalPostgresReplayService,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def one(self):
        return self._rows[0]


class _Transaction:
    def __init__(self):
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True


class _Connection:
    def __init__(self):
        self.driver_sql = []
        self.queries = []
        self.transaction = _Transaction()

    def begin(self):
        return self.transaction

    def exec_driver_sql(self, statement):
        self.driver_sql.append(statement)

    def execute(self, statement, params=None):
        query = str(statement)
        self.queries.append((query, params or {}))
        if "information_schema.columns" in query:
            return _Result(
                [
                    "id",
                    "session_id",
                    "created_at",
                    "maid_count",
                    "maids",
                    "pois",
                    "center",
                ]
            )
        if "source_row_count" in query:
            return _Result(
                [
                    {
                        "source_row_count": 287,
                        "declared_maid_count": 4_496_802,
                        "earliest_source_timestamp": datetime(
                            2026, 4, 27, tzinfo=timezone.utc
                        ),
                        "latest_source_timestamp": datetime(
                            2026, 7, 8, tzinfo=timezone.utc
                        ),
                    }
                ]
            )
        return _Result(
            [
                {
                    "location_name": "montreal",
                    "primary_poi_type": "restaurant",
                    "bounded_count": 1500,
                },
                {
                    "location_name": "montreal",
                    "primary_poi_type": "hospital",
                    "bounded_count": 2000,
                },
            ]
        )


class _ConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Engine:
    def __init__(self):
        self.connection = _Connection()

    def connect(self):
        return _ConnectionContext(self.connection)


def test_contract_rejects_demo_privacy_thresholds_and_sql_injection():
    with pytest.raises(ValueError, match="min_cohort_size"):
        HistoricalReplayConfig(tenant_id="punk_internal", min_cohort_size=999)

    with pytest.raises(ValueError, match="table_name"):
        HistoricalReplayConfig(
            tenant_id="punk_internal",
            table_name='maid_extractions"; DROP TABLE x;--',
        )


def test_replay_requires_explicit_restricted_processing_confirmation():
    service = ProviderHistoricalPostgresReplayService(engine=_Engine())

    with pytest.raises(PermissionError):
        service.replay(HistoricalReplayConfig(tenant_id="punk_internal"))


def test_replay_returns_only_dp_safe_non_sensitive_aggregates():
    engine = _Engine()
    service = ProviderHistoricalPostgresReplayService(
        engine=engine,
        rng=random.Random(7),
    )

    result = service.replay(
        HistoricalReplayConfig(tenant_id="punk_internal"),
        confirm_restricted_identifier_processing=True,
    )

    assert result["status"] == "historical_privacy_replay_completed"
    assert result["safe_cohort_count"] == 1
    assert result["blocked_sensitive_cohort_count"] == 1
    assert result["safe_feature_rows"][0]["primary_poi_type"] == "restaurant"
    assert "bounded_count" not in result["safe_feature_rows"][0]
    assert result["raw_identifier_values_returned"] is False
    assert result["raw_observations_read"] is False
    assert result["eligible_for_activation"] is False
    assert result["privacy_budget_persisted"] is False
    assert engine.connection.driver_sql[0] == "SET TRANSACTION READ ONLY"
    assert engine.connection.transaction.rolled_back is True


def test_bounded_query_never_selects_identifiers_outside_internal_cte():
    service = ProviderHistoricalPostgresReplayService(engine=_Engine())
    query = service._bounded_cohort_query(  # noqa: SLF001
        HistoricalReplayConfig(tenant_id="punk_internal")
    )

    final_select = query.rsplit("SELECT", 1)[-1].lower()
    assert "entity_id" not in final_select
    assert "bounded_count >= :min_cohort_size" in query
    assert "select distinct" in query.lower()
    assert "'establishment'" in query
    assert "'point_of_interest'" in query


def test_same_historical_release_has_replay_stable_private_counts(monkeypatch):
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "s" * 64)
    config = HistoricalReplayConfig(tenant_id="punk_internal")

    first = ProviderHistoricalPostgresReplayService(
        engine=_Engine()
    ).replay(
        config,
        confirm_restricted_identifier_processing=True,
    )
    second = ProviderHistoricalPostgresReplayService(
        engine=_Engine()
    ).replay(
        config,
        confirm_restricted_identifier_processing=True,
    )

    assert first["replay_id"] == second["replay_id"]
    assert first["safe_feature_rows"] == second["safe_feature_rows"]
    assert first["dp_release_replay_stable"] is True
