import numpy as np
import pytest

from app.services import postgres_vector_store_service as service


class FakeCursor:
    def __init__(self):
        self.executions = []
        self._one = None
        self._all = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        sql_text = str(sql)
        self.executions.append(
            {
                "sql": sql_text,
                "params": params,
            }
        )

        if "FROM audience_vector_models" in sql_text:
            self._one = (
                {
                    "backend": "sklearn_hashing",
                    "dimension": 3,
                },
                3,
                2,
            )

        if "WITH scored AS" in sql_text:
            self._all = [
                (
                    {
                        "trait_text": (
                            "montreal cafe evening"
                        ),
                        "location_name": "montreal",
                    },
                    0.95,
                    0,
                )
            ]

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.cursor_instance


def test_similarity_search_runs_inside_postgres(
    monkeypatch,
):
    connection = FakeConnection()

    monkeypatch.setattr(
        service,
        "ensure_postgres_vector_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        service,
        "_connect",
        lambda: connection,
    )

    result = service.postgres_similarity_search(
        job_id="job_1",
        query_vector=np.asarray(
            [1.0, 0.0, 0.0]
        ),
        top_k=5,
    )

    assert len(result) == 1
    assert result[0]["similarity_score"] == 0.95

    sql = "\n".join(
        item["sql"]
        for item in connection.cursor_instance.executions
    )

    assert "WITH scored AS" in sql
    assert "unnest(av.embedding)" in sql
    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_similarity_search_applies_filters(
    monkeypatch,
):
    connection = FakeConnection()

    monkeypatch.setattr(
        service,
        "ensure_postgres_vector_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        service,
        "_connect",
        lambda: connection,
    )

    service.postgres_similarity_search(
        job_id="job_1",
        query_vector=np.asarray(
            [1.0, 0.0, 0.0]
        ),
        top_k=10,
        location_name="montreal",
        primary_poi_type="cafe",
        created_day_part="evening",
        min_quality=0.4,
    )

    search_execution = next(
        item
        for item in connection.cursor_instance.executions
        if "WITH scored AS" in item["sql"]
    )

    params = search_execution["params"]

    assert params["location_name"] == "montreal"
    assert params["primary_poi_type"] == "cafe"
    assert params["created_day_part"] == "evening"
    assert params["min_quality"] == 0.4


def test_similarity_search_rejects_dimension_mismatch(
    monkeypatch,
):
    connection = FakeConnection()

    monkeypatch.setattr(
        service,
        "ensure_postgres_vector_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        service,
        "_connect",
        lambda: connection,
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        service.postgres_similarity_search(
            job_id="job_1",
            query_vector=np.asarray([1.0, 0.0]),
        )


def test_similarity_search_rejects_zero_vector(
    monkeypatch,
):
    monkeypatch.setattr(
        service,
        "ensure_postgres_vector_schema",
        lambda: None,
    )

    with pytest.raises(
        ValueError,
        match="non-zero norm",
    ):
        service.postgres_similarity_search(
            job_id="job_1",
            query_vector=np.asarray(
                [0.0, 0.0, 0.0]
            ),
        )


def test_schema_readiness_is_cached(
    monkeypatch,
):
    calls = {"schema": 0}

    def fake_schema():
        calls["schema"] += 1

    monkeypatch.setattr(
        service,
        "_SCHEMA_READY",
        False,
    )

    monkeypatch.setattr(
        service,
        "ensure_postgres_vector_schema",
        fake_schema,
    )

    service._ensure_postgres_vector_schema_once()
    service._ensure_postgres_vector_schema_once()

    assert calls["schema"] == 1
