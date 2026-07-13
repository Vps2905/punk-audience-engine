import numpy as np

from app.api import embedding_routes
from app.services import embedding_service
from app.services import (
    postgres_vector_store_service as postgres_service,
)


def test_query_ontology_adds_cafe_term():
    normalized = (
        embedding_service.normalize_embedding_query(
            "coffee shop visitors in Montreal"
        )
    )

    assert "coffee shop" in normalized.lower()
    assert "cafe" in normalized.lower()


def test_query_without_known_alias_is_preserved():
    query = "restaurant visitors in Toronto"

    normalized = (
        embedding_service.normalize_embedding_query(
            query
        )
    )

    assert normalized == query


def test_empty_query_is_rejected():
    try:
        embedding_service.normalize_embedding_query(
            "   "
        )
    except ValueError as exc:
        assert "query cannot be empty" in str(exc)
    else:
        raise AssertionError(
            "Empty query should be rejected."
        )


def test_poi_alias_normalizes_to_stored_taxonomy():
    assert (
        postgres_service.canonicalize_poi_type(
            "coffee_shop"
        )
        == "cafe"
    )

    assert (
        postgres_service.canonicalize_poi_type(
            "Coffee Shop"
        )
        == "cafe"
    )

    assert (
        postgres_service.canonicalize_poi_type(
            "fitness centre"
        )
        == "gym"
    )

    assert (
        postgres_service.canonicalize_poi_type(
            "Supermarket"
        )
        == "grocery_store"
    )


def test_unknown_poi_type_is_safely_normalized():
    assert (
        postgres_service.canonicalize_poi_type(
            "Art Gallery"
        )
        == "art_gallery"
    )


def test_search_api_forwards_structured_filters(
    monkeypatch,
):
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)

        return {
            "status": "ok",
            "results": [],
        }

    monkeypatch.setattr(
        embedding_routes,
        "search_similar_audiences",
        fake_search,
    )

    request = (
        embedding_routes.SimilarSearchRequest(
            job_id="job_1",
            query="coffee shop visitors",
            top_k=10,
            location_name="montreal",
            primary_poi_type="coffee_shop",
            created_day_part="evening",
            min_quality=0.4,
        )
    )

    embedding_routes.search_similar(request)

    assert captured["job_id"] == "job_1"
    assert captured["query"] == (
        "coffee shop visitors"
    )
    assert captured["top_k"] == 10
    assert captured["location_name"] == "montreal"
    assert (
        captured["primary_poi_type"]
        == "coffee_shop"
    )
    assert (
        captured["created_day_part"]
        == "evening"
    )
    assert captured["min_quality"] == 0.4


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
                "params": params or {},
            }
        )

        if (
            "FROM audience_vector_models"
            in sql_text
        ):
            self._one = (
                {
                    "backend": "sklearn_hashing",
                },
                3,
                1,
            )

        if "WITH scored AS" in sql_text:
            self._all = [
                (
                    {
                        "vector_index": 0,
                        "primary_poi_type": "cafe",
                        "location_name": "montreal",
                        "created_day_part": "evening",
                    },
                    0.9,
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


def test_sql_search_canonicalizes_poi_alias(
    monkeypatch,
):
    connection = FakeConnection()

    monkeypatch.setattr(
        postgres_service,
        "_SCHEMA_READY",
        True,
    )

    monkeypatch.setattr(
        postgres_service,
        "_connect",
        lambda: connection,
    )

    results = (
        postgres_service.postgres_similarity_search(
            job_id="job_1",
            query_vector=np.asarray(
                [1.0, 0.0, 0.0]
            ),
            primary_poi_type="coffee_shop",
        )
    )

    search_call = next(
        call
        for call in (
            connection.cursor_instance.executions
        )
        if "WITH scored AS" in call["sql"]
    )

    assert (
        search_call["params"][
            "primary_poi_type"
        ]
        == "cafe"
    )

    assert len(results) == 1
    assert (
        results[0]["primary_poi_type"]
        == "cafe"
    )
