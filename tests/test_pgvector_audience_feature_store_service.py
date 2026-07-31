from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)


class FakeResult:
    def __init__(self, *, scalar_value=None, rows=None):
        self._scalar_value = scalar_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar_value

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append((sql, params or {}))
        if "to_regclass" in sql:
            return FakeResult(scalar_value=True)
        if "WITH vector_ranked AS" in sql:
            return FakeResult(
                rows=[
                    {
                        "feature_id": "feature_1",
                        "location_name": "montreal_downtown",
                        "primary_poi_type": "restaurant",
                        "created_day_part": "evening",
                        "lookback_bucket": "8_30d",
                        "cohort_size": 4200,
                        "quality_score": 0.82,
                        "privacy_status": "passed",
                        "rights_status": "historical_internal_only",
                        "purpose": "internal_evaluation",
                        "source_latest_at": None,
                        "freshness_status": "stale",
                        "data_use_mode": "historical_preview",
                        "eligible_for_activation": False,
                        "metadata": {},
                        "vector_score": 0.91,
                        "lexical_score": 0.4,
                        "vector_rank": 1,
                        "lexical_rank": 1,
                        "fused_score": 2 / 61,
                    }
                ]
            )
        return FakeResult()


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_hybrid_sql_applies_safety_and_structured_filters_before_ranking():
    engine = FakeEngine()
    store = PgvectorAudienceFeatureStore(engine=engine)
    vector = [0.0] * 384
    vector[0] = 1.0

    rows = store.hybrid_search(
        tenant_id="tenant-a",
        feature_set_id="feature-set-1",
        feature_set_version=1,
        query_text="evening restaurant visitors",
        query_embedding=vector,
        execution_mode="historical_preview",
        locations=["Montréal"],
        categories=["Restaurant"],
        dayparts=["Evening"],
        exclusions=["Casino"],
        top_k=10,
    )

    search_sql, params = next(
        call
        for call in engine.connection.calls
        if "WITH vector_ranked AS" in call[0]
    )
    assert "afv.tenant_id = :tenant_id" in search_sql
    assert "afv.eligible_for_retrieval = TRUE" in search_sql
    assert "afv.privacy_status IN" in search_sql
    assert "plainto_tsquery" in search_sql
    assert "primary_poi_type = ANY" in search_sql
    assert "created_day_part = ANY" in search_sql
    assert "NOT (afv.primary_poi_type = ANY" in search_sql
    assert "<=> CAST(:query_embedding AS vector(384))" in search_sql
    assert "websearch_to_tsquery" in search_sql
    assert params["locations"] == ["montreal"]
    assert params["categories"] == ["restaurant"]
    assert params["dayparts"] == ["evening"]
    assert params["exclusions"] == ["casino"]
    assert rows[0]["feature_id"] == "feature_1"


def test_production_search_adds_freshness_and_activation_filters():
    engine = FakeEngine()
    store = PgvectorAudienceFeatureStore(engine=engine)
    vector = [0.0] * 384
    vector[0] = 1.0

    store.hybrid_search(
        tenant_id="tenant-a",
        feature_set_id="feature-set-1",
        feature_set_version=1,
        query_text="audience",
        query_embedding=vector,
        execution_mode="production",
        locations=[],
        categories=[],
        dayparts=[],
        exclusions=[],
        top_k=5,
    )

    search_sql = next(
        sql
        for sql, _ in engine.connection.calls
        if "WITH vector_ranked AS" in sql
    )
    assert "afv.eligible_for_activation = TRUE" in search_sql
    assert "afv.freshness_status = 'fresh'" in search_sql
    assert "afv.data_use_mode = 'production'" in search_sql


def test_every_store_operation_sets_tenant_context_before_search():
    engine = FakeEngine()
    store = PgvectorAudienceFeatureStore(engine=engine)
    vector = [0.0] * 384
    vector[0] = 1.0

    store.hybrid_search(
        tenant_id="tenant-a",
        feature_set_id="feature-set-1",
        feature_set_version=1,
        query_text="audience",
        query_embedding=vector,
        execution_mode="historical_preview",
        locations=[],
        categories=[],
        dayparts=[],
        exclusions=[],
        top_k=5,
    )

    sql_calls = [sql for sql, _ in engine.connection.calls]
    tenant_context_index = next(
        index
        for index, sql in enumerate(sql_calls)
        if "set_config" in sql
    )
    search_index = next(
        index
        for index, sql in enumerate(sql_calls)
        if "WITH vector_ranked AS" in sql
    )
    assert tenant_context_index < search_index


def test_exact_baseline_disables_ann_index_scans():
    engine = FakeEngine()
    store = PgvectorAudienceFeatureStore(engine=engine)
    vector = [0.0] * 384
    vector[0] = 1.0

    store.hybrid_search(
        tenant_id="tenant-a",
        feature_set_id="feature-set-1",
        feature_set_version=1,
        query_text="audience",
        query_embedding=vector,
        execution_mode="historical_preview",
        locations=[],
        categories=[],
        dayparts=[],
        exclusions=[],
        top_k=5,
        search_strategy="exact",
    )

    sql_calls = [sql for sql, _ in engine.connection.calls]
    assert any("enable_indexscan = off" in sql for sql in sql_calls)
    assert any("enable_bitmapscan = off" in sql for sql in sql_calls)
    assert not any("hnsw.ef_search" in sql for sql in sql_calls)


def test_ann_search_sets_bounded_hnsw_search_breadth(monkeypatch):
    monkeypatch.setenv("PGVECTOR_HNSW_EF_SEARCH", "5000")
    engine = FakeEngine()
    store = PgvectorAudienceFeatureStore(engine=engine)
    vector = [0.0] * 384
    vector[0] = 1.0

    store.hybrid_search(
        tenant_id="tenant-a",
        feature_set_id="feature-set-1",
        feature_set_version=1,
        query_text="audience",
        query_embedding=vector,
        execution_mode="historical_preview",
        locations=[],
        categories=[],
        dayparts=[],
        exclusions=[],
        top_k=5,
    )

    _, params = next(
        call
        for call in engine.connection.calls
        if "hnsw.ef_search" in call[0]
    )
    assert params["ef_search"] == "1000"
