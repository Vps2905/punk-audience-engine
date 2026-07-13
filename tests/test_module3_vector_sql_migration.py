from pathlib import Path


def test_vector_sql_search_migration_exists():
    path = Path(
        "migrations/"
        "0002_audience_vector_sql_search.sql"
    )

    sql = path.read_text().lower()

    assert "embedding_norm" in sql
    assert "unnest(embedding)" in sql
    assert "set not null" in sql
    assert "idx_audience_vectors_filter" in sql
