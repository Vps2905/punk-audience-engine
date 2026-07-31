from __future__ import annotations

import copy

import pytest

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkCaseCatalog,
    EmbeddingBenchmarkDocumentCatalog,
)
from app.services.production_embedding_benchmark_dataset_service import (
    PgvectorEmbeddingBenchmarkCatalogSourceService,
    ProductionEmbeddingBenchmarkDatasetBuilderService,
)
from tests.test_production_embedding_benchmark_service import (
    production_benchmark_dataset,
)


def _document_catalog(documents=None, **overrides):
    dataset = production_benchmark_dataset()
    values = {
        "catalog_id": "global-safe-catalog",
        "catalog_version": "v1",
        "source_type": "curated_synthetic_features",
        "source_fingerprint": "a" * 64,
        "privacy_status": "safe",
        "rights_status": "synthetic_evaluation",
        "contains_raw_identifiers": False,
        "documents": (
            documents
            if documents is not None
            else [value.to_dict() for value in dataset.documents]
        ),
        "lineage": {"generator": "unit-test"},
    }
    values.update(overrides)
    return EmbeddingBenchmarkDocumentCatalog.from_mapping(values)


def _case_catalog(cases=None, **overrides):
    dataset = production_benchmark_dataset()
    values = {
        "catalog_id": "global-gold-cases",
        "catalog_version": "v1",
        "review_status": "approved",
        "reviewed_by": "model-owner",
        "reviewed_at": "2026-07-30T12:00:00+00:00",
        "cases": (
            cases
            if cases is not None
            else [value.to_dict() for value in dataset.cases]
        ),
    }
    values.update(overrides)
    return EmbeddingBenchmarkCaseCatalog.from_mapping(values)


def test_builder_produces_reviewed_ready_content_addressed_dataset():
    dataset, readiness = (
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="punk-global-evaluation",
            dataset_version="immutable-v1",
            document_catalogs=[_document_catalog()],
            case_catalog=_case_catalog(),
        )
    )

    assert readiness["status"] == "ready_for_model_evaluation"
    assert readiness["ready_for_model_evaluation"] is True
    assert readiness["gold_label_integrity"] == "passed"
    assert readiness["coverage"]["case_count"] == 100
    assert readiness["coverage"]["document_count"] == 100
    assert dataset.authoring_manifest["reviewed_by"] == "model_owner"
    assert dataset.contains_raw_identifiers is False
    assert len(dataset.fingerprint) == 64


def test_builder_blocks_insufficient_coverage_without_faking_readiness():
    source = production_benchmark_dataset()
    documents = [value.to_dict() for value in source.documents[:10]]
    known = {value["document_id"] for value in documents}
    cases = []
    for value in source.cases[:5]:
        record = value.to_dict()
        record["relevant_document_ids"] = [
            item
            for item in record["relevant_document_ids"]
            if item in known
        ]
        record["hard_negative_document_ids"] = [
            item
            for item in record["hard_negative_document_ids"]
            if item in known
        ]
        cases.append(record)

    _, readiness = (
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="small-evaluation",
            dataset_version="immutable-small-v1",
            document_catalogs=[
                _document_catalog(documents=documents)
            ],
            case_catalog=_case_catalog(cases=cases),
        )
    )

    assert readiness["status"] == (
        "blocked_insufficient_benchmark_coverage"
    )
    assert readiness["ready_for_model_evaluation"] is False
    assert (
        readiness["coverage_results"]["case_count"]["passed"]
        is False
    )


def test_builder_rejects_relevant_document_constraint_conflict():
    cases = [
        production_benchmark_dataset().cases[0].to_dict()
    ]
    cases[0]["expected_locations"] = ["wrong_location"]

    with pytest.raises(ValueError, match="conflicts"):
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="invalid-labels",
            dataset_version="v1",
            document_catalogs=[_document_catalog()],
            case_catalog=_case_catalog(cases=cases),
        )


def test_builder_rejects_hard_negative_that_satisfies_all_constraints():
    case = production_benchmark_dataset().cases[0].to_dict()
    case["relevant_document_ids"] = ["doc-000"]
    case["hard_negative_document_ids"] = ["doc-001"]

    with pytest.raises(ValueError, match="Hard-negative"):
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="invalid-hard-negative",
            dataset_version="v1",
            document_catalogs=[_document_catalog()],
            case_catalog=_case_catalog(cases=[case]),
        )


def test_builder_rejects_unsupported_location_present_in_catalog():
    case = production_benchmark_dataset().cases[-1].to_dict()
    case["expected_locations"] = ["target_location"]

    with pytest.raises(ValueError, match="Unsupported-location"):
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="invalid-unsupported-location",
            dataset_version="v1",
            document_catalogs=[_document_catalog()],
            case_catalog=_case_catalog(cases=[case]),
        )


def test_builder_rejects_inconsistent_multilingual_gold_labels():
    cases = [
        value.to_dict()
        for value in production_benchmark_dataset().cases[:2]
    ]
    cases[0]["relevant_document_ids"] = ["doc-000"]
    cases[1]["relevant_document_ids"] = ["doc-001"]

    with pytest.raises(ValueError, match="inconsistent gold labels"):
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="invalid-multilingual",
            dataset_version="v1",
            document_catalogs=[_document_catalog()],
            case_catalog=_case_catalog(cases=cases),
        )


def test_builder_rejects_duplicate_document_ids_across_catalogs():
    with pytest.raises(ValueError, match="unique across catalogs"):
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id="duplicate-documents",
            dataset_version="v1",
            document_catalogs=[
                _document_catalog(),
                _document_catalog(
                    catalog_id="second-catalog",
                    source_fingerprint="b" * 64,
                ),
            ],
            case_catalog=_case_catalog(),
        )


class FakeResult:
    def __init__(self, *, scalar=None, row=None, rows=None):
        self._scalar = scalar
        self._row = row
        self._rows = rows or []

    def scalar(self):
        return self._scalar

    def mappings(self):
        return self

    def first(self):
        return self._row

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self):
        self.calls = []
        self.feature_set = {
            "tenant_id": "tenant_a",
            "feature_set_id": "feature-set-1",
            "version": 1,
            "source_fingerprint": "f" * 64,
            "data_use_mode": "historical_preview",
            "freshness_status": "stale",
            "privacy_policy_version": "privacy-v1",
            "rights_policy_id": "rights-v1",
            "purpose": "offline-evaluation",
            "eligible_for_retrieval": True,
            "feature_count": 2,
        }
        self.rows = [
            {
                "feature_id": "feature-1",
                "location_name": "alpha",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "trait_text": "location alpha category restaurant evening",
                "privacy_status": "safe",
                "rights_status": "historical_internal_only",
            },
            {
                "feature_id": "feature-2",
                "location_name": "beta",
                "primary_poi_type": "cafe",
                "created_day_part": "morning",
                "trait_text": "location beta category cafe morning",
                "privacy_status": "safe",
                "rights_status": "historical_internal_only",
            },
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append((sql, params or {}))
        if "to_regclass" in sql:
            return FakeResult(scalar=True)
        if "FROM audience_feature_sets" in sql:
            return FakeResult(row=self.feature_set)
        if "FROM audience_feature_vectors" in sql:
            return FakeResult(rows=self.rows)
        return FakeResult()


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_pgvector_catalog_export_reads_only_safe_trait_fields():
    engine = FakeEngine()
    catalog = PgvectorEmbeddingBenchmarkCatalogSourceService(
        engine=engine
    ).export_catalog(
        tenant_id="tenant-a",
        feature_set_id="feature-set-1",
        feature_set_version=1,
        catalog_id="historical-safe-features",
        catalog_version="v1",
    )

    assert len(catalog.documents) == 2
    assert catalog.rights_status == "offline_evaluation_only"
    assert catalog.documents[0].document_id == "feature:feature-1"
    assert catalog.lineage["embeddings_exported"] is False
    assert catalog.lineage["cohort_sizes_exported"] is False
    sql = "\n".join(value for value, _ in engine.connection.calls)
    vector_select = sql.split("FROM audience_feature_vectors")[0]
    vector_select = vector_select.rsplit("SELECT", 1)[-1]
    assert "embedding" not in vector_select
    assert "cohort_size" not in vector_select


def test_pgvector_catalog_export_fails_on_count_or_rights_mismatch():
    count_engine = FakeEngine()
    count_engine.connection.feature_set["feature_count"] = 3
    with pytest.raises(RuntimeError, match="count"):
        PgvectorEmbeddingBenchmarkCatalogSourceService(
            engine=count_engine
        ).export_catalog(
            tenant_id="tenant-a",
            feature_set_id="feature-set-1",
            feature_set_version=1,
            catalog_id="catalog",
            catalog_version="v1",
        )

    rights_engine = FakeEngine()
    rights_engine.connection.rows[0] = {
        **copy.deepcopy(rights_engine.connection.rows[0]),
        "rights_status": "unknown",
    }
    with pytest.raises(RuntimeError, match="rights"):
        PgvectorEmbeddingBenchmarkCatalogSourceService(
            engine=rights_engine
        ).export_catalog(
            tenant_id="tenant-a",
            feature_set_id="feature-set-1",
            feature_set_version=1,
            catalog_id="catalog",
            catalog_version="v1",
        )
