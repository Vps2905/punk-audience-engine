from __future__ import annotations

import pytest

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkCaseCatalog,
    EmbeddingBenchmarkDocumentCatalog,
)


def _document_catalog_payload():
    return {
        "catalog_id": "safe-feature-catalog",
        "catalog_version": "v1",
        "source_type": "curated_synthetic_features",
        "source_fingerprint": "a" * 64,
        "privacy_status": "safe",
        "rights_status": "synthetic_evaluation",
        "contains_raw_identifiers": False,
        "documents": [
            {
                "document_id": "doc-1",
                "text": "location alpha category restaurant evening",
                "location": "alpha",
                "category": "restaurant",
                "daypart": "evening",
            }
        ],
        "lineage": {
            "generator": "reviewed-synthetic-catalog-v1",
            "review_status": "approved",
            "reviewed_by": "model-owner",
            "reviewed_at": "2026-07-30T12:00:00+00:00",
            "audience_volume_claimed": False,
            "document_source_lineage": {
                "doc-1": "synthetic-source-doc-1",
            },
        },
    }


def _case_catalog_payload():
    return {
        "catalog_id": "gold-query-catalog",
        "catalog_version": "v1",
        "review_status": "approved",
        "reviewed_by": "model-owner",
        "reviewed_at": "2026-07-30T12:00:00+00:00",
        "cases": [
            {
                "case_id": "case-1",
                "query": "evening restaurant visitors in alpha",
                "language": "en",
                "relevant_document_ids": ["doc-1"],
                "expected_locations": ["alpha"],
                "expected_categories": ["restaurant"],
                "expected_dayparts": ["evening"],
            }
        ],
    }


def test_authoring_catalogs_are_content_addressed_and_reviewed():
    documents = EmbeddingBenchmarkDocumentCatalog.from_mapping(
        _document_catalog_payload()
    )
    cases = EmbeddingBenchmarkCaseCatalog.from_mapping(
        _case_catalog_payload()
    )

    assert len(documents.fingerprint) == 64
    assert len(cases.fingerprint) == 64
    assert documents.contains_raw_identifiers is False
    assert cases.review_status == "approved"
    assert cases.reviewed_by == "model_owner"


def test_document_catalog_rejects_raw_data_or_unknown_source():
    unsafe = _document_catalog_payload()
    unsafe["contains_raw_identifiers"] = True
    with pytest.raises(ValueError, match="raw identifiers"):
        EmbeddingBenchmarkDocumentCatalog.from_mapping(unsafe)

    unsafe = _document_catalog_payload()
    unsafe["source_type"] = "provider_raw_events"
    with pytest.raises(ValueError, match="source_type"):
        EmbeddingBenchmarkDocumentCatalog.from_mapping(unsafe)


def test_case_catalog_requires_named_approved_review():
    unreviewed = _case_catalog_payload()
    unreviewed["review_status"] = "pending"
    with pytest.raises(ValueError, match="not approved"):
        EmbeddingBenchmarkCaseCatalog.from_mapping(unreviewed)

    unreviewed = _case_catalog_payload()
    unreviewed["reviewed_by"] = ""
    with pytest.raises(ValueError, match="named reviewer"):
        EmbeddingBenchmarkCaseCatalog.from_mapping(unreviewed)


def test_curated_document_catalog_requires_review_and_unique_lineage():
    unreviewed = _document_catalog_payload()
    unreviewed["lineage"]["review_status"] = "pending"
    with pytest.raises(ValueError, match="approved human review"):
        EmbeddingBenchmarkDocumentCatalog.from_mapping(unreviewed)

    missing_reviewer = _document_catalog_payload()
    missing_reviewer["lineage"]["reviewed_by"] = ""
    with pytest.raises(ValueError, match="named reviewer"):
        EmbeddingBenchmarkDocumentCatalog.from_mapping(missing_reviewer)

    volume_claim = _document_catalog_payload()
    volume_claim["lineage"]["audience_volume_claimed"] = True
    with pytest.raises(ValueError, match="cannot claim audience volume"):
        EmbeddingBenchmarkDocumentCatalog.from_mapping(volume_claim)

    missing_lineage = _document_catalog_payload()
    missing_lineage["lineage"]["document_source_lineage"] = {}
    with pytest.raises(ValueError, match="cover every benchmark document"):
        EmbeddingBenchmarkDocumentCatalog.from_mapping(missing_lineage)


def test_pgvector_document_catalog_does_not_require_curated_review_manifest():
    payload = _document_catalog_payload()
    payload["source_type"] = "pgvector_privacy_safe_features"
    payload["rights_status"] = "offline_evaluation_only"
    payload["lineage"] = {
        "feature_set_id": "feature-set-1",
        "feature_set_version": 1,
    }

    catalog = EmbeddingBenchmarkDocumentCatalog.from_mapping(payload)

    assert catalog.source_type == "pgvector_privacy_safe_features"
