from __future__ import annotations

import pytest

from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDataset,
    ProductionEmbeddingBenchmarkPolicy,
)


def _dataset_payload():
    return {
        "benchmark_id": "global-retrieval-eval",
        "dataset_version": "immutable-dataset-v1",
        "privacy_status": "safe",
        "rights_status": "synthetic_evaluation",
        "contains_raw_identifiers": False,
        "authoring_manifest": {
            "schema_version": "punk-embedding-benchmark-authoring-v1",
            "builder_version": "production-dataset-builder-v1",
            "review_status": "approved",
            "reviewed_by": "benchmark-reviewer",
            "reviewed_at": "2026-07-30T12:00:00+00:00",
            "case_catalog_fingerprint": "a" * 64,
            "document_catalog_fingerprints": ["b" * 64],
        },
        "documents": [
            {
                "document_id": "doc-1",
                "text": "location montreal category restaurant evening",
                "location": "Montréal",
                "category": "Restaurant",
                "daypart": "Evening",
            },
            {
                "document_id": "doc-2",
                "text": "location toronto category cafe morning",
                "location": "Toronto",
                "category": "Cafe",
                "daypart": "Morning",
            },
        ],
        "cases": [
            {
                "case_id": "case-1",
                "query": "evening restaurant visitors in Montreal",
                "language": "en",
                "relevant_document_ids": ["doc-1"],
                "hard_negative_document_ids": ["doc-2"],
                "expected_locations": ["Montréal"],
                "expected_categories": ["Restaurant"],
                "expected_dayparts": ["Evening"],
            }
        ],
    }


def test_dataset_is_canonical_and_content_addressed():
    first = EmbeddingBenchmarkDataset.from_mapping(_dataset_payload())
    payload = _dataset_payload()
    payload["documents"] = list(reversed(payload["documents"]))
    second = EmbeddingBenchmarkDataset.from_mapping(payload)

    assert first.fingerprint == second.fingerprint
    assert first.documents[0].location == "montreal"
    assert first.cases[0].expected_categories == ("restaurant",)


def test_dataset_rejects_duplicate_or_unknown_document_identity():
    duplicate = _dataset_payload()
    duplicate["documents"][1]["document_id"] = "doc-1"
    with pytest.raises(ValueError, match="unique"):
        EmbeddingBenchmarkDataset.from_mapping(duplicate)

    unknown = _dataset_payload()
    unknown["cases"][0]["relevant_document_ids"] = ["not-present"]
    with pytest.raises(ValueError, match="unknown"):
        EmbeddingBenchmarkDataset.from_mapping(unknown)


def test_supported_cases_require_relevance_but_unsupported_cases_do_not():
    invalid = _dataset_payload()
    invalid["cases"][0]["relevant_document_ids"] = []
    with pytest.raises(ValueError, match="require relevant"):
        EmbeddingBenchmarkDataset.from_mapping(invalid)

    invalid["cases"][0]["unsupported_location"] = True
    dataset = EmbeddingBenchmarkDataset.from_mapping(invalid)
    assert dataset.cases[0].unsupported_location is True


def test_dataset_rejects_raw_identifier_or_unapproved_rights_claims():
    unsafe = _dataset_payload()
    unsafe["contains_raw_identifiers"] = True
    with pytest.raises(ValueError, match="raw identifiers"):
        EmbeddingBenchmarkDataset.from_mapping(unsafe)

    unsafe = _dataset_payload()
    unsafe["rights_status"] = "unknown"
    with pytest.raises(ValueError, match="rights"):
        EmbeddingBenchmarkDataset.from_mapping(unsafe)


def test_production_policy_rejects_weaker_thresholds():
    policy = ProductionEmbeddingBenchmarkPolicy()

    with pytest.raises(ValueError, match="weaker"):
        policy.resolve_thresholds({"recall_at_k": 0.1})

    with pytest.raises(ValueError, match="weaker"):
        policy.resolve_thresholds(
            {"unsupported_location_false_match_rate": 0.5}
        )

    with pytest.raises(ValueError, match="weaker"):
        policy.resolve_thresholds({"language_count": 1})
