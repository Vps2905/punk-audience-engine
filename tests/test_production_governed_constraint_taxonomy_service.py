from __future__ import annotations

from copy import deepcopy

import pytest

from app.models.embedding_benchmark_contracts import EmbeddingBenchmarkDataset
from app.services.production_governed_constraint_taxonomy_service import (
    attach_governed_taxonomy_to_dataset,
    build_engineering_multilingual_taxonomy_artifacts,
    load_governed_constraint_taxonomy,
)


def _dataset() -> dict:
    return {
        "benchmark_id": "module2-taxonomy-test",
        "dataset_version": "immutable-v1",
        "privacy_status": "safe",
        "rights_status": "offline_evaluation_only",
        "contains_raw_identifiers": False,
        "authoring_manifest": {
            "schema_version": "punk-embedding-benchmark-authoring-v1",
            "builder_version": "production-dataset-builder-v1",
            "review_status": "approved",
            "reviewed_by": "taxonomy-test-owner",
            "reviewed_at": "2026-08-01T00:00:00+00:00",
            "case_catalog_fingerprint": "b" * 64,
            "document_catalog_fingerprints": ["c" * 64],
        },
        "native_human_signoff_complete": False,
        "documents": [
            {
                "document_id": "doc-carwash",
                "text": "location montreal category car wash daypart afternoon",
                "location": "montreal",
                "category": "car_wash",
                "daypart": "afternoon",
            },
            {
                "document_id": "doc-restaurant",
                "text": "location montreal category restaurant daypart evening",
                "location": "montreal",
                "category": "restaurant",
                "daypart": "evening",
            },
        ],
        "cases": [
            {
                "case_id": "supported-bn",
                "query": "Montreal-এ দুপুরে কার ওয়াশ অডিয়েন্স",
                "language": "bn",
                "expected_locations": ["montreal"],
                "expected_categories": ["car_wash"],
                "expected_dayparts": ["afternoon"],
                "relevant_document_ids": ["doc-carwash"],
                "unsupported_location": False,
            },
            {
                "case_id": "supported-en",
                "query": "Montreal restaurant audience in the evening",
                "language": "en",
                "expected_locations": ["montreal"],
                "expected_categories": ["restaurant"],
                "expected_dayparts": ["evening"],
                "relevant_document_ids": ["doc-restaurant"],
                "unsupported_location": False,
            },
        ],
    }


def _language_pack() -> dict:
    return {
        "pack_id": "module2-taxonomy-test-pack",
        "pack_version": "reviewed-v1",
        "review_status": "approved",
        "reviewed_by": "ai-review-delegated-by-taxonomy-owner",
        "translation_review_completed": True,
        "language_order": ["bn", "en"],
        "languages": {
            "bn": {
                "category_labels": {
                    "car_wash": "কার ওয়াশ",
                    "restaurant": "রেস্তোরাঁ",
                },
                "daypart_labels": {
                    "afternoon": "দুপুরে",
                    "evening": "সন্ধ্যায়",
                },
            },
            "en": {
                "category_labels": {
                    "car_wash": "car wash",
                    "restaurant": "restaurant",
                },
                "daypart_labels": {
                    "afternoon": "afternoon",
                    "evening": "evening",
                },
            },
        },
        "lineage": {
            "approval_scope": "engineering_benchmark_generation_only",
            "native_human_review_completed": False,
            "machine_translation_auto_approved": False,
            "requires_native_language_review": True,
            "requires_named_human_review": True,
            "language_count": 2,
            "production_certification_status": (
                "pending_native_human_signoff"
            ),
            "review_method": "ai_assisted_multilingual_quality_review",
            "observed_category_count": 2,
            "observed_daypart_count": 2,
        },
    }


def test_builder_preserves_dataset_identity_and_records_non_oracle_lineage():
    dataset = _dataset()
    expected_fingerprint = EmbeddingBenchmarkDataset.from_mapping(
        dataset
    ).fingerprint

    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=dataset,
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )

    assert artifacts.canonical_dataset_fingerprint == expected_fingerprint
    assert EmbeddingBenchmarkDataset.from_mapping(
        artifacts.dataset_envelope
    ).fingerprint == expected_fingerprint
    assert artifacts.taxonomy_payload["lineage"][
        "oracle_case_constraints_used"
    ] is False
    assert artifacts.taxonomy_payload["lineage"][
        "native_human_review_completed"
    ] is False
    assert artifacts.taxonomy_payload["lineage"][
        "source_language_pack_sha256"
    ] == "a" * 64
    assert artifacts.taxonomy_payload["lineage"][
        "source_dataset_fingerprint"
    ] == expected_fingerprint
    assert artifacts.category_count == 2
    assert artifacts.daypart_count == 2


def test_builder_requires_complete_reviewed_labels_for_every_dataset_value():
    pack = _language_pack()
    del pack["languages"]["bn"]["category_labels"]["restaurant"]

    with pytest.raises(
        ValueError,
        match="Missing reviewed taxonomy label",
    ):
        build_engineering_multilingual_taxonomy_artifacts(
            dataset_payload=_dataset(),
            language_pack=pack,
            language_pack_sha256="a" * 64,
        )


def test_loader_rejects_declared_taxonomy_fingerprint_mismatch():
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=_dataset(),
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )
    payload = deepcopy(artifacts.taxonomy_payload)
    payload["taxonomy_fingerprint"] = "0" * 64

    with pytest.raises(
        ValueError,
        match="Declared taxonomy_fingerprint does not match",
    ):
        load_governed_constraint_taxonomy(
            payload,
            allow_engineering_scope=True,
            require_native_human_review=False,
        )


def test_loader_rejects_oracle_case_constraint_lineage():
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=_dataset(),
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )
    payload = deepcopy(artifacts.taxonomy_payload)
    payload["lineage"]["oracle_case_constraints_used"] = True

    with pytest.raises(ValueError, match="oracle case constraints"):
        load_governed_constraint_taxonomy(
            payload,
            allow_engineering_scope=True,
            require_native_human_review=False,
        )


def test_production_scope_requires_native_human_review():
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=_dataset(),
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )
    payload = deepcopy(artifacts.taxonomy_payload)
    payload["lineage"]["approval_scope"] = "production_retrieval"

    with pytest.raises(ValueError, match="native-human review"):
        load_governed_constraint_taxonomy(
            payload,
            allow_engineering_scope=False,
            require_native_human_review=True,
        )


def test_attachment_does_not_mutate_original_dataset_or_identity():
    dataset = _dataset()
    original = deepcopy(dataset)
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=dataset,
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )

    envelope = attach_governed_taxonomy_to_dataset(
        dataset_payload=dataset,
        taxonomy_payload=artifacts.taxonomy_payload,
    )

    assert dataset == original
    assert "governed_constraint_taxonomy" not in dataset
    assert "governed_constraint_taxonomy" in envelope
    assert EmbeddingBenchmarkDataset.from_mapping(envelope).fingerprint == (
        EmbeddingBenchmarkDataset.from_mapping(dataset).fingerprint
    )

    envelope["documents"][0]["location"] = "changed"
    envelope["governed_constraint_taxonomy"]["lineage"][
        "approval_scope"
    ] = "changed"
    assert dataset == original
    assert artifacts.taxonomy_payload["lineage"]["approval_scope"] == (
        "engineering_benchmark_only"
    )


def test_loader_rejects_missing_language_pack_checksum_lineage():
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=_dataset(),
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )
    payload = deepcopy(artifacts.taxonomy_payload)
    del payload["lineage"]["source_language_pack_sha256"]

    with pytest.raises(ValueError, match="source_language_pack_sha256"):
        load_governed_constraint_taxonomy(
            payload,
            allow_engineering_scope=True,
            require_native_human_review=False,
        )


def test_attachment_rejects_taxonomy_built_for_different_dataset():
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=_dataset(),
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )
    other_dataset = _dataset()
    other_dataset["dataset_version"] = "immutable-v2"

    with pytest.raises(ValueError, match="different benchmark dataset"):
        attach_governed_taxonomy_to_dataset(
            dataset_payload=other_dataset,
            taxonomy_payload=artifacts.taxonomy_payload,
        )


def test_production_scope_requires_explicit_production_certification():
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=_dataset(),
        language_pack=_language_pack(),
        language_pack_sha256="a" * 64,
    )
    payload = deepcopy(artifacts.taxonomy_payload)
    payload["lineage"].update(
        {
            "approval_scope": "production_retrieval",
            "native_human_review_completed": True,
            "production_certification_status": "review_complete",
        }
    )

    with pytest.raises(ValueError, match="explicitly certified"):
        load_governed_constraint_taxonomy(
            payload,
            allow_engineering_scope=False,
            require_native_human_review=True,
        )
