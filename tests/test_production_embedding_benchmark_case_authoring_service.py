from __future__ import annotations

from collections import Counter

import pytest

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkDocumentCatalog,
)
from app.models.embedding_benchmark_case_authoring_contracts import (
    EmbeddingBenchmarkLanguagePack,
)
from app.services.production_embedding_benchmark_case_authoring_service import (
    ProductionEmbeddingBenchmarkCaseAuthoringService,
)


def _catalog() -> EmbeddingBenchmarkDocumentCatalog:
    documents = []
    source_lineage = {}
    constraints = ("category", "location", "daypart")
    dayparts = ("morning", "afternoon", "evening", "night")
    for index in range(45):
        violated = constraints[index % len(constraints)]
        anchor = {
            "document_id": f"anchor-{index:02d}",
            "text": f"Safe anchor {index}",
            "location": f"city-{index:02d}",
            "category": f"category-{index:02d}",
            "daypart": dayparts[index % len(dayparts)],
        }
        negative = dict(anchor)
        negative["document_id"] = f"negative-{index:02d}"
        negative["text"] = f"Safe negative {index}"
        if violated == "category":
            negative["category"] = f"negative-category-{index:02d}"
        elif violated == "location":
            negative["location"] = f"negative-city-{index:02d}"
        else:
            negative["daypart"] = dayparts[
                (index + 1) % len(dayparts)
            ]
        documents.extend([anchor, negative])
        source_lineage[anchor["document_id"]] = (
            f"source:{anchor['document_id']}"
        )
        source_lineage[negative["document_id"]] = (
            f"source:{negative['document_id']}"
        )
    for index in range(10):
        filler = {
            "document_id": f"filler-{index:02d}",
            "text": f"Safe filler {index}",
            "location": f"filler-city-{index:02d}",
            "category": f"filler-category-{index:02d}",
            "daypart": dayparts[index % len(dayparts)],
        }
        documents.append(filler)
        source_lineage[filler["document_id"]] = (
            f"source:{filler['document_id']}"
        )
    return EmbeddingBenchmarkDocumentCatalog.from_mapping(
        {
            "catalog_id": "safe-documents",
            "catalog_version": "immutable-v1",
            "source_type": "curated_synthetic_features",
            "source_fingerprint": "a" * 64,
            "privacy_status": "safe",
            "rights_status": "synthetic_evaluation",
            "contains_raw_identifiers": False,
            "documents": documents,
            "lineage": {
                "review_status": "approved",
                "reviewed_by": "document-reviewer",
                "reviewed_at": "2026-08-04T12:00:00+00:00",
                "audience_volume_claimed": False,
                "document_source_lineage": source_lineage,
            },
        }
    )


def _language_pack(catalog: EmbeddingBenchmarkDocumentCatalog):
    categories = sorted({value.category for value in catalog.documents})
    dayparts = sorted({value.daypart for value in catalog.documents})
    languages = {}
    for language in ("en", "fr", "es", "hi", "bn"):
        languages[language] = {
            "supported_query_template": (
                "Find {category} in {location} during {daypart}."
            ),
            "unsupported_query_template": "Find in {location}.",
            "category_labels": {
                value: f"{language}-{value}"
                for value in categories
            },
            "daypart_labels": {
                value: f"{language}-{value}"
                for value in dayparts
            },
        }
    return EmbeddingBenchmarkLanguagePack.from_mapping(
        {
            "pack_id": "reviewed-language-pack",
            "pack_version": "immutable-v1",
            "review_status": "approved",
            "reviewed_by": "language-reviewer",
            "reviewed_at": "2026-08-04T12:00:00+00:00",
            "translation_review_completed": True,
            "language_order": ["en", "fr", "es", "hi", "bn"],
            "languages": languages,
            "unsupported_location_labels": {
                value: {
                    language: value.replace("_", " ").title()
                    for language in languages
                }
                for value in (
                    "brindlehaven",
                    "cedar_harbor",
                    "dovewood",
                    "rivermere",
                )
            },
            "lineage": {"purpose": "unit-test"},
        }
    )


def _plan(catalog: EmbeddingBenchmarkDocumentCatalog) -> dict:
    candidates = []
    for index in range(45):
        candidates.append(
            {
                "anchor_document_id": f"anchor-{index:02d}",
                "hard_negative_document_id": f"negative-{index:02d}",
                "violated_constraint": (
                    "category",
                    "location",
                    "daypart",
                )[index % 3],
                "shared_constraints": [],
                "requires_human_review": True,
            }
        )
    return {
        "status": "ready_for_human_case_authoring",
        "policy_id": "punk-global-embedding-policy-v2",
        "document_catalog_fingerprints": [catalog.fingerprint],
        "grounded_hard_negative_candidates": candidates,
        "blockers": [],
        "plan_fingerprint": "b" * 64,
    }


def test_case_authoring_builds_balanced_pending_review_catalog():
    catalog = _catalog()
    draft, audit = (
        ProductionEmbeddingBenchmarkCaseAuthoringService().author(
            document_catalogs=[catalog],
            curation_plan=_plan(catalog),
            language_pack=_language_pack(catalog),
            catalog_id="global-cases",
            catalog_version="draft-v1",
        )
    )

    assert draft["review_status"] == "pending"
    assert draft["reviewed_by"] == ""
    assert draft["reviewed_at"] is None
    assert len(draft["cases"]) == 100
    assert audit["status"] == "ready_for_human_case_review"
    assert audit["language_counts"] == {
        "bn": 20,
        "en": 20,
        "es": 20,
        "fr": 20,
        "hi": 20,
    }
    assert audit["unsupported_location_case_count"] == 20
    assert audit["hard_negative_case_count"] == 80
    assert audit["multilingual_group_count"] == 10
    assert audit["distinct_supported_anchor_count"] == 40
    assert audit["hard_negative_violation_counts"] == {
        "category": 30,
        "daypart": 25,
        "location": 25,
    }
    assert audit["fallback_translation_case_count"] == 0
    assert audit["placeholder_unsupported_case_count"] == 0
    assert audit["gold_labels_auto_approved"] is False
    assert all(
        "unsupported" not in value["query"].lower()
        and "_" not in value["query"]
        for value in draft["cases"]
        if value["unsupported_location"]
    )


def test_case_authoring_rejects_missing_translation_coverage():
    catalog = _catalog()
    payload = _language_pack(catalog).to_dict()
    del payload["languages"]["fr"]["category_labels"]["category_00"]
    pack = EmbeddingBenchmarkLanguagePack.from_mapping(payload)

    with pytest.raises(ValueError, match="does not cover all"):
        ProductionEmbeddingBenchmarkCaseAuthoringService().author(
            document_catalogs=[catalog],
            curation_plan=_plan(catalog),
            language_pack=pack,
            catalog_id="global-cases",
            catalog_version="draft-v1",
        )


def test_case_authoring_rejects_unbalanced_candidate_supply():
    catalog = _catalog()
    plan = _plan(catalog)
    plan["grounded_hard_negative_candidates"] = [
        value
        for value in plan["grounded_hard_negative_candidates"]
        if value["violated_constraint"] == "category"
    ]

    with pytest.raises(ValueError, match="balanced location"):
        ProductionEmbeddingBenchmarkCaseAuthoringService().author(
            document_catalogs=[catalog],
            curation_plan=plan,
            language_pack=_language_pack(catalog),
            catalog_id="global-cases",
            catalog_version="draft-v1",
        )
