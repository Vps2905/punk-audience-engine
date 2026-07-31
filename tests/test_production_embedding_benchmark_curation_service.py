from __future__ import annotations

import copy

import pytest

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkDocumentCatalog,
)
from app.services.production_embedding_benchmark_curation_service import (
    ProductionEmbeddingBenchmarkCurationService,
)


def _catalog(
    *,
    catalog_id: str = "safe-catalog",
    start: int = 0,
    count: int = 100,
) -> EmbeddingBenchmarkDocumentCatalog:
    dayparts = ("morning", "afternoon", "evening", "night")
    documents = []
    for offset in range(count):
        index = start + offset
        documents.append(
            {
                "document_id": f"document-{index:03d}",
                "text": f"Safe aggregate trait document {index}",
                "location": f"location-{index % 10:02d}",
                "category": f"category-{index % 10:02d}",
                "daypart": dayparts[index % len(dayparts)],
            }
        )
    return EmbeddingBenchmarkDocumentCatalog.from_mapping(
        {
            "catalog_id": catalog_id,
            "catalog_version": "immutable-v1",
            "source_type": "curated_synthetic_features",
            "source_fingerprint": (
                "a" * 64
                if start == 0
                else "b" * 64
            ),
            "privacy_status": "safe",
            "rights_status": "synthetic_evaluation",
            "contains_raw_identifiers": False,
            "documents": documents,
            "lineage": {"purpose": "unit-test"},
        }
    )


def _hard_negative_catalog() -> EmbeddingBenchmarkDocumentCatalog:
    documents = []
    for group in range(25):
        for variant, location in enumerate(
            ("location-a", "location-b")
        ):
            documents.append(
                {
                    "document_id": (
                        f"pair-{group:02d}-{variant}"
                    ),
                    "text": (
                        f"Safe aggregate pair {group} {variant}"
                    ),
                    "location": location,
                    "category": f"category-{group:02d}",
                    "daypart": "evening",
                }
            )
    documents.extend(
        _catalog(start=50, count=50).to_dict()["documents"]
    )
    return EmbeddingBenchmarkDocumentCatalog.from_mapping(
        {
            "catalog_id": "hard-negative-ready",
            "catalog_version": "v1",
            "source_type": "curated_synthetic_features",
            "source_fingerprint": "c" * 64,
            "privacy_status": "safe",
            "rights_status": "synthetic_evaluation",
            "contains_raw_identifiers": False,
            "documents": documents,
            "lineage": {"purpose": "unit-test"},
        }
    )


def test_plan_reports_exact_document_deficit_without_generating_data():
    plan = ProductionEmbeddingBenchmarkCurationService().plan(
        document_catalogs=[_catalog(count=86)]
    )

    assert plan["status"] == "blocked_curation_coverage"
    assert plan["document_coverage"]["document_count"] == 86
    assert plan["document_deficit"] == 14
    assert len(plan["supplemental_document_slots"]) == 14
    assert plan["documents_auto_generated"] is False
    assert plan["gold_labels_auto_approved"] is False
    assert plan["synthetic_audience_volume_claimed"] is False
    assert len(plan["plan_fingerprint"]) == 64


def test_plan_emits_only_grounded_single_constraint_negatives():
    plan = ProductionEmbeddingBenchmarkCurationService().plan(
        document_catalogs=[_hard_negative_catalog()]
    )

    assert (
        plan["grounded_hard_negative_candidate_count"]
        >= 20
    )
    assert plan["grounded_hard_negative_result"]["passed"] is True
    for candidate in plan["grounded_hard_negative_candidates"]:
        assert candidate["violated_constraint"] in {
            "location",
            "category",
            "daypart",
        }
        assert len(candidate["shared_constraints"]) == 2
        assert candidate["requires_human_review"] is True


def test_plan_is_content_addressed_and_never_self_approves():
    service = ProductionEmbeddingBenchmarkCurationService()
    first = service.plan(document_catalogs=[_hard_negative_catalog()])
    second = service.plan(document_catalogs=[_hard_negative_catalog()])

    assert first == second
    assert first["requires_human_gold_label_review"] is True
    assert first["queries_auto_generated"] is False
    assert first["activation_or_export_performed"] is False


def test_plan_rejects_duplicate_ids_across_catalogs():
    second = copy.deepcopy(_catalog()).to_dict()
    second["catalog_id"] = "second"
    second["source_fingerprint"] = "d" * 64

    with pytest.raises(ValueError, match="unique across catalogs"):
        ProductionEmbeddingBenchmarkCurationService().plan(
            document_catalogs=[
                _catalog(),
                EmbeddingBenchmarkDocumentCatalog.from_mapping(second),
            ]
        )


def test_plan_requires_at_least_one_safe_catalog():
    with pytest.raises(ValueError, match="At least one"):
        ProductionEmbeddingBenchmarkCurationService().plan(
            document_catalogs=[]
        )
