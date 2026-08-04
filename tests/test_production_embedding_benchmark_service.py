from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDataset,
)
from app.services.production_embedding_benchmark_service import (
    ProductionEmbeddingBenchmarkReportValidator,
    ProductionEmbeddingBenchmarkService,
)
from tests.test_production_feature_embedding_service import _model


class ControlledEncoder:
    def encode_documents(self, texts, *, model, batch_size):
        vectors = np.zeros((len(texts), model.dimension), dtype=float)
        for index, text in enumerate(texts):
            if text.startswith("relevant"):
                vectors[index, 0] = 1.0
            else:
                vectors[index, 0] = -1.0
        return vectors

    def encode_queries(self, texts, *, model, batch_size):
        vectors = np.zeros((len(texts), model.dimension), dtype=float)
        for index, text in enumerate(texts):
            if text.startswith("unsupported"):
                vectors[index, 1] = 1.0
            else:
                vectors[index, 0] = 1.0
        return vectors


def production_benchmark_dataset() -> EmbeddingBenchmarkDataset:
    documents = []
    dayparts = ("morning", "afternoon", "evening", "night")
    for index in range(100):
        relevant = index < 10
        documents.append(
            {
                "document_id": f"doc-{index:03d}",
                "text": (
                    f"relevant document {index}"
                    if relevant
                    else f"distractor document {index}"
                ),
                "location": (
                    "target_location"
                    if relevant
                    else f"other_location_{index % 19:02d}"
                ),
                "category": (
                    "target_category"
                    if relevant
                    else f"other_category_{index % 19:02d}"
                ),
                "daypart": (
                    "target_daypart"
                    if relevant
                    else dayparts[index % len(dayparts)]
                ),
            }
        )

    languages = ("en", "fr", "hi", "es", "de")
    cases = []
    for index in range(80):
        cases.append(
            {
                "case_id": f"supported-{index:03d}",
                "query": f"supported query {index}",
                "language": languages[index % len(languages)],
                "relevant_document_ids": [
                    f"doc-{value:03d}" for value in range(10)
                ],
                "hard_negative_document_ids": (
                    ["doc-090"] if index < 20 else []
                ),
                "expected_locations": ["target_location"],
                "expected_categories": ["target_category"],
                "expected_dayparts": ["target_daypart"],
                "semantic_group_id": (
                    f"group-{index // 2:03d}"
                    if index < 20
                    else None
                ),
            }
        )
    for index in range(20):
        cases.append(
            {
                "case_id": f"unsupported-{index:03d}",
                "query": f"unsupported query {index}",
                "language": languages[index % len(languages)],
                "relevant_document_ids": [],
                "expected_locations": [
                    f"unsupported_location_{index:03d}"
                ],
                "unsupported_location": True,
            }
        )
    return EmbeddingBenchmarkDataset.from_mapping(
        {
            "benchmark_id": "punk-global-retrieval-eval",
            "dataset_version": "immutable-dataset-v1",
            "privacy_status": "safe",
            "rights_status": "synthetic_evaluation",
            "contains_raw_identifiers": False,
            "authoring_manifest": {
                "schema_version": (
                    "punk-embedding-benchmark-authoring-v1"
                ),
                "builder_version": "production-dataset-builder-v1",
                "review_status": "approved",
                "reviewed_by": "benchmark-reviewer",
                "reviewed_at": "2026-07-30T12:00:00+00:00",
                "case_catalog_fingerprint": "a" * 64,
                "document_catalog_fingerprints": ["b" * 64],
            },
            "documents": documents,
            "cases": cases,
        }
    )


def single_relevant_benchmark_dataset() -> EmbeddingBenchmarkDataset:
    payload = production_benchmark_dataset().to_dict()
    for case in payload["cases"]:
        if not case["unsupported_location"]:
            case["relevant_document_ids"] = ["doc-000"]
    return EmbeddingBenchmarkDataset.from_mapping(payload)


def production_benchmark_report():
    ticks = iter(
        value / 1000.0
        for value in range(1000)
    )
    return ProductionEmbeddingBenchmarkService(
        encoder=ControlledEncoder(),
        clock=lambda: next(ticks),
        memory_mb_fn=lambda: 128.0,
        now_fn=lambda: datetime(
            2026,
            7,
            30,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    ).evaluate(
        dataset=production_benchmark_dataset(),
        model=_model(),
        top_k=10,
        rejection_similarity_threshold=0.5,
        cost_per_1000_queries_usd=0.0,
    )


def test_benchmark_calculates_complete_production_evidence():
    report = production_benchmark_report()

    assert report["passed"] is True
    assert report["evaluation"]["case_count"] == 100
    assert report["evaluation"]["document_count"] == 100
    assert report["evaluation"]["language_count"] == 5
    assert report["evaluation"]["location_value_count"] == 20
    assert report["evaluation"]["category_value_count"] == 20
    assert report["evaluation"]["daypart_value_count"] == 5
    assert report["evaluation"]["minimum_cases_per_language"] == 20
    assert (
        report["evaluation"]["constraint_intersection_case_count"]
        == 80
    )
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["precision_at_k"] == 1.0
    assert report["metrics"]["ndcg_at_k"] == 1.0
    assert report["metrics"]["hard_negative_rejection_rate"] == 1.0
    assert (
        report["metrics"]["unsupported_location_false_match_rate"]
        == 0.0
    )
    assert report["metrics"]["multilingual_consistency"] == 1.0
    assert report["activation_or_export_performed"] is False
    assert report["raw_identifiers_read"] is False


def test_benchmark_rejects_an_impossible_precision_top_k_before_inference():
    with pytest.raises(
        ValueError,
        match="theoretical maximum",
    ):
        ProductionEmbeddingBenchmarkService(
            encoder=ControlledEncoder(),
            memory_mb_fn=lambda: 128.0,
        ).evaluate(
            dataset=single_relevant_benchmark_dataset(),
            model=_model(),
            top_k=10,
            rejection_similarity_threshold=0.5,
            cost_per_1000_queries_usd=0.0,
        )


def test_single_relevant_labels_can_run_as_an_explicit_top_one_benchmark():
    dataset = single_relevant_benchmark_dataset()
    report = ProductionEmbeddingBenchmarkService(
        encoder=ControlledEncoder(),
        memory_mb_fn=lambda: 128.0,
    ).evaluate(
        dataset=dataset,
        model=_model(),
        top_k=1,
        rejection_similarity_threshold=0.5,
        cost_per_1000_queries_usd=0.0,
    )

    assert report["passed"] is True
    assert report["evaluation"]["top_k"] == 1
    assert (
        report["evaluation"][
            "theoretical_maximum_precision_at_k"
        ]
        == 1.0
    )
    assert report["metrics"]["precision_at_k"] == 1.0
    assert ProductionEmbeddingBenchmarkReportValidator().validate(
        report,
        model=_model(),
        dataset=dataset,
    )["passed"] is True


def test_report_validator_rejects_tampered_theoretical_precision_evidence():
    dataset = single_relevant_benchmark_dataset()
    report = ProductionEmbeddingBenchmarkService(
        encoder=ControlledEncoder(),
        memory_mb_fn=lambda: 128.0,
    ).evaluate(
        dataset=dataset,
        model=_model(),
        top_k=1,
        rejection_similarity_threshold=0.5,
        cost_per_1000_queries_usd=0.0,
    )
    report["evaluation"][
        "theoretical_maximum_precision_at_k"
    ] = 0.5

    with pytest.raises(ValueError, match="theoretical precision"):
        ProductionEmbeddingBenchmarkReportValidator().validate(
            report,
            model=_model(),
            dataset=dataset,
        )


def test_report_validator_detects_metric_threshold_and_fingerprint_tampering():
    report = production_benchmark_report()
    validator = ProductionEmbeddingBenchmarkReportValidator()
    assert validator.validate(
        report,
        model=_model(),
        dataset=production_benchmark_dataset(),
    )["passed"] is True

    report["metrics"]["recall_at_k"] = 0.0
    with pytest.raises(ValueError, match="threshold evidence"):
        validator.validate(
            report,
            model=_model(),
            dataset=production_benchmark_dataset(),
        )

    report = production_benchmark_report()
    report["report_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        validator.validate(
            report,
            model=_model(),
            dataset=production_benchmark_dataset(),
        )


def test_report_validator_rejects_a_different_model_spec():
    report = production_benchmark_report()

    with pytest.raises(ValueError, match="requested model"):
        ProductionEmbeddingBenchmarkReportValidator().validate(
            report,
            model=_model(query_prefix="search: "),
            dataset=production_benchmark_dataset(),
        )


def test_report_validator_rejects_a_different_dataset_artifact():
    report = production_benchmark_report()
    payload = production_benchmark_dataset().to_dict()
    payload["dataset_version"] = "different-immutable-version"

    with pytest.raises(ValueError, match="supplied dataset"):
        ProductionEmbeddingBenchmarkReportValidator().validate(
            report,
            model=_model(),
            dataset=EmbeddingBenchmarkDataset.from_mapping(payload),
        )


def test_small_benchmark_cannot_pass_production_coverage():
    payload = production_benchmark_dataset().to_dict()
    payload["documents"] = payload["documents"][:10]
    known = {
        value["document_id"] for value in payload["documents"]
    }
    payload["cases"] = [
        {
            **value,
            "hard_negative_document_ids": [
                item
                for item in value["hard_negative_document_ids"]
                if item in known
            ],
        }
        for value in payload["cases"][:5]
    ]
    dataset = EmbeddingBenchmarkDataset.from_mapping(payload)
    report = ProductionEmbeddingBenchmarkService(
        encoder=ControlledEncoder(),
        memory_mb_fn=lambda: 128.0,
    ).evaluate(
        dataset=dataset,
        model=_model(),
        top_k=10,
        rejection_similarity_threshold=0.5,
        cost_per_1000_queries_usd=0.0,
    )

    assert report["passed"] is False
    assert report["threshold_results"]["case_count"]["passed"] is False
    assert (
        report["threshold_results"]["document_count"]["passed"]
        is False
    )


def test_bad_embedding_shape_and_zero_vectors_fail_closed():
    class BadEncoder(ControlledEncoder):
        def encode_documents(self, texts, *, model, batch_size):
            return np.ones((len(texts), 3), dtype=float)

    with pytest.raises(ValueError, match="shape"):
        ProductionEmbeddingBenchmarkService(
            encoder=BadEncoder()
        ).evaluate(
            dataset=production_benchmark_dataset(),
            model=_model(),
            top_k=10,
            cost_per_1000_queries_usd=0.0,
        )

    class ZeroQueryEncoder(ControlledEncoder):
        def encode_queries(self, texts, *, model, batch_size):
            return np.zeros((len(texts), model.dimension), dtype=float)

    with pytest.raises(ValueError, match="zero vectors"):
        ProductionEmbeddingBenchmarkService(
            encoder=ZeroQueryEncoder()
        ).evaluate(
            dataset=production_benchmark_dataset(),
            model=_model(),
            top_k=10,
            cost_per_1000_queries_usd=0.0,
        )


def test_policy_thresholds_cannot_be_weakened_by_cli_overrides():
    with pytest.raises(ValueError, match="weaker"):
        ProductionEmbeddingBenchmarkService(
            encoder=ControlledEncoder()
        ).evaluate(
            dataset=production_benchmark_dataset(),
            model=_model(),
            top_k=10,
            cost_per_1000_queries_usd=0.0,
            threshold_overrides={"recall_at_k": 0.1},
        )
