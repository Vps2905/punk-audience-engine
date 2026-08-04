from __future__ import annotations

import numpy as np
import pytest

from app.models.embedding_benchmark_contracts import EmbeddingBenchmarkDataset

from app.models.production_governed_retrieval_benchmark_contracts import (
    GovernedRetrievalBenchmarkPolicy,
)
from app.services.production_governed_retrieval_benchmark_service import (
    OfflinePinnedModelApprovalGate,
    ProductionGovernedRetrievalBenchmarkService,
    validate_governed_retrieval_benchmark_report,
)
from app.services.production_multilingual_constraint_canonicalization_service import (
    SemanticConstraintScore,
)


class FakeDocumentEncoder:
    def encode(self, *, texts, model, batch_size):
        del batch_size
        rows = []
        for text in texts:
            vector = np.zeros(model.dimension, dtype=np.float64)
            if "car wash" in text:
                vector[0] = 1.0
            elif "restaurant" in text:
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            rows.append(vector)
        return np.asarray(rows)


class FakeQueryEncoder:
    def encode(self, *, query_text, model):
        vector = np.zeros(model.dimension, dtype=np.float64)
        if "car wash" in query_text.lower() or "কার ওয়াশ" in query_text:
            vector[0] = 1.0
        elif "restaurant" in query_text.lower():
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector.tolist()


class FakeSemanticResolver:
    def score(self, *, query_text, dimension, entries, model):
        del model
        values = {entry.canonical_value for entry in entries}
        if dimension == "category":
            winner = "car_wash" if (
                "car wash" in query_text.lower() or "কার ওয়াশ" in query_text
            ) else "restaurant"
        else:
            winner = "afternoon" if (
                "afternoon" in query_text.lower() or "দুপুরে" in query_text
            ) else "evening"
        return [
            SemanticConstraintScore(
                canonical_value=value,
                score=0.95 if value == winner else 0.10,
            )
            for value in sorted(values)
        ] + [SemanticConstraintScore("_unspecified", 0.05)]


def _dataset():
    return {
        "benchmark_id": "module2-1-test",
        "dataset_version": "immutable-v1",
        "privacy_status": "safe",
        "rights_status": "offline_evaluation_only",
        "contains_raw_identifiers": False,
        "authoring_manifest": {
            "schema_version": "punk-embedding-benchmark-authoring-v1",
            "builder_version": "production-dataset-builder-v1",
            "review_status": "approved",
            "reviewed_by": "benchmark-test-owner",
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
            {
                "case_id": "unsupported",
                "query": "Find audiences in Ottawa.",
                "language": "en",
                "expected_locations": ["ottawa"],
                "expected_categories": [],
                "expected_dayparts": [],
                "relevant_document_ids": [],
                "unsupported_location": True,
            },
        ],
    }


def _service(policy=None):
    models = (
        ProductionGovernedRetrievalBenchmarkService.PRIMARY_MODEL,
        ProductionGovernedRetrievalBenchmarkService.COMPLEMENTARY_MODEL,
    )
    return ProductionGovernedRetrievalBenchmarkService(
        policy=policy,
        document_encoder=FakeDocumentEncoder(),
        query_encoder=FakeQueryEncoder(),
        semantic_resolver=FakeSemanticResolver(),
        model_gate=OfflinePinnedModelApprovalGate(models),
    )


def test_real_canonicalizer_and_retrieval_pipeline_pass_offline_engineering_gate():
    report = _service().evaluate(_dataset())

    assert report["engineering_passed"] is True
    assert report["summary"]["full_canonicalization_accuracy"] == 1.0
    assert report["summary"]["final_candidate_recall"] == 1.0
    assert report["summary"]["structured_semantic_top1_accuracy"] == 1.0
    assert report["summary"]["unsupported_rejection_rate"] == 1.0
    assert report["registration_allowed"] is False
    assert report["production_routing_enabled"] is False
    assert report["raw_query_text_stored"] is False
    assert report["document_text_stored"] is False
    assert report["embeddings_stored"] is False


def test_benchmark_uses_query_only_and_never_oracle_request_constraints():
    report = _service().evaluate(_dataset())
    supported = next(
        row for row in report["case_diagnostics"] if row["case_id"] == "supported-bn"
    )

    assert supported["observed"]["locations"] == ["montreal"]
    assert supported["observed"]["categories"] == ["car_wash"]
    assert supported["observed"]["dayparts"] == ["afternoon"]
    assert "query" not in supported
    assert "document_text" not in supported


def test_unsupported_location_is_rejected_before_candidate_search():
    report = _service().evaluate(_dataset())
    unsupported = next(
        row for row in report["case_diagnostics"] if row["case_id"] == "unsupported"
    )

    assert unsupported["observed"]["retrieval_status"] == "blocked"
    assert unsupported["metrics"]["safe_unsupported_rejection"] is True
    assert unsupported["observed"]["selected_document_ids"] == []


def test_production_certification_stays_blocked_by_sample_size_and_human_review():
    report = _service().evaluate(_dataset())

    certification = report["production_certification"]
    assert report["benchmark_evidence_ready"] is False
    assert certification["ready"] is False
    assert certification["minimum_zero_failure_unsupported_cases"] == 299
    assert "native_human_signoff_pending" in certification["reason_codes"]
    assert "unsupported_calibration_sample_insufficient" in certification[
        "reason_codes"
    ]
    assert "model_registration_and_external_release_gates_pending" in (
        certification["reason_codes"]
    )


def test_policy_failure_is_reported_without_enabling_routing():
    strict = GovernedRetrievalBenchmarkPolicy(max_p95_latency_ms=0.000001, max_p99_latency_ms=0.000002)
    report = _service(policy=strict).evaluate(_dataset())

    assert report["engineering_passed"] is False
    assert report["checks"]["p95_latency"] is False
    assert report["production_routing_enabled"] is False
    assert report["eligible_for_automatic_proposal"] is False


def test_dataset_accepts_real_unsupported_case_shape_without_category_or_daypart():
    dataset = _dataset()
    unsupported = next(
        case for case in dataset["cases"] if case["unsupported_location"]
    )

    assert unsupported["expected_categories"] == []
    assert unsupported["expected_dayparts"] == []

    report = _service().evaluate(dataset)
    diagnostic = next(
        row
        for row in report["case_diagnostics"]
        if row["case_id"] == unsupported["case_id"]
    )

    assert diagnostic["metrics"]["safe_unsupported_rejection"] is True


def test_supported_case_still_requires_expected_category():
    dataset = _dataset()
    dataset["cases"][0]["expected_categories"] = []

    with pytest.raises(
        ValueError,
        match="Supported cases require expected locations, categories",
    ):
        _service().evaluate(dataset)


def test_dataset_identity_uses_canonical_dataset_fingerprint():
    dataset = _dataset()
    expected = EmbeddingBenchmarkDataset.from_mapping(dataset).fingerprint

    report = _service().evaluate(dataset)

    assert report["benchmark"]["dataset_fingerprint"] == expected


def test_dataset_rejects_mismatched_declared_fingerprint():
    dataset = _dataset()
    dataset["dataset_fingerprint"] = "0" * 64

    with pytest.raises(
        ValueError,
        match="Declared dataset_fingerprint does not match",
    ):
        _service().evaluate(dataset)


def test_dataset_rejects_raw_identifier_fields():
    dataset = _dataset()
    dataset["documents"][0]["device_id"] = "forbidden"

    with pytest.raises(ValueError, match="Raw identifier"):
        _service().evaluate(dataset)


def test_dataset_rejects_unknown_relevant_document_ids():
    dataset = _dataset()
    dataset["cases"][0]["relevant_document_ids"] = ["missing"]

    with pytest.raises(ValueError, match="unknown documents"):
        _service().evaluate(dataset)


def test_report_validator_rejects_tampering_and_unsafe_flags():
    report = _service().evaluate(_dataset())
    validate_governed_retrieval_benchmark_report(report)

    tampered = dict(report)
    tampered["production_routing_enabled"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch|Unsafe"):
        validate_governed_retrieval_benchmark_report(tampered)
