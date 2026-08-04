from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pytest

from app.services.production_embedding_benchmark_diagnostics_service import (
    ProductionEmbeddingBenchmarkDiagnosticsService,
    ProductionEmbeddingBenchmarkDiagnosticsValidator,
)
from tests.test_production_embedding_benchmark_service import (
    ControlledEncoder,
    production_benchmark_dataset,
    single_relevant_benchmark_dataset,
)
from tests.test_production_feature_embedding_service import _model


def diagnostics_report():
    return ProductionEmbeddingBenchmarkDiagnosticsService(
        encoder=ControlledEncoder(),
        memory_mb_fn=lambda: 64.0,
        now_fn=lambda: datetime(
            2026,
            8,
            4,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    ).evaluate(
        dataset=production_benchmark_dataset(),
        model=_model(),
        top_k=1,
        diagnostic_rank_depth=5,
        batch_size=16,
        rejection_similarity_threshold=0.5,
        target_unsupported_false_match_rate=0.01,
    )


def test_diagnostics_persist_privacy_safe_case_level_evidence():
    report = diagnostics_report()

    assert report["registration_allowed"] is False
    assert report["recommended_threshold_auto_applied"] is False
    assert report["activation_or_export_performed"] is False
    assert report["raw_identifiers_read"] is False
    assert len(report["case_diagnostics"]) == 100
    assert report["aggregate"]["exact_top1_accuracy"] == 1.0
    assert (
        report["aggregate"]["semantic_signature_top1_accuracy"]
        == 1.0
    )
    assert report["aggregate"]["hard_negative_exclusion_at_k"] == 1.0
    assert report["per_language"]["en"]["case_count"] == 20
    first = report["case_diagnostics"][0]
    assert len(first["top_candidates"]) == 5
    assert "query" not in first
    assert "text" not in first["top_candidates"][0]
    assert "embedding" not in first


def test_diagnostics_distinguish_semantic_signature_ambiguity():
    payload = single_relevant_benchmark_dataset().to_dict()
    payload["documents"][1]["location"] = payload["documents"][0][
        "location"
    ]
    payload["documents"][1]["category"] = payload["documents"][0][
        "category"
    ]
    payload["documents"][1]["daypart"] = payload["documents"][0][
        "daypart"
    ]
    from app.models.embedding_benchmark_contracts import (
        EmbeddingBenchmarkDataset,
    )

    dataset = EmbeddingBenchmarkDataset.from_mapping(payload)
    report = ProductionEmbeddingBenchmarkDiagnosticsService(
        encoder=ControlledEncoder(),
        memory_mb_fn=lambda: 64.0,
    ).evaluate(
        dataset=dataset,
        model=_model(),
        top_k=1,
        diagnostic_rank_depth=3,
        rejection_similarity_threshold=0.5,
    )

    assert report["ambiguity"][
        "duplicate_semantic_signature_group_count"
    ] >= 1
    assert report["ambiguity"]["ambiguous_supported_case_count"] == 80
    assert all(
        value["ambiguous_exact_label"]
        for value in report["case_diagnostics"][:80]
    )


def test_calibration_is_review_only_and_blocks_small_unsupported_sample():
    report = diagnostics_report()
    calibration = report["calibration"]

    assert calibration["status"] == "candidate_requires_human_review"
    assert calibration["recommended_candidate"] is not None
    assert calibration["recommended_threshold_auto_applied"] is False
    assert calibration["unsupported_calibration_case_count"] == 20
    assert calibration[
        "minimum_observable_nonzero_false_match_rate"
    ] == 0.05
    assert calibration[
        "minimum_zero_failure_sample_count_for_confidence_target"
    ] == 299
    assert calibration["production_calibration_ready"] is False
    assert (
        "insufficient_unsupported_calibration_cases"
        in calibration["production_calibration_blockers"]
    )


def test_calibration_surfaces_model_score_scale_failure():
    class HighUnsupportedEncoder(ControlledEncoder):
        def encode_documents(self, texts, *, model, batch_size):
            vectors = np.zeros((len(texts), model.dimension), dtype=float)
            vectors[:, 0] = 1.0
            vectors[:, 1] = np.linspace(0.01, 0.2, len(texts))
            return vectors

        def encode_queries(self, texts, *, model, batch_size):
            vectors = np.zeros((len(texts), model.dimension), dtype=float)
            vectors[:, 0] = 1.0
            return vectors

    report = ProductionEmbeddingBenchmarkDiagnosticsService(
        encoder=HighUnsupportedEncoder(),
        memory_mb_fn=lambda: 64.0,
    ).evaluate(
        dataset=single_relevant_benchmark_dataset(),
        model=_model(),
        top_k=1,
        diagnostic_rank_depth=3,
        rejection_similarity_threshold=0.78,
    )

    assert report["aggregate"][
        "current_threshold_unsupported_false_match_rate"
    ] == 1.0
    assert report["calibration"]["recommended_candidate"] is not None
    assert report["calibration"][
        "recommended_threshold_auto_applied"
    ] is False


def test_diagnostics_validator_detects_tampering_and_forbidden_content():
    report = diagnostics_report()
    validator = ProductionEmbeddingBenchmarkDiagnosticsValidator()
    dataset = production_benchmark_dataset()

    assert validator.validate(
        report,
        model=_model(),
        dataset=dataset,
    )["registration_allowed"] is False

    tampered = deepcopy(report)
    tampered["case_diagnostics"][0]["query"] = "do not persist"
    with pytest.raises(ValueError, match="forbidden raw content"):
        validator.validate(
            tampered,
            model=_model(),
            dataset=dataset,
        )

    tampered = deepcopy(report)
    tampered["diagnostics_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        validator.validate(
            tampered,
            model=_model(),
            dataset=dataset,
        )
