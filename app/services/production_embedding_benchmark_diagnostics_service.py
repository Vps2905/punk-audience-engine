from __future__ import annotations

import json
import math
import os
import resource
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.models.audience_feature_contracts import stable_digest
from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkCase,
    EmbeddingBenchmarkDataset,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.services.production_embedding_benchmark_service import (
    EmbeddingBenchmarkEncoder,
    SentenceTransformerEmbeddingBenchmarkEncoder,
)

BENCHMARK_DIAGNOSTICS_SCHEMA_VERSION = (
    "punk-embedding-benchmark-diagnostics-v1"
)


class ProductionEmbeddingBenchmarkDiagnosticsService:
    """Privacy-safe diagnostics for one immutable embedding model revision.

    This service is intentionally separate from the production pass/fail report.
    It exposes enough case-level ranking evidence to diagnose retrieval failures,
    semantic-label ambiguity, score separation, and threshold calibration without
    exposing query text, document text, embeddings, identifiers, credentials, or
    activation payloads.
    """

    def __init__(
        self,
        *,
        encoder: EmbeddingBenchmarkEncoder | None = None,
        now_fn: Callable[[], datetime] | None = None,
        memory_mb_fn: Callable[[], float] | None = None,
    ) -> None:
        self._encoder = (
            encoder or SentenceTransformerEmbeddingBenchmarkEncoder()
        )
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._memory_mb_fn = memory_mb_fn or self._peak_memory_mb

    def evaluate(
        self,
        *,
        dataset: EmbeddingBenchmarkDataset,
        model: EmbeddingModelSpec,
        top_k: int = 1,
        diagnostic_rank_depth: int = 10,
        batch_size: int = 64,
        rejection_similarity_threshold: float = 0.78,
        target_unsupported_false_match_rate: float = 0.01,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        k = int(top_k)
        depth = int(diagnostic_rank_depth)
        batch = int(batch_size)
        document_count = len(dataset.documents)

        if not 1 <= k <= document_count:
            raise ValueError(
                "top_k must be between 1 and the benchmark document count."
            )
        if not k <= depth <= document_count:
            raise ValueError(
                "diagnostic_rank_depth must be between top_k and the "
                "benchmark document count."
            )
        if not 1 <= batch <= 4096:
            raise ValueError("batch_size must be between 1 and 4096.")

        current_threshold = self._bounded_rate_or_score(
            rejection_similarity_threshold,
            "rejection_similarity_threshold",
            minimum=-1.0,
            maximum=1.0,
        )
        target_false_match_rate = self._bounded_rate_or_score(
            target_unsupported_false_match_rate,
            "target_unsupported_false_match_rate",
            minimum=0.0,
            maximum=1.0,
        )
        confidence = self._bounded_rate_or_score(
            confidence_level,
            "confidence_level",
            minimum=0.5,
            maximum=0.999999,
        )

        documents = list(dataset.documents)
        document_ids = [value.document_id for value in documents]
        document_by_id = {value.document_id: value for value in documents}
        document_index = {
            document_id: index
            for index, document_id in enumerate(document_ids)
        }

        document_vectors = np.asarray(
            self._encoder.encode_documents(
                [value.text for value in documents],
                model=model,
                batch_size=batch,
            ),
            dtype=np.float64,
        )
        self._validate_matrix(
            document_vectors,
            rows=document_count,
            dimension=model.dimension,
            label="document",
        )
        document_vectors = self._unit_rows(document_vectors)

        semantic_signature_ids = {
            case.case_id: self._semantic_signature_document_ids(
                case,
                documents=documents,
            )
            for case in dataset.cases
        }

        case_diagnostics: list[dict[str, Any]] = []
        for case in dataset.cases:
            query_vector = np.asarray(
                self._encoder.encode_queries(
                    [case.query],
                    model=model,
                    batch_size=1,
                ),
                dtype=np.float64,
            )
            self._validate_matrix(
                query_vector,
                rows=1,
                dimension=model.dimension,
                label="query",
            )
            query_vector = self._unit_rows(query_vector)[0]
            scores = document_vectors @ query_vector
            ranked_positions = np.argsort(-scores, kind="stable")
            ranked_ids = [
                document_ids[int(position)]
                for position in ranked_positions
            ]
            case_diagnostics.append(
                self._case_diagnostic(
                    case=case,
                    ranked_ids=ranked_ids,
                    ranked_positions=ranked_positions,
                    scores=scores,
                    document_by_id=document_by_id,
                    document_index=document_index,
                    semantic_relevant_ids=semantic_signature_ids[
                        case.case_id
                    ],
                    top_k=k,
                    diagnostic_rank_depth=depth,
                    rejection_similarity_threshold=current_threshold,
                )
            )

        ambiguity = self._ambiguity_summary(
            dataset=dataset,
            semantic_signature_ids=semantic_signature_ids,
        )
        aggregate = self._aggregate_diagnostics(
            case_diagnostics,
            top_k=k,
        )
        per_language = self._per_language_diagnostics(
            case_diagnostics,
            top_k=k,
        )
        score_distributions = self._score_distributions(case_diagnostics)
        calibration = self._calibration_evidence(
            case_diagnostics,
            current_threshold=current_threshold,
            target_false_match_rate=target_false_match_rate,
            confidence_level=confidence,
        )

        report: dict[str, Any] = {
            "schema_version": BENCHMARK_DIAGNOSTICS_SCHEMA_VERSION,
            "benchmark_id": dataset.benchmark_id,
            "dataset_version": dataset.dataset_version,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_safety": {
                "privacy_status": dataset.privacy_status,
                "rights_status": dataset.rights_status,
                "contains_raw_identifiers": False,
            },
            "model": model.to_safe_dict(),
            "evaluation": {
                "top_k": k,
                "diagnostic_rank_depth": depth,
                "batch_size": batch,
                "case_count": len(dataset.cases),
                "document_count": document_count,
                "rejection_similarity_threshold": current_threshold,
                "target_unsupported_false_match_rate": (
                    target_false_match_rate
                ),
                "confidence_level": confidence,
            },
            "ambiguity": ambiguity,
            "aggregate": aggregate,
            "per_language": per_language,
            "score_distributions": score_distributions,
            "calibration": calibration,
            "case_diagnostics": case_diagnostics,
            "peak_memory_mb": max(0.0, float(self._memory_mb_fn())),
            "generated_at": self._now_fn().astimezone(
                timezone.utc
            ).isoformat(),
            "registration_allowed": False,
            "recommended_threshold_auto_applied": False,
            "activation_or_export_performed": False,
            "credentials_exposed": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
        }
        report["diagnostics_fingerprint"] = stable_digest(report)
        return report

    def _case_diagnostic(
        self,
        *,
        case: EmbeddingBenchmarkCase,
        ranked_ids: Sequence[str],
        ranked_positions: np.ndarray,
        scores: np.ndarray,
        document_by_id: Mapping[str, Any],
        document_index: Mapping[str, int],
        semantic_relevant_ids: Sequence[str],
        top_k: int,
        diagnostic_rank_depth: int,
        rejection_similarity_threshold: float,
    ) -> dict[str, Any]:
        exact_relevant = set(case.relevant_document_ids)
        semantic_relevant = set(semantic_relevant_ids)
        hard_negatives = set(case.hard_negative_document_ids)
        top_ids = list(ranked_ids[:top_k])
        top_document = document_by_id[top_ids[0]]
        top_score = float(scores[int(ranked_positions[0])])
        second_score = (
            float(scores[int(ranked_positions[1])])
            if len(ranked_positions) > 1
            else None
        )

        ranks = {
            document_id: index + 1
            for index, document_id in enumerate(ranked_ids)
        }

        def best_score(ids: set[str]) -> float | None:
            if not ids:
                return None
            return max(
                float(scores[document_index[document_id]])
                for document_id in ids
            )

        def best_rank(ids: set[str]) -> int | None:
            if not ids:
                return None
            return min(ranks[document_id] for document_id in ids)

        best_exact_score = best_score(exact_relevant)
        best_semantic_score = best_score(semantic_relevant)
        hard_negative_score = best_score(hard_negatives)
        hard_negative_rank = best_rank(hard_negatives)

        candidates = []
        for position in ranked_positions[:diagnostic_rank_depth]:
            index = int(position)
            document_id = ranked_ids[len(candidates)]
            document = document_by_id[document_id]
            candidates.append(
                {
                    "rank": len(candidates) + 1,
                    "document_id": document_id,
                    "score": float(scores[index]),
                    "location": document.location,
                    "category": document.category,
                    "daypart": document.daypart,
                    "exact_relevant": document_id in exact_relevant,
                    "semantic_signature_relevant": (
                        document_id in semantic_relevant
                    ),
                    "named_hard_negative": document_id in hard_negatives,
                }
            )

        accepted = top_score >= rejection_similarity_threshold
        exact_hit = bool(exact_relevant.intersection(top_ids))
        semantic_hit = bool(semantic_relevant.intersection(top_ids))

        return {
            "case_id": case.case_id,
            "language": case.language,
            "semantic_group_id": case.semantic_group_id,
            "unsupported_location": case.unsupported_location,
            "expected_locations": list(case.expected_locations),
            "expected_categories": list(case.expected_categories),
            "expected_dayparts": list(case.expected_dayparts),
            "exact_relevant_document_ids": sorted(exact_relevant),
            "semantic_signature_relevant_document_ids": sorted(
                semantic_relevant
            ),
            "ambiguous_exact_label": (
                not case.unsupported_location
                and semantic_relevant != exact_relevant
                and bool(semantic_relevant)
            ),
            "top_ids": top_ids,
            "top_score": top_score,
            "second_score": second_score,
            "top1_score_margin": (
                top_score - second_score
                if second_score is not None
                else None
            ),
            "exact_hit_at_k": exact_hit,
            "semantic_signature_hit_at_k": semantic_hit,
            "best_exact_relevant_rank": best_rank(exact_relevant),
            "best_exact_relevant_score": best_exact_score,
            "best_semantic_relevant_rank": best_rank(semantic_relevant),
            "best_semantic_relevant_score": best_semantic_score,
            "top_document_location_match": (
                top_document.location in case.expected_locations
                if case.expected_locations
                else None
            ),
            "top_document_category_match": (
                top_document.category in case.expected_categories
                if case.expected_categories
                else None
            ),
            "top_document_daypart_match": (
                top_document.daypart in case.expected_dayparts
                if case.expected_dayparts
                else None
            ),
            "hard_negative_document_ids": sorted(hard_negatives),
            "hard_negative_excluded_at_k": (
                hard_negatives.isdisjoint(top_ids)
                if hard_negatives
                else None
            ),
            "best_hard_negative_rank": hard_negative_rank,
            "best_hard_negative_score": hard_negative_score,
            "exact_relevant_to_hard_negative_margin": (
                best_exact_score - hard_negative_score
                if (
                    best_exact_score is not None
                    and hard_negative_score is not None
                )
                else None
            ),
            "accepted_at_current_threshold": accepted,
            "unsupported_false_match_at_current_threshold": (
                case.unsupported_location and accepted
            ),
            "top_candidates": candidates,
        }

    def _semantic_signature_document_ids(
        self,
        case: EmbeddingBenchmarkCase,
        *,
        documents: Sequence[Any],
    ) -> tuple[str, ...]:
        if case.unsupported_location:
            return ()
        has_constraints = bool(
            case.expected_locations
            or case.expected_categories
            or case.expected_dayparts
        )
        if not has_constraints:
            return tuple(case.relevant_document_ids)
        return tuple(
            document.document_id
            for document in documents
            if (
                (
                    not case.expected_locations
                    or document.location in case.expected_locations
                )
                and (
                    not case.expected_categories
                    or document.category in case.expected_categories
                )
                and (
                    not case.expected_dayparts
                    or document.daypart in case.expected_dayparts
                )
            )
        )

    def _ambiguity_summary(
        self,
        *,
        dataset: EmbeddingBenchmarkDataset,
        semantic_signature_ids: Mapping[str, Sequence[str]],
    ) -> dict[str, Any]:
        signature_groups: dict[
            tuple[str, str, str], list[str]
        ] = defaultdict(list)
        normalized_text_groups: dict[str, list[str]] = defaultdict(list)
        for document in dataset.documents:
            signature_groups[
                (document.location, document.category, document.daypart)
            ].append(document.document_id)
            normalized_text_groups[
                " ".join(document.text.lower().split())
            ].append(document.document_id)

        duplicate_signatures = {
            signature: tuple(sorted(document_ids))
            for signature, document_ids in signature_groups.items()
            if len(document_ids) > 1
        }
        duplicate_text_groups = sum(
            len(document_ids) > 1
            for document_ids in normalized_text_groups.values()
        )
        supported = [
            case
            for case in dataset.cases
            if not case.unsupported_location
        ]
        ambiguous = [
            case
            for case in supported
            if set(semantic_signature_ids[case.case_id])
            != set(case.relevant_document_ids)
        ]
        by_language = Counter(case.language for case in ambiguous)
        return {
            "unique_semantic_signature_count": len(signature_groups),
            "duplicate_semantic_signature_group_count": len(
                duplicate_signatures
            ),
            "duplicate_semantic_signature_document_count": sum(
                len(document_ids)
                for document_ids in duplicate_signatures.values()
            ),
            "duplicate_exact_text_group_count": duplicate_text_groups,
            "ambiguous_supported_case_count": len(ambiguous),
            "ambiguous_supported_case_rate": (
                len(ambiguous) / len(supported)
                if supported
                else 0.0
            ),
            "ambiguous_case_counts_by_language": dict(
                sorted(by_language.items())
            ),
        }

    def _aggregate_diagnostics(
        self,
        case_diagnostics: Sequence[Mapping[str, Any]],
        *,
        top_k: int,
    ) -> dict[str, Any]:
        supported = [
            value
            for value in case_diagnostics
            if not value["unsupported_location"]
        ]
        unsupported = [
            value
            for value in case_diagnostics
            if value["unsupported_location"]
        ]
        hard_negative_cases = [
            value
            for value in supported
            if value["hard_negative_excluded_at_k"] is not None
        ]
        hard_negative_margins = [
            float(value["exact_relevant_to_hard_negative_margin"])
            for value in hard_negative_cases
            if value["exact_relevant_to_hard_negative_margin"] is not None
        ]
        return {
            "supported_case_count": len(supported),
            "unsupported_case_count": len(unsupported),
            "exact_hit_at_k_rate": self._rate(
                value["exact_hit_at_k"] for value in supported
            ),
            "semantic_signature_hit_at_k_rate": self._rate(
                value["semantic_signature_hit_at_k"]
                for value in supported
            ),
            "exact_top1_accuracy": self._rate(
                bool(
                    value["exact_relevant_document_ids"]
                    and value["top_ids"][0]
                    in value["exact_relevant_document_ids"]
                )
                for value in supported
            ),
            "semantic_signature_top1_accuracy": self._rate(
                bool(
                    value["semantic_signature_relevant_document_ids"]
                    and value["top_ids"][0]
                    in value[
                        "semantic_signature_relevant_document_ids"
                    ]
                )
                for value in supported
            ),
            "top_k": int(top_k),
            "hard_negative_case_count": len(hard_negative_cases),
            "hard_negative_exclusion_at_k": self._rate(
                bool(value["hard_negative_excluded_at_k"])
                for value in hard_negative_cases
            ),
            "hard_negative_positive_margin_rate": self._rate(
                margin > 0.0
                for margin in hard_negative_margins
            ),
            "current_threshold_supported_acceptance_rate": self._rate(
                bool(value["accepted_at_current_threshold"])
                for value in supported
            ),
            "current_threshold_unsupported_false_match_rate": self._rate(
                bool(
                    value[
                        "unsupported_false_match_at_current_threshold"
                    ]
                )
                for value in unsupported
            ),
        }

    def _per_language_diagnostics(
        self,
        case_diagnostics: Sequence[Mapping[str, Any]],
        *,
        top_k: int,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for value in case_diagnostics:
            grouped[str(value["language"])].append(value)

        output: dict[str, dict[str, Any]] = {}
        for language, values in sorted(grouped.items()):
            supported = [
                value
                for value in values
                if not value["unsupported_location"]
            ]
            unsupported = [
                value
                for value in values
                if value["unsupported_location"]
            ]
            output[language] = {
                "case_count": len(values),
                "supported_case_count": len(supported),
                "unsupported_case_count": len(unsupported),
                "exact_hit_at_k_rate": self._rate(
                    value["exact_hit_at_k"] for value in supported
                ),
                "semantic_signature_hit_at_k_rate": self._rate(
                    value["semantic_signature_hit_at_k"]
                    for value in supported
                ),
                "exact_top1_accuracy": self._rate(
                    bool(
                        value["exact_relevant_document_ids"]
                        and value["top_ids"][0]
                        in value["exact_relevant_document_ids"]
                    )
                    for value in supported
                ),
                "semantic_signature_top1_accuracy": self._rate(
                    bool(
                        value[
                            "semantic_signature_relevant_document_ids"
                        ]
                        and value["top_ids"][0]
                        in value[
                            "semantic_signature_relevant_document_ids"
                        ]
                    )
                    for value in supported
                ),
                "current_threshold_supported_acceptance_rate": self._rate(
                    bool(value["accepted_at_current_threshold"])
                    for value in supported
                ),
                "current_threshold_unsupported_false_match_rate": (
                    self._rate(
                        bool(
                            value[
                                "unsupported_false_match_at_current_threshold"
                            ]
                        )
                        for value in unsupported
                    )
                ),
                "top_k": int(top_k),
            }
        return output

    def _score_distributions(
        self,
        case_diagnostics: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, float | int | None]]:
        supported = [
            value
            for value in case_diagnostics
            if not value["unsupported_location"]
        ]
        unsupported = [
            value
            for value in case_diagnostics
            if value["unsupported_location"]
        ]
        return {
            "supported_top_score": self._distribution(
                [float(value["top_score"]) for value in supported]
            ),
            "unsupported_top_score": self._distribution(
                [float(value["top_score"]) for value in unsupported]
            ),
            "best_exact_relevant_score": self._distribution(
                [
                    float(value["best_exact_relevant_score"])
                    for value in supported
                    if value["best_exact_relevant_score"] is not None
                ]
            ),
            "best_hard_negative_score": self._distribution(
                [
                    float(value["best_hard_negative_score"])
                    for value in supported
                    if value["best_hard_negative_score"] is not None
                ]
            ),
            "exact_relevant_to_hard_negative_margin": self._distribution(
                [
                    float(
                        value[
                            "exact_relevant_to_hard_negative_margin"
                        ]
                    )
                    for value in supported
                    if value[
                        "exact_relevant_to_hard_negative_margin"
                    ]
                    is not None
                ]
            ),
            "top1_score_margin": self._distribution(
                [
                    float(value["top1_score_margin"])
                    for value in case_diagnostics
                    if value["top1_score_margin"] is not None
                ]
            ),
        }

    def _calibration_evidence(
        self,
        case_diagnostics: Sequence[Mapping[str, Any]],
        *,
        current_threshold: float,
        target_false_match_rate: float,
        confidence_level: float,
    ) -> dict[str, Any]:
        supported = [
            value
            for value in case_diagnostics
            if not value["unsupported_location"]
        ]
        unsupported = [
            value
            for value in case_diagnostics
            if value["unsupported_location"]
        ]
        supported_scores = [float(value["top_score"]) for value in supported]
        unsupported_scores = [
            float(value["top_score"]) for value in unsupported
        ]

        minimum_nonzero_rate = (
            1.0 / len(unsupported_scores)
            if unsupported_scores
            else None
        )
        alpha = 1.0 - confidence_level
        minimum_zero_failure_sample_count = (
            int(
                math.ceil(
                    math.log(alpha)
                    / math.log(1.0 - target_false_match_rate)
                )
            )
            if 0.0 < target_false_match_rate < 1.0
            else None
        )

        candidates = {-1.0, 1.0, current_threshold}
        for score in unsupported_scores:
            candidates.add(float(score))
            above = float(np.nextafter(score, math.inf))
            if above <= 1.0:
                candidates.add(above)

        evaluated = []
        for threshold in sorted(candidates):
            unsupported_false_matches = sum(
                score >= threshold
                for score in unsupported_scores
            )
            unsupported_rate = (
                unsupported_false_matches / len(unsupported_scores)
                if unsupported_scores
                else 0.0
            )
            supported_accepted = [
                value
                for value in supported
                if float(value["top_score"]) >= threshold
            ]
            supported_acceptance_rate = (
                len(supported_accepted) / len(supported)
                if supported
                else 0.0
            )
            exact_correct_accepted = sum(
                bool(value["exact_hit_at_k"])
                for value in supported_accepted
            )
            accepted_exact_precision = (
                exact_correct_accepted / len(supported_accepted)
                if supported_accepted
                else 0.0
            )
            evaluated.append(
                {
                    "threshold": float(threshold),
                    "unsupported_false_match_count": (
                        unsupported_false_matches
                    ),
                    "unsupported_false_match_rate": unsupported_rate,
                    "supported_acceptance_rate": supported_acceptance_rate,
                    "accepted_supported_exact_hit_precision": (
                        accepted_exact_precision
                    ),
                }
            )

        feasible = [
            value
            for value in evaluated
            if value["unsupported_false_match_rate"]
            <= target_false_match_rate + 1e-12
        ]
        recommended = (
            sorted(
                feasible,
                key=lambda value: (
                    -float(value["supported_acceptance_rate"]),
                    -float(
                        value[
                            "accepted_supported_exact_hit_precision"
                        ]
                    ),
                    float(value["threshold"]),
                ),
            )[0]
            if feasible
            else None
        )

        current = min(
            evaluated,
            key=lambda value: abs(
                float(value["threshold"]) - current_threshold
            ),
        )
        zero_failure_upper_bound = (
            1.0
            - math.pow(
                1.0 - confidence_level,
                1.0 / len(unsupported_scores),
            )
            if unsupported_scores
            and current["unsupported_false_match_count"] == 0
            else None
        )
        calibration_ready = bool(
            recommended
            and minimum_zero_failure_sample_count is not None
            and len(unsupported_scores)
            >= minimum_zero_failure_sample_count
            and recommended["unsupported_false_match_count"] == 0
        )

        return {
            "status": (
                "candidate_requires_human_review"
                if recommended
                else "blocked_no_feasible_threshold"
            ),
            "selection_objective": (
                "maximize_supported_acceptance_subject_to_target_"
                "unsupported_false_match_rate"
            ),
            "current_threshold": current,
            "recommended_candidate": recommended,
            "recommended_threshold_auto_applied": False,
            "target_unsupported_false_match_rate": (
                target_false_match_rate
            ),
            "unsupported_calibration_case_count": len(
                unsupported_scores
            ),
            "minimum_observable_nonzero_false_match_rate": (
                minimum_nonzero_rate
            ),
            "target_below_empirical_sample_resolution": bool(
                minimum_nonzero_rate is not None
                and target_false_match_rate < minimum_nonzero_rate
            ),
            "confidence_level": confidence_level,
            "zero_failure_upper_confidence_bound_at_current_threshold": (
                zero_failure_upper_bound
            ),
            "minimum_zero_failure_sample_count_for_confidence_target": (
                minimum_zero_failure_sample_count
            ),
            "production_calibration_ready": calibration_ready,
            "production_calibration_blockers": [
                blocker
                for blocker, active in (
                    (
                        "no_feasible_threshold",
                        recommended is None,
                    ),
                    (
                        "insufficient_unsupported_calibration_cases",
                        minimum_zero_failure_sample_count is not None
                        and len(unsupported_scores)
                        < minimum_zero_failure_sample_count,
                    ),
                    (
                        "native_human_threshold_approval_required",
                        True,
                    ),
                )
                if active
            ],
        }

    def _distribution(
        self,
        values: Sequence[float],
    ) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "minimum": None,
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
                "maximum": None,
                "mean": None,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "minimum": float(np.min(array)),
            "p05": float(np.percentile(array, 5)),
            "p25": float(np.percentile(array, 25)),
            "p50": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
            "p95": float(np.percentile(array, 95)),
            "maximum": float(np.max(array)),
            "mean": float(np.mean(array)),
        }

    def _validate_matrix(
        self,
        matrix: np.ndarray,
        *,
        rows: int,
        dimension: int,
        label: str,
    ) -> None:
        if matrix.shape != (rows, dimension):
            raise ValueError(
                f"Benchmark diagnostics {label} embedding shape is invalid."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Benchmark diagnostics {label} embeddings contain "
                "non-finite values."
            )
        if np.any(np.linalg.norm(matrix, axis=1) <= 0):
            raise ValueError(
                f"Benchmark diagnostics {label} embeddings contain zero "
                "vectors."
            )

    def _unit_rows(self, matrix: np.ndarray) -> np.ndarray:
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def _rate(self, values: Sequence[Any] | Any) -> float:
        materialized = [bool(value) for value in values]
        return (
            float(sum(materialized) / len(materialized))
            if materialized
            else 0.0
        )

    def _bounded_rate_or_score(
        self,
        value: Any,
        label: str,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.") from None
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must be finite.")
        if not minimum <= parsed <= maximum:
            raise ValueError(
                f"{label} must be between {minimum} and {maximum}."
            )
        return parsed

    def _peak_memory_mb(self) -> float:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name == "posix" and value > 1024 * 1024:
            return value / (1024 * 1024)
        return value / 1024.0


class ProductionEmbeddingBenchmarkDiagnosticsValidator:
    """Validate privacy, lineage, structure, and immutability of diagnostics."""

    def validate(
        self,
        diagnostics: Mapping[str, Any],
        *,
        model: EmbeddingModelSpec,
        dataset: EmbeddingBenchmarkDataset,
    ) -> dict[str, Any]:
        if not isinstance(diagnostics, Mapping):
            raise TypeError("benchmark_diagnostics must be an object.")
        try:
            safe = json.loads(
                json.dumps(dict(diagnostics), allow_nan=False)
            )
        except (TypeError, ValueError):
            raise ValueError(
                "benchmark_diagnostics must contain finite JSON values."
            ) from None

        required = {
            "schema_version",
            "benchmark_id",
            "dataset_version",
            "dataset_fingerprint",
            "dataset_safety",
            "model",
            "evaluation",
            "ambiguity",
            "aggregate",
            "per_language",
            "score_distributions",
            "calibration",
            "case_diagnostics",
            "peak_memory_mb",
            "generated_at",
            "registration_allowed",
            "recommended_threshold_auto_applied",
            "activation_or_export_performed",
            "credentials_exposed",
            "raw_identifiers_read",
            "raw_identifiers_stored",
            "diagnostics_fingerprint",
        }
        if required.difference(safe):
            raise ValueError(
                "benchmark_diagnostics is missing required evidence fields."
            )
        if (
            safe["schema_version"]
            != BENCHMARK_DIAGNOSTICS_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported benchmark diagnostics schema.")
        if (
            safe["benchmark_id"] != dataset.benchmark_id
            or safe["dataset_version"] != dataset.dataset_version
            or safe["dataset_fingerprint"] != dataset.fingerprint
        ):
            raise ValueError(
                "benchmark_diagnostics does not match the supplied dataset."
            )
        if safe["model"] != model.to_safe_dict():
            raise ValueError(
                "benchmark_diagnostics model does not match the requested "
                "model."
            )
        expected_safety = {
            "privacy_status": dataset.privacy_status,
            "rights_status": dataset.rights_status,
            "contains_raw_identifiers": False,
        }
        if safe["dataset_safety"] != expected_safety:
            raise ValueError(
                "benchmark_diagnostics dataset safety evidence is invalid."
            )
        if any(
            safe[key] is not False
            for key in (
                "registration_allowed",
                "recommended_threshold_auto_applied",
                "activation_or_export_performed",
                "credentials_exposed",
                "raw_identifiers_read",
                "raw_identifiers_stored",
            )
        ):
            raise ValueError(
                "benchmark_diagnostics violates the offline review boundary."
            )

        cases = safe["case_diagnostics"]
        if not isinstance(cases, list):
            raise TypeError(
                "benchmark_diagnostics case_diagnostics must be an array."
            )
        expected_case_ids = {value.case_id for value in dataset.cases}
        observed_case_ids = {
            str(value.get("case_id") or "")
            for value in cases
            if isinstance(value, dict)
        }
        if (
            len(cases) != len(dataset.cases)
            or observed_case_ids != expected_case_ids
        ):
            raise ValueError(
                "benchmark_diagnostics cases do not match the dataset."
            )

        forbidden_keys = {
            "query",
            "document_text",
            "text",
            "embedding",
            "embeddings",
            "vector",
            "vectors",
            "maid",
            "device_id",
            "email",
            "phone",
            "latitude",
            "longitude",
        }
        self._reject_forbidden_keys(safe, forbidden_keys)

        calibration = safe["calibration"]
        if (
            not isinstance(calibration, dict)
            or calibration.get("recommended_threshold_auto_applied")
            is not False
            or calibration.get("production_calibration_ready") is True
            and calibration.get("production_calibration_blockers")
        ):
            raise ValueError(
                "benchmark_diagnostics calibration evidence is inconsistent."
            )

        supplied_fingerprint = str(
            safe.pop("diagnostics_fingerprint")
        ).lower()
        if supplied_fingerprint != stable_digest(safe):
            raise ValueError(
                "benchmark_diagnostics fingerprint verification failed."
            )
        safe["diagnostics_fingerprint"] = supplied_fingerprint
        return safe

    def _reject_forbidden_keys(
        self,
        value: Any,
        forbidden_keys: set[str],
    ) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    raise ValueError(
                        "benchmark_diagnostics contains forbidden raw content."
                    )
                self._reject_forbidden_keys(child, forbidden_keys)
        elif isinstance(value, list):
            for child in value:
                self._reject_forbidden_keys(child, forbidden_keys)
