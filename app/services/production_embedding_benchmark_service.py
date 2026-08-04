from __future__ import annotations

import json
import math
import os
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from app.models.audience_feature_contracts import stable_digest
from app.models.embedding_benchmark_contracts import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    EmbeddingBenchmarkCase,
    EmbeddingBenchmarkDataset,
    ProductionEmbeddingBenchmarkPolicy,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec

REQUIRED_BENCHMARK_METRICS = {
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "mrr",
    "geographic_constraint_accuracy",
    "category_constraint_accuracy",
    "daypart_accuracy",
    "hard_negative_rejection_rate",
    "unsupported_location_false_match_rate",
    "multilingual_consistency",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "peak_memory_mb",
    "cost_per_1000_queries_usd",
}

REQUIRED_BENCHMARK_COVERAGE = {
    "case_count",
    "document_count",
    "language_count",
    "geographic_case_count",
    "category_case_count",
    "daypart_case_count",
    "hard_negative_case_count",
    "unsupported_location_case_count",
    "multilingual_group_count",
    "location_value_count",
    "category_value_count",
    "daypart_value_count",
    "minimum_cases_per_language",
    "constraint_intersection_case_count",
}


def theoretical_maximum_precision_at_k(
    dataset: EmbeddingBenchmarkDataset,
    *,
    top_k: int,
) -> float:
    """Return the best standard precision@k the labels can ever achieve."""

    k = int(top_k)
    if not 1 <= k <= len(dataset.documents):
        raise ValueError(
            "top_k must be between 1 and the benchmark document count."
        )
    supported = [
        value
        for value in dataset.cases
        if not value.unsupported_location
    ]
    if not supported:
        return 0.0
    return float(
        sum(
            min(len(set(value.relevant_document_ids)), k) / k
            for value in supported
        )
        / len(supported)
    )


def benchmark_dataset_coverage(
    dataset: EmbeddingBenchmarkDataset,
) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for case in dataset.cases:
        if case.semantic_group_id:
            groups[case.semantic_group_id].add(case.language)
    multilingual_groups = sum(
        len(languages) >= 2
        for languages in groups.values()
    )
    language_counts = Counter(
        value.language
        for value in dataset.cases
    )
    return {
        "case_count": len(dataset.cases),
        "document_count": len(dataset.documents),
        "language_count": len(
            {value.language for value in dataset.cases}
        ),
        "geographic_case_count": sum(
            bool(value.expected_locations)
            for value in dataset.cases
        ),
        "category_case_count": sum(
            bool(value.expected_categories)
            for value in dataset.cases
        ),
        "daypart_case_count": sum(
            bool(value.expected_dayparts)
            for value in dataset.cases
        ),
        "hard_negative_case_count": sum(
            bool(value.hard_negative_document_ids)
            for value in dataset.cases
        ),
        "unsupported_location_case_count": sum(
            value.unsupported_location
            for value in dataset.cases
        ),
        "multilingual_group_count": multilingual_groups,
        "location_value_count": len(
            {value.location for value in dataset.documents}
        ),
        "category_value_count": len(
            {value.category for value in dataset.documents}
        ),
        "daypart_value_count": len(
            {value.daypart for value in dataset.documents}
        ),
        "minimum_cases_per_language": min(
            language_counts.values(),
            default=0,
        ),
        "constraint_intersection_case_count": sum(
            bool(
                value.expected_locations
                and value.expected_categories
                and value.expected_dayparts
            )
            for value in dataset.cases
        ),
    }


class EmbeddingBenchmarkEncoder(Protocol):
    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray: ...

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray: ...


class SentenceTransformerEmbeddingBenchmarkEncoder:
    """Pinned query/document inference with no mutable-model fallback."""

    def __init__(self, *, device: str | None = None) -> None:
        self._device = (
            device
            or os.getenv("FEATURE_EMBEDDING_DEVICE")
            or "cpu"
        )
        self._models: dict[tuple[str, str, str], Any] = {}

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray:
        return self._encode(
            texts,
            prefix=model.document_prefix,
            model=model,
            batch_size=batch_size,
        )

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray:
        return self._encode(
            texts,
            prefix=model.query_prefix,
            model=model,
            batch_size=batch_size,
        )

    def _encode(
        self,
        texts: Sequence[str],
        *,
        prefix: str,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray:
        if model.backend != "sentence_transformers":
            raise RuntimeError(
                "The local benchmark runner only supports "
                "sentence_transformers."
            )
        from sentence_transformers import SentenceTransformer

        key = (model.model_name, model.model_revision, self._device)
        loaded = self._models.get(key)
        if loaded is None:
            loaded = SentenceTransformer(
                model.model_name,
                revision=model.model_revision,
                device=self._device,
                trust_remote_code=False,
            )
            self._models[key] = loaded
        vectors = loaded.encode(
            [f"{prefix}{value}" for value in texts],
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=model.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class ProductionEmbeddingBenchmarkService:
    """
    Reproducible global retrieval evaluation for one immutable model revision.

    The evaluator ranks only privacy-safe benchmark documents. It calculates
    the pass decision from a versioned production policy; callers cannot set
    `passed` directly.
    """

    def __init__(
        self,
        *,
        encoder: EmbeddingBenchmarkEncoder | None = None,
        policy: ProductionEmbeddingBenchmarkPolicy | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now_fn: Callable[[], datetime] | None = None,
        memory_mb_fn: Callable[[], float] | None = None,
    ) -> None:
        self._encoder = (
            encoder or SentenceTransformerEmbeddingBenchmarkEncoder()
        )
        self._policy = policy or ProductionEmbeddingBenchmarkPolicy()
        self._clock = clock
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._memory_mb_fn = memory_mb_fn or self._peak_memory_mb

    def evaluate(
        self,
        *,
        dataset: EmbeddingBenchmarkDataset,
        model: EmbeddingModelSpec,
        top_k: int = 10,
        batch_size: int = 64,
        rejection_similarity_threshold: float = 0.78,
        cost_per_1000_queries_usd: float = 0.0,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not 1 <= int(top_k) <= len(dataset.documents):
            raise ValueError(
                "top_k must be between 1 and the benchmark document count."
            )
        if not 1 <= int(batch_size) <= 4096:
            raise ValueError("batch_size must be between 1 and 4096.")
        rejection_threshold = self._finite_float(
            rejection_similarity_threshold,
            "rejection_similarity_threshold",
        )
        if not -1.0 <= rejection_threshold <= 1.0:
            raise ValueError(
                "rejection_similarity_threshold must be between -1 and 1."
            )
        cost = self._finite_float(
            cost_per_1000_queries_usd,
            "cost_per_1000_queries_usd",
        )
        if cost < 0:
            raise ValueError(
                "cost_per_1000_queries_usd cannot be negative."
            )
        thresholds = self._policy.resolve_thresholds(
            threshold_overrides
        )
        theoretical_precision = theoretical_maximum_precision_at_k(
            dataset,
            top_k=int(top_k),
        )
        required_precision = float(thresholds["precision_at_k"])
        if theoretical_precision + 1e-12 < required_precision:
            raise ValueError(
                "Benchmark dataset labels and top_k cannot satisfy the "
                "production precision_at_k threshold: theoretical maximum "
                f"is {theoretical_precision:.6f}, but policy requires "
                f"{required_precision:.6f}. Reduce top_k or add reviewed "
                "relevant documents per supported case."
            )

        document_vectors = np.asarray(
            self._encoder.encode_documents(
                [value.text for value in dataset.documents],
                model=model,
                batch_size=int(batch_size),
            ),
            dtype=np.float64,
        )
        self._validate_matrix(
            document_vectors,
            rows=len(dataset.documents),
            dimension=model.dimension,
            label="document",
        )
        document_vectors = self._unit_rows(document_vectors)
        document_by_id = {
            value.document_id: value
            for value in dataset.documents
        }
        document_ids = [
            value.document_id
            for value in dataset.documents
        ]

        case_results: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        for case in dataset.cases:
            started = self._clock()
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
            elapsed_ms = max(
                0.0,
                (self._clock() - started) * 1000.0,
            )
            latencies_ms.append(elapsed_ms)
            ranked_ids = [
                document_ids[int(position)]
                for position in ranked_positions
            ]
            case_results.append(
                self._case_result(
                    case=case,
                    ranked_ids=ranked_ids,
                    scores=scores,
                    ranked_positions=ranked_positions,
                    document_by_id=document_by_id,
                    top_k=int(top_k),
                    rejection_similarity_threshold=rejection_threshold,
                )
            )

        coverage = self._coverage(dataset)
        metrics = self._aggregate_metrics(
            dataset=dataset,
            case_results=case_results,
            latencies_ms=latencies_ms,
            cost_per_1000_queries_usd=cost,
            peak_memory_mb=self._memory_mb_fn(),
        )
        threshold_results = self._threshold_results(
            metrics=metrics,
            coverage=coverage,
            thresholds=thresholds,
        )
        passed = all(
            value["passed"]
            for value in threshold_results.values()
        )
        report: dict[str, Any] = {
            "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
            "benchmark_id": dataset.benchmark_id,
            "dataset_version": dataset.dataset_version,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_safety": {
                "privacy_status": dataset.privacy_status,
                "rights_status": dataset.rights_status,
                "contains_raw_identifiers": False,
            },
            "model": model.to_safe_dict(),
            "policy": self._policy.to_dict(),
            "evaluation": {
                "top_k": int(top_k),
                "batch_size": int(batch_size),
                "rejection_similarity_threshold": rejection_threshold,
                "theoretical_maximum_precision_at_k": (
                    theoretical_precision
                ),
                **coverage,
                "languages": sorted(
                    {value.language for value in dataset.cases}
                ),
            },
            "thresholds": thresholds,
            "metrics": metrics,
            "threshold_results": threshold_results,
            "passed": passed,
            "generated_at": self._now_fn().astimezone(
                timezone.utc
            ).isoformat(),
            "activation_or_export_performed": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
        }
        report["report_fingerprint"] = stable_digest(report)
        return report

    def _case_result(
        self,
        *,
        case: EmbeddingBenchmarkCase,
        ranked_ids: list[str],
        scores: np.ndarray,
        ranked_positions: np.ndarray,
        document_by_id: Mapping[str, Any],
        top_k: int,
        rejection_similarity_threshold: float,
    ) -> dict[str, Any]:
        top_ids = ranked_ids[:top_k]
        relevant = set(case.relevant_document_ids)
        relevant_hits = [
            document_id
            for document_id in top_ids
            if document_id in relevant
        ]
        first_relevant_rank = next(
            (
                position + 1
                for position, document_id in enumerate(ranked_ids)
                if document_id in relevant
            ),
            None,
        )
        top_document = document_by_id[top_ids[0]]
        hard_negatives = set(case.hard_negative_document_ids)
        rejected_hard_negatives = (
            len(hard_negatives.difference(top_ids))
            if hard_negatives
            else None
        )
        top_score = float(scores[int(ranked_positions[0])])
        return {
            "case_id": case.case_id,
            "language": case.language,
            "semantic_group_id": case.semantic_group_id,
            "unsupported_location": case.unsupported_location,
            "top_ids": top_ids,
            "relevant_hits": relevant_hits,
            "relevant_count": len(relevant),
            "first_relevant_rank": first_relevant_rank,
            "top_score": top_score,
            "unsupported_false_match": (
                case.unsupported_location
                and top_score >= rejection_similarity_threshold
            ),
            "geographic_match": (
                top_document.location in case.expected_locations
                if case.expected_locations
                else None
            ),
            "category_match": (
                top_document.category in case.expected_categories
                if case.expected_categories
                else None
            ),
            "daypart_match": (
                top_document.daypart in case.expected_dayparts
                if case.expected_dayparts
                else None
            ),
            "hard_negative_count": len(hard_negatives),
            "rejected_hard_negative_count": rejected_hard_negatives,
        }

    def _aggregate_metrics(
        self,
        *,
        dataset: EmbeddingBenchmarkDataset,
        case_results: Sequence[dict[str, Any]],
        latencies_ms: Sequence[float],
        cost_per_1000_queries_usd: float,
        peak_memory_mb: float,
    ) -> dict[str, float]:
        supported = [
            value
            for value in case_results
            if not value["unsupported_location"]
        ]
        recalls = [
            len(value["relevant_hits"]) / value["relevant_count"]
            for value in supported
            if value["relevant_count"]
        ]
        precisions = [
            len(value["relevant_hits"]) / len(value["top_ids"])
            for value in supported
        ]
        ndcgs = [
            self._ndcg(
                value["top_ids"],
                set(
                    next(
                        case.relevant_document_ids
                        for case in dataset.cases
                        if case.case_id == value["case_id"]
                    )
                ),
            )
            for value in supported
        ]
        reciprocal_ranks = [
            (
                1.0 / value["first_relevant_rank"]
                if value["first_relevant_rank"]
                else 0.0
            )
            for value in supported
        ]
        unsupported = [
            value
            for value in case_results
            if value["unsupported_location"]
        ]
        hard_negative_results = [
            value
            for value in case_results
            if value["hard_negative_count"]
        ]
        hard_negative_total = sum(
            value["hard_negative_count"]
            for value in hard_negative_results
        )
        hard_negative_rejected = sum(
            int(value["rejected_hard_negative_count"] or 0)
            for value in hard_negative_results
        )
        return {
            "recall_at_k": self._mean(recalls),
            "precision_at_k": self._mean(precisions),
            "ndcg_at_k": self._mean(ndcgs),
            "mrr": self._mean(reciprocal_ranks),
            "geographic_constraint_accuracy": self._optional_accuracy(
                supported,
                "geographic_match",
            ),
            "category_constraint_accuracy": self._optional_accuracy(
                case_results,
                "category_match",
            ),
            "daypart_accuracy": self._optional_accuracy(
                case_results,
                "daypart_match",
            ),
            "hard_negative_rejection_rate": (
                hard_negative_rejected / hard_negative_total
                if hard_negative_total
                else 0.0
            ),
            "unsupported_location_false_match_rate": (
                sum(
                    bool(value["unsupported_false_match"])
                    for value in unsupported
                )
                / len(unsupported)
                if unsupported
                else 0.0
            ),
            "multilingual_consistency": (
                self._multilingual_consistency(case_results)
            ),
            "p50_latency_ms": self._percentile(latencies_ms, 50),
            "p95_latency_ms": self._percentile(latencies_ms, 95),
            "p99_latency_ms": self._percentile(latencies_ms, 99),
            "peak_memory_mb": max(0.0, float(peak_memory_mb)),
            "cost_per_1000_queries_usd": float(
                cost_per_1000_queries_usd
            ),
        }

    def _coverage(
        self,
        dataset: EmbeddingBenchmarkDataset,
    ) -> dict[str, int]:
        return benchmark_dataset_coverage(dataset)

    def _threshold_results(
        self,
        *,
        metrics: Mapping[str, float],
        coverage: Mapping[str, int],
        thresholds: Mapping[str, float | int],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for name in sorted(REQUIRED_BENCHMARK_METRICS):
            observed = float(metrics[name])
            threshold = float(thresholds[name])
            maximum = name in self._policy.maximum_metric_thresholds
            results[name] = {
                "operator": "<=" if maximum else ">=",
                "threshold": threshold,
                "observed": observed,
                "passed": (
                    observed <= threshold
                    if maximum
                    else observed >= threshold
                ),
            }
        for name in sorted(REQUIRED_BENCHMARK_COVERAGE):
            observed = int(coverage[name])
            threshold = int(thresholds[name])
            results[name] = {
                "operator": ">=",
                "threshold": threshold,
                "observed": observed,
                "passed": observed >= threshold,
            }
        return results

    def _multilingual_consistency(
        self,
        case_results: Sequence[dict[str, Any]],
    ) -> float:
        groups: dict[str, list[set[str]]] = defaultdict(list)
        for value in case_results:
            group_id = value.get("semantic_group_id")
            if group_id:
                groups[str(group_id)].append(set(value["top_ids"]))
        similarities: list[float] = []
        for rankings in groups.values():
            if len(rankings) < 2:
                continue
            for left_index in range(len(rankings)):
                for right_index in range(left_index + 1, len(rankings)):
                    union = rankings[left_index].union(
                        rankings[right_index]
                    )
                    similarities.append(
                        len(
                            rankings[left_index].intersection(
                                rankings[right_index]
                            )
                        )
                        / len(union)
                        if union
                        else 1.0
                    )
        return self._mean(similarities)

    def _ndcg(self, ranked_ids: Sequence[str], relevant: set[str]) -> float:
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position, document_id in enumerate(ranked_ids)
            if document_id in relevant
        )
        ideal_hits = min(len(relevant), len(ranked_ids))
        ideal = sum(
            1.0 / math.log2(position + 2)
            for position in range(ideal_hits)
        )
        return dcg / ideal if ideal else 0.0

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
                f"Benchmark {label} embedding shape is invalid."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Benchmark {label} embeddings contain non-finite values."
            )
        if np.any(np.linalg.norm(matrix, axis=1) <= 0):
            raise ValueError(
                f"Benchmark {label} embeddings contain zero vectors."
            )

    def _unit_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / norms

    def _optional_accuracy(
        self,
        results: Sequence[dict[str, Any]],
        key: str,
    ) -> float:
        values = [
            bool(value[key])
            for value in results
            if value[key] is not None
        ]
        return self._mean(values)

    def _mean(self, values: Sequence[Any]) -> float:
        return (
            float(sum(float(value) for value in values) / len(values))
            if values
            else 0.0
        )

    def _percentile(
        self,
        values: Sequence[float],
        percentile: float,
    ) -> float:
        return (
            float(np.percentile(np.asarray(values), percentile))
            if values
            else 0.0
        )

    def _finite_float(self, value: Any, label: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.") from None
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must be finite.")
        return parsed

    def _peak_memory_mb(self) -> float:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name == "posix" and value > 1024 * 1024:
            return value / (1024 * 1024)
        return value / 1024.0


class ProductionEmbeddingBenchmarkReportValidator:
    """Fail-closed validation of benchmark evidence before model approval."""

    def __init__(
        self,
        *,
        policy: ProductionEmbeddingBenchmarkPolicy | None = None,
    ) -> None:
        self._policy = policy or ProductionEmbeddingBenchmarkPolicy()

    def validate(
        self,
        report: Mapping[str, Any],
        *,
        model: EmbeddingModelSpec,
        dataset: EmbeddingBenchmarkDataset,
    ) -> dict[str, Any]:
        if not isinstance(report, Mapping):
            raise TypeError("benchmark_report must be an object.")
        try:
            safe = json.loads(
                json.dumps(dict(report), allow_nan=False)
            )
        except (TypeError, ValueError):
            raise ValueError(
                "benchmark_report must contain finite JSON values."
            ) from None
        required = {
            "schema_version",
            "benchmark_id",
            "dataset_version",
            "dataset_fingerprint",
            "dataset_safety",
            "model",
            "policy",
            "evaluation",
            "thresholds",
            "metrics",
            "threshold_results",
            "passed",
            "generated_at",
            "report_fingerprint",
            "activation_or_export_performed",
            "raw_identifiers_read",
            "raw_identifiers_stored",
        }
        if required.difference(safe):
            raise ValueError(
                "benchmark_report is missing required evidence fields."
            )
        if safe["schema_version"] != BENCHMARK_REPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported benchmark report schema.")
        for key in ("benchmark_id", "dataset_version"):
            if not str(safe[key] or "").strip():
                raise ValueError(f"benchmark_report {key} is required.")
        fingerprint = str(safe["dataset_fingerprint"] or "").lower()
        if len(fingerprint) != 64 or any(
            value not in "0123456789abcdef"
            for value in fingerprint
        ):
            raise ValueError(
                "benchmark_report dataset_fingerprint is invalid."
            )
        if (
            safe["benchmark_id"] != dataset.benchmark_id
            or safe["dataset_version"] != dataset.dataset_version
            or fingerprint != dataset.fingerprint
        ):
            raise ValueError(
                "benchmark_report does not match the supplied dataset."
            )
        dataset_safety = safe["dataset_safety"]
        if (
            not isinstance(dataset_safety, dict)
            or dataset_safety.get("privacy_status")
            not in {"safe", "passed", "privacy_safe"}
            or dataset_safety.get("rights_status")
            not in {
                "permitted",
                "offline_evaluation_only",
                "synthetic_evaluation",
            }
            or dataset_safety.get("contains_raw_identifiers") is not False
        ):
            raise ValueError(
                "benchmark_report dataset safety evidence is invalid."
            )
        expected_dataset_safety = {
            "privacy_status": dataset.privacy_status,
            "rights_status": dataset.rights_status,
            "contains_raw_identifiers": False,
        }
        if dataset_safety != expected_dataset_safety:
            raise ValueError(
                "benchmark_report dataset safety evidence conflicts "
                "with the supplied dataset."
            )
        if safe["model"] != model.to_safe_dict():
            raise ValueError(
                "benchmark_report model does not match the requested model."
            )
        policy = safe["policy"]
        if (
            not isinstance(policy, dict)
            or policy != self._policy.to_dict()
        ):
            raise ValueError(
                "benchmark_report policy does not match production policy."
            )
        if not isinstance(safe["evaluation"], dict):
            raise TypeError("benchmark_report evaluation is required.")
        expected_coverage = benchmark_dataset_coverage(dataset)
        observed_coverage = {
            key: safe["evaluation"].get(key)
            for key in REQUIRED_BENCHMARK_COVERAGE
        }
        if observed_coverage != expected_coverage:
            raise ValueError(
                "benchmark_report coverage does not match the dataset."
            )
        if safe["evaluation"].get("languages") != sorted(
            {value.language for value in dataset.cases}
        ):
            raise ValueError(
                "benchmark_report languages do not match the dataset."
            )
        if not isinstance(safe["metrics"], dict):
            raise TypeError("benchmark_report metrics are required.")
        if REQUIRED_BENCHMARK_METRICS.difference(safe["metrics"]):
            raise ValueError(
                "benchmark_report is missing required metrics."
            )
        thresholds = self._policy.resolve_thresholds(safe["thresholds"])
        if thresholds != safe["thresholds"]:
            raise ValueError(
                "benchmark_report thresholds are not canonical."
            )
        try:
            top_k = int(safe["evaluation"].get("top_k"))
        except (TypeError, ValueError):
            raise ValueError(
                "benchmark_report top_k is invalid."
            ) from None
        theoretical_precision = theoretical_maximum_precision_at_k(
            dataset,
            top_k=top_k,
        )
        observed_theoretical_precision = safe["evaluation"].get(
            "theoretical_maximum_precision_at_k"
        )
        if (
            not isinstance(observed_theoretical_precision, (int, float))
            or not math.isfinite(float(observed_theoretical_precision))
            or not math.isclose(
                float(observed_theoretical_precision),
                theoretical_precision,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "benchmark_report theoretical precision evidence is invalid."
            )
        if theoretical_precision + 1e-12 < float(
            thresholds["precision_at_k"]
        ):
            raise ValueError(
                "benchmark_report dataset labels and top_k cannot satisfy "
                "the production precision_at_k threshold."
            )
        expected_results = self._expected_threshold_results(
            metrics=safe["metrics"],
            evaluation=safe["evaluation"],
            thresholds=thresholds,
        )
        if safe["threshold_results"] != expected_results:
            raise ValueError(
                "benchmark_report threshold evidence is inconsistent."
            )
        expected_pass = all(
            value["passed"]
            for value in expected_results.values()
        )
        if safe["passed"] is not expected_pass or not expected_pass:
            raise ValueError(
                "benchmark_report must pass every production threshold."
            )
        if (
            safe["activation_or_export_performed"] is not False
            or safe["raw_identifiers_read"] is not False
            or safe["raw_identifiers_stored"] is not False
        ):
            raise ValueError(
                "benchmark_report violates the offline privacy boundary."
            )
        generated_at = str(safe["generated_at"] or "")
        try:
            parsed_generated_at = datetime.fromisoformat(
                generated_at.replace("Z", "+00:00")
            )
        except ValueError:
            raise ValueError(
                "benchmark_report generated_at is invalid."
            ) from None
        if parsed_generated_at.tzinfo is None:
            raise ValueError(
                "benchmark_report generated_at must include a timezone."
            )
        supplied_fingerprint = str(
            safe.pop("report_fingerprint")
        ).lower()
        if supplied_fingerprint != stable_digest(safe):
            raise ValueError(
                "benchmark_report fingerprint verification failed."
            )
        safe["report_fingerprint"] = supplied_fingerprint
        return safe

    def _expected_threshold_results(
        self,
        *,
        metrics: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        thresholds: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for name in sorted(REQUIRED_BENCHMARK_METRICS):
            observed = float(metrics[name])
            threshold = float(thresholds[name])
            maximum = name in self._policy.maximum_metric_thresholds
            results[name] = {
                "operator": "<=" if maximum else ">=",
                "threshold": threshold,
                "observed": observed,
                "passed": (
                    observed <= threshold
                    if maximum
                    else observed >= threshold
                ),
            }
        for name in sorted(REQUIRED_BENCHMARK_COVERAGE):
            if name not in evaluation:
                raise ValueError(
                    "benchmark_report evaluation is missing coverage."
                )
            observed = int(evaluation[name])
            threshold = int(thresholds[name])
            results[name] = {
                "operator": ">=",
                "threshold": threshold,
                "observed": observed,
                "passed": observed >= threshold,
            }
        return results
