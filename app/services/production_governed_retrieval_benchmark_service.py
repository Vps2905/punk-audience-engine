from __future__ import annotations

import math
import resource
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app.models.audience_feature_contracts import normalize_taxonomy_value, stable_digest
from app.models.embedding_benchmark_contracts import EmbeddingBenchmarkDataset
from app.models.production_audience_retrieval_contracts import (
    ConstraintTaxonomyEntry,
    GovernedAudienceRetrievalRequest,
    GovernedConstraintTaxonomy,
    ProductionRetrievalModelBinding,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.models.production_governed_retrieval_benchmark_contracts import (
    GovernedRetrievalBenchmarkDatasetIdentity,
    GovernedRetrievalBenchmarkPolicy,
)
from app.services.production_dual_model_candidate_retrieval_service import (
    ProductionDualModelCandidateRetrievalService,
    ProductionQueryEncoder,
)
from app.services.production_governed_audience_retrieval_service import (
    ProductionGovernedAudienceRetrievalService,
)
from app.services.production_multilingual_constraint_canonicalization_service import (
    ConstraintSemanticResolver,
    ProductionMultilingualConstraintCanonicalizationService,
    SemanticConstraintScore,
)


class _SharedPinnedSentenceTransformerRuntime:
    """Load each immutable model revision once for the complete benchmark run."""

    def __init__(self, *, device: str = "cpu") -> None:
        self._device = str(device or "cpu")
        self._models: dict[tuple[str, str, str], Any] = {}

    def encode(
        self,
        *,
        texts: Sequence[str],
        model: EmbeddingModelSpec,
        batch_size: int = 16,
    ) -> np.ndarray:
        if model.backend != "sentence_transformers":
            raise RuntimeError("Offline benchmark requires sentence_transformers.")
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
            list(texts),
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=model.normalize_embeddings,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise RuntimeError("Benchmark embeddings contain invalid values.")
        return matrix


class _SharedRuntimeDocumentEncoder:
    def __init__(self, runtime: _SharedPinnedSentenceTransformerRuntime) -> None:
        self._runtime = runtime

    def encode(
        self,
        *,
        texts: Sequence[str],
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray:
        matrix = self._runtime.encode(
            texts=[f"{model.document_prefix}{text}" for text in texts],
            model=model,
            batch_size=batch_size,
        )
        if matrix.ndim != 2 or matrix.shape != (len(texts), model.dimension):
            raise RuntimeError("Document embedding matrix has an invalid shape.")
        return matrix


class _SharedRuntimeQueryEncoder:
    def __init__(self, runtime: _SharedPinnedSentenceTransformerRuntime) -> None:
        self._runtime = runtime

    def encode(
        self,
        *,
        query_text: str,
        model: EmbeddingModelSpec,
    ) -> Sequence[float]:
        matrix = self._runtime.encode(
            texts=[f"{model.query_prefix}{query_text}"],
            model=model,
            batch_size=1,
        )
        if matrix.shape != (1, model.dimension):
            raise RuntimeError("Query embedding has an invalid shape.")
        vector = matrix[0]
        if np.linalg.norm(vector) <= 0:
            raise RuntimeError("Query embedding is invalid.")
        return vector.tolist()


class _SharedRuntimeSemanticResolver:
    _UNSPECIFIED_DESCRIPTIONS = {
        "category": (
            "no requested venue category",
            "general audience without a business category",
        ),
        "daypart": (
            "no requested time of day",
            "visits at any time without a temporal constraint",
        ),
    }

    def __init__(self, runtime: _SharedPinnedSentenceTransformerRuntime) -> None:
        self._runtime = runtime

    def score(
        self,
        *,
        query_text: str,
        dimension: str,
        entries: Sequence[ConstraintTaxonomyEntry],
        model: EmbeddingModelSpec,
    ) -> Sequence[SemanticConstraintScore]:
        if dimension == "location":
            raise RuntimeError("Semantic location mapping is disabled.")
        label_texts: list[str] = []
        ranges: dict[str, tuple[int, int]] = {}
        for entry in entries:
            start = len(label_texts)
            label_texts.extend(
                text
                for raw in (*entry.aliases, *entry.descriptions)
                if (text := " ".join(str(raw or "").split()))
            )
            ranges[entry.canonical_value] = (start, len(label_texts))
        unspecified_start = len(label_texts)
        label_texts.extend(self._UNSPECIFIED_DESCRIPTIONS[dimension])
        ranges["_unspecified"] = (unspecified_start, len(label_texts))
        if not label_texts:
            return []
        matrix = self._runtime.encode(
            texts=[f"{model.query_prefix}{query_text}"]
            + [f"{model.document_prefix}{text}" for text in label_texts],
            model=model,
            batch_size=32,
        )
        query_vector = matrix[0]
        label_vectors = matrix[1:]
        output: list[SemanticConstraintScore] = []
        for canonical_value, (start, end) in ranges.items():
            if end <= start:
                continue
            output.append(
                SemanticConstraintScore(
                    canonical_value=canonical_value,
                    score=float(np.max(label_vectors[start:end] @ query_vector)),
                )
            )
        return sorted(output, key=lambda item: item.score, reverse=True)


class BenchmarkDocumentEncoder(Protocol):
    def encode(
        self,
        *,
        texts: Sequence[str],
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray: ...


class OfflinePinnedModelApprovalGate:
    """Offline-only approval simulation for exact benchmarked model revisions."""

    def __init__(self, allowed_models: Sequence[EmbeddingModelSpec]) -> None:
        self._allowed = {model.fingerprint: model for model in allowed_models}

    def require_approved(
        self,
        *,
        tenant_id: str,
        model: EmbeddingModelSpec,
    ) -> dict[str, Any]:
        if model.fingerprint not in self._allowed:
            raise RuntimeError("Model is not allowed by the offline pinned-model gate.")
        return {
            "tenant_id": normalize_taxonomy_value(tenant_id),
            "model_fingerprint": model.fingerprint,
            "approved": True,
            "approval_scope": "offline_benchmark_only",
            "production_registration_performed": False,
        }


@dataclass
class _SearchCall:
    feature_set_id: str
    top_k: int
    document_ids: tuple[str, ...]


class _InMemoryBenchmarkFeatureStore:
    def __init__(
        self,
        *,
        tenant_id: str,
        execution_mode: str,
        documents: Sequence[Mapping[str, Any]],
        embeddings_by_feature_set: Mapping[str, np.ndarray],
        bindings: Sequence[ProductionRetrievalModelBinding],
    ) -> None:
        self._tenant_id = tenant_id
        self._execution_mode = execution_mode
        self._documents = [dict(value) for value in documents]
        self._embeddings = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in embeddings_by_feature_set.items()
        }
        self._bindings = {binding.feature_set_id: binding for binding in bindings}
        self.search_calls: list[_SearchCall] = []

    def reset_calls(self) -> None:
        self.search_calls.clear()

    def get_feature_set(
        self,
        *,
        tenant_id: str,
        feature_set_id: str | None = None,
        version: int | None = None,
        data_use_mode: str | None = None,
    ) -> dict[str, Any]:
        if tenant_id != self._tenant_id or feature_set_id not in self._bindings:
            raise RuntimeError("Offline benchmark feature set was not found.")
        binding = self._bindings[str(feature_set_id)]
        if version != binding.feature_set_version:
            raise RuntimeError("Offline benchmark feature-set version mismatch.")
        if data_use_mode != self._execution_mode:
            raise RuntimeError("Offline benchmark data-use mode mismatch.")
        return {
            "feature_set_id": binding.feature_set_id,
            "version": binding.feature_set_version,
            "model_backend": binding.model.backend,
            "model_name": binding.model.model_name,
            "model_version": binding.model.model_revision,
            "embedding_dimension": binding.model.dimension,
            "eligible_for_retrieval": True,
            "eligible_for_activation": False,
            "data_use_mode": self._execution_mode,
            "freshness_status": "fresh",
            "lineage": {
                "embedding_model_spec": binding.model.to_safe_dict(),
                "offline_benchmark_only": True,
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
            },
        }

    def hybrid_search(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
        query_text: str,
        query_embedding: Sequence[float],
        execution_mode: str,
        locations: Sequence[str],
        categories: Sequence[str],
        dayparts: Sequence[str],
        exclusions: Sequence[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        del query_text, locations, categories, dayparts
        if tenant_id != self._tenant_id or execution_mode != self._execution_mode:
            raise RuntimeError("Offline benchmark search context mismatch.")
        binding = self._bindings.get(feature_set_id)
        if binding is None or binding.feature_set_version != feature_set_version:
            raise RuntimeError("Offline benchmark search binding mismatch.")
        matrix = self._embeddings[feature_set_id]
        vector = np.asarray(query_embedding, dtype=np.float64)
        if vector.shape != (binding.model.dimension,):
            raise RuntimeError("Offline benchmark query vector has invalid shape.")
        scores = matrix @ vector
        excluded = {str(value) for value in exclusions}
        ranked = sorted(
            range(len(self._documents)),
            key=lambda index: (-float(scores[index]), self._documents[index]["document_id"]),
        )
        output: list[dict[str, Any]] = []
        for index in ranked:
            document = self._documents[index]
            if document["document_id"] in excluded:
                continue
            output.append(
                {
                    "feature_id": document["document_id"],
                    "location_name": document["location"],
                    "primary_poi_type": document["category"],
                    "created_day_part": document["daypart"],
                    "lookback_bucket": document.get("lookback_bucket") or "benchmark",
                    "cohort_size": int(document.get("cohort_size") or 1000),
                    "quality_score": _bounded_float(document.get("quality_score"), 0.5),
                    "privacy_status": "passed",
                    "rights_status": "offline_evaluation_only",
                    "purpose": "offline_governed_retrieval_benchmark",
                    "source_latest_at": "1970-01-01T00:00:00+00:00",
                    "freshness_status": "fresh",
                    "data_use_mode": self._execution_mode,
                    "vector_score": float(scores[index]),
                    "lexical_score": 0.0,
                    "fused_score": float(scores[index]),
                    "metadata": {
                        "canonical_feature_fingerprint": document["document_id"],
                        "offline_benchmark_only": True,
                    },
                }
            )
            if len(output) >= int(top_k):
                break
        self.search_calls.append(
            _SearchCall(
                feature_set_id=feature_set_id,
                top_k=int(top_k),
                document_ids=tuple(item["feature_id"] for item in output),
            )
        )
        return output


class ProductionGovernedRetrievalBenchmarkService:
    """Evaluate the real Module 2.1 canonicalizer and retrieval pipeline offline."""

    PRIMARY_MODEL = EmbeddingModelSpec(
        backend="sentence_transformers",
        model_name="intfloat/multilingual-e5-small",
        model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        dimension=384,
        normalize_embeddings=True,
        document_prefix="passage: ",
        query_prefix="query: ",
    )
    COMPLEMENTARY_MODEL = EmbeddingModelSpec(
        backend="sentence_transformers",
        model_name=(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        dimension=384,
        normalize_embeddings=True,
        document_prefix="",
        query_prefix="",
    )

    def __init__(
        self,
        *,
        policy: GovernedRetrievalBenchmarkPolicy | None = None,
        document_encoder: BenchmarkDocumentEncoder | None = None,
        query_encoder: ProductionQueryEncoder | None = None,
        semantic_resolver: ConstraintSemanticResolver | None = None,
        model_gate: Any | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        semantic_min_score: float = 0.55,
        semantic_min_margin: float = 0.05,
    ) -> None:
        self._policy = policy or GovernedRetrievalBenchmarkPolicy()
        shared_runtime = _SharedPinnedSentenceTransformerRuntime(device=device)
        self._document_encoder = document_encoder or _SharedRuntimeDocumentEncoder(
            shared_runtime
        )
        self._query_encoder = query_encoder or _SharedRuntimeQueryEncoder(
            shared_runtime
        )
        self._semantic_resolver = semantic_resolver or _SharedRuntimeSemanticResolver(
            shared_runtime
        )
        self._model_gate = model_gate or OfflinePinnedModelApprovalGate(
            (self.PRIMARY_MODEL, self.COMPLEMENTARY_MODEL)
        )
        self._batch_size = int(batch_size)
        self._semantic_min_score = float(semantic_min_score)
        self._semantic_min_margin = float(semantic_min_margin)
        if not 1 <= self._batch_size <= 4096:
            raise ValueError("batch_size must be between 1 and 4096.")

    def evaluate(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        identity, documents, cases = self._validate_dataset(dataset)
        taxonomy, taxonomy_source = self._build_taxonomy(dataset, documents)
        tenant_id = "offline_benchmark_tenant"
        execution_mode = "historical_preview"
        bindings = (
            ProductionRetrievalModelBinding(
                role="primary",
                feature_set_id="offline-benchmark-e5",
                feature_set_version=1,
                model=self.PRIMARY_MODEL,
                initial_depth=10,
                expansion_depth=30,
            ),
            ProductionRetrievalModelBinding(
                role="complementary",
                feature_set_id="offline-benchmark-minilm",
                feature_set_version=1,
                model=self.COMPLEMENTARY_MODEL,
                initial_depth=10,
                expansion_depth=10,
            ),
        )

        setup_started = time.perf_counter()
        texts = [document["text"] for document in documents]
        embeddings = {
            bindings[0].feature_set_id: self._document_encoder.encode(
                texts=texts,
                model=self.PRIMARY_MODEL,
                batch_size=self._batch_size,
            ),
            bindings[1].feature_set_id: self._document_encoder.encode(
                texts=texts,
                model=self.COMPLEMENTARY_MODEL,
                batch_size=self._batch_size,
            ),
        }
        feature_store = _InMemoryBenchmarkFeatureStore(
            tenant_id=tenant_id,
            execution_mode=execution_mode,
            documents=documents,
            embeddings_by_feature_set=embeddings,
            bindings=bindings,
        )
        canonicalizer = ProductionMultilingualConstraintCanonicalizationService(
            model_registry=self._model_gate,
            semantic_resolver=self._semantic_resolver,
            semantic_min_score=self._semantic_min_score,
            semantic_min_margin=self._semantic_min_margin,
        )
        candidate_retrieval = ProductionDualModelCandidateRetrievalService(
            feature_store=feature_store,
            model_registry=self._model_gate,
            query_encoder=self._query_encoder,
        )
        orchestrator = ProductionGovernedAudienceRetrievalService(
            canonicalizer=canonicalizer,
            candidate_retrieval=candidate_retrieval,
        )
        setup_ms = (time.perf_counter() - setup_started) * 1000.0

        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        language_counts: dict[str, Counter[str]] = defaultdict(Counter)
        initial_peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        for case in cases:
            feature_store.reset_calls()
            request = GovernedAudienceRetrievalRequest(
                tenant_id=tenant_id,
                query_text=case["query"],
                language=case["language"],
                execution_mode=execution_mode,
                taxonomy=taxonomy,
                canonicalizer_model=self.PRIMARY_MODEL,
                primary_model=bindings[0],
                complementary_model=bindings[1],
                requested_locations=(),
                requested_categories=(),
                requested_dayparts=(),
                exclusions=(),
                result_limit=10,
            )
            started = time.perf_counter()
            result = orchestrator.retrieve(request)
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)

            constraint_report = dict(result.get("constraints") or {})
            constraint_fields = dict(constraint_report.get("constraints") or {})
            locations = _resolution_values(constraint_fields.get("locations"))
            categories = _resolution_values(constraint_fields.get("categories"))
            dayparts = _resolution_values(constraint_fields.get("dayparts"))
            expected_locations = tuple(case["expected_locations"])
            expected_categories = tuple(case["expected_categories"])
            expected_dayparts = tuple(case["expected_dayparts"])
            location_correct = set(locations) == set(expected_locations)
            category_correct = set(categories) == set(expected_categories)
            daypart_correct = (
                set(dayparts) == set(expected_dayparts)
                if expected_dayparts
                else not dayparts
            )
            full_canonicalization_correct = (
                location_correct and category_correct and daypart_correct
            )

            call_groups = _group_search_calls(feature_store.search_calls, bindings)
            relevant_ids = set(case["relevant_document_ids"])
            base_ids = set(call_groups["base"])
            final_ids = set(call_groups["final"])
            base_candidate_hit = bool(base_ids.intersection(relevant_ids))
            final_candidate_hit = bool(final_ids.intersection(relevant_ids))

            selected = list(result.get("selected_candidates") or [])
            selected_top = selected[0] if selected else None
            selected_ids = set()
            if selected_top:
                selected_ids.update(
                    str(value)
                    for value in dict(selected_top.get("feature_ids") or {}).values()
                    if value
                )
            exact_top1 = bool(selected_ids.intersection(relevant_ids))
            semantic_top1 = bool(
                selected_top
                and selected_top.get("location_name") in expected_locations
                and selected_top.get("primary_poi_type") in expected_categories
                and (
                    not expected_dayparts
                    or selected_top.get("created_day_part") in expected_dayparts
                )
            )
            unsupported = bool(case["unsupported_location"])
            retrieval_ready = result.get("status") == (
                "retrieval_ready_for_human_review"
            )
            safe_unsupported_rejection = unsupported and not retrieval_ready
            clarification_expected = unsupported or not full_canonicalization_correct
            clarification_observed = bool(
                constraint_report.get("clarification_fields")
            )
            clarification_correct = (
                clarification_observed if clarification_expected else not clarification_observed
            )
            expanded = bool(
                dict(result.get("retrieval") or {}).get("expanded_primary")
            )

            language = case["language"]
            counters = language_counts[language]
            counters["case_count"] += 1
            counters["supported_count"] += int(not unsupported)
            counters["unsupported_count"] += int(unsupported)
            counters["full_canonicalization_correct"] += int(
                full_canonicalization_correct and not unsupported
            )
            counters["final_candidate_hit"] += int(
                final_candidate_hit and not unsupported
            )
            counters["semantic_top1"] += int(semantic_top1 and not unsupported)
            counters["unsupported_rejected"] += int(safe_unsupported_rejection)
            counters["case_success"] += int(
                safe_unsupported_rejection
                if unsupported
                else full_canonicalization_correct
                and final_candidate_hit
                and semantic_top1
            )

            rows.append(
                {
                    "case_id": case["case_id"],
                    "language": language,
                    "unsupported_location": unsupported,
                    "expected": {
                        "locations": list(expected_locations),
                        "categories": list(expected_categories),
                        "dayparts": list(expected_dayparts),
                        "relevant_document_ids": list(case["relevant_document_ids"]),
                    },
                    "observed": {
                        "locations": list(locations),
                        "categories": list(categories),
                        "dayparts": list(dayparts),
                        "constraint_reason_code": constraint_report.get("reason_code"),
                        "retrieval_status": result.get("status"),
                        "retrieval_reason_code": result.get("reason_code"),
                        "selected_document_ids": sorted(selected_ids),
                    },
                    "metrics": {
                        "location_correct": location_correct,
                        "category_correct": category_correct,
                        "daypart_correct": daypart_correct,
                        "full_canonicalization_correct": (
                            full_canonicalization_correct
                        ),
                        "base_candidate_hit": base_candidate_hit,
                        "final_candidate_hit": final_candidate_hit,
                        "exact_top1": exact_top1,
                        "semantic_top1": semantic_top1,
                        "safe_unsupported_rejection": (
                            safe_unsupported_rejection
                        ),
                        "clarification_correct": clarification_correct,
                        "expanded_primary": expanded,
                        "search_call_count": len(feature_store.search_calls),
                        "latency_ms": round(latency_ms, 6),
                    },
                }
            )

        summary = self._summarize(rows, latencies)
        per_language = self._language_summary(language_counts)
        checks = self._checks(summary, per_language)
        engineering_passed = all(checks.values())
        required_unsupported = _minimum_zero_failure_sample_count(
            target_rate=self._policy.target_unsupported_false_match_rate,
            confidence_level=self._policy.confidence_level,
        )
        benchmark_evidence_ready = (
            engineering_passed
            and identity.unsupported_case_count >= required_unsupported
            and bool(dataset.get("native_human_signoff_complete"))
        )
        process_peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report = {
            "contract_version": "production-governed-retrieval-benchmark-report-v1",
            "status": "benchmark_completed",
            "benchmark": {
                **identity.to_safe_dict(),
                "canonical_content_fingerprint": stable_digest(
                    {"documents": documents, "cases": cases}
                ),
            },
            "policy": self._policy.to_safe_dict(),
            "taxonomy": {
                "taxonomy_id": taxonomy.taxonomy_id,
                "taxonomy_version": taxonomy.version,
                "taxonomy_fingerprint": taxonomy.fingerprint,
                "taxonomy_source": taxonomy_source,
                "review_status": taxonomy.review_status,
            },
            "models": {
                "canonicalizer": self.PRIMARY_MODEL.to_safe_dict(),
                "primary_retrieval": self.PRIMARY_MODEL.to_safe_dict(),
                "complementary_retrieval": self.COMPLEMENTARY_MODEL.to_safe_dict(),
                "offline_gate_only": True,
                "production_registration_performed": False,
            },
            "summary": summary,
            "per_language": per_language,
            "checks": checks,
            "engineering_passed": engineering_passed,
            "benchmark_evidence_ready": benchmark_evidence_ready,
            "production_certification": {
                "ready": False,
                "native_human_signoff_complete": bool(
                    dataset.get("native_human_signoff_complete")
                ),
                "unsupported_case_count": identity.unsupported_case_count,
                "minimum_zero_failure_unsupported_cases": required_unsupported,
                "empirical_unsupported_rate_resolution": (
                    1.0 / identity.unsupported_case_count
                    if identity.unsupported_case_count
                    else None
                ),
                "reason_codes": _certification_reasons(
                    engineering_passed=engineering_passed,
                    native_human_signoff=bool(
                        dataset.get("native_human_signoff_complete")
                    ),
                    unsupported_count=identity.unsupported_case_count,
                    required_unsupported=required_unsupported,
                ),
            },
            "runtime": {
                "setup_ms": round(setup_ms, 6),
                "process_peak_memory_mb": round(process_peak_kb / 1024.0, 6),
                "process_peak_memory_delta_mb": round(
                    max(0.0, process_peak_kb - initial_peak_kb) / 1024.0,
                    6,
                ),
                "native_allocator_peak_is_process_level": True,
                "cost_measured": False,
                "cost_mode": "local_offline_engineering_evaluation",
            },
            "case_diagnostics": rows,
            "registration_allowed": False,
            "production_routing_enabled": False,
            "eligible_for_automatic_proposal": False,
            "threshold_changed": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
            "raw_query_text_stored": False,
            "document_text_stored": False,
            "embeddings_stored": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
        }
        report["report_fingerprint"] = stable_digest(report)
        return report

    def _validate_dataset(
        self,
        dataset: Mapping[str, Any],
    ) -> tuple[
        GovernedRetrievalBenchmarkDatasetIdentity,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        if not isinstance(dataset, Mapping):
            raise ValueError("Benchmark dataset must be a mapping.")
        _assert_no_identifier_keys(dataset)
        documents_raw = dataset.get("documents")
        cases_raw = dataset.get("cases")
        if not isinstance(documents_raw, Sequence) or isinstance(
            documents_raw, (str, bytes)
        ):
            raise ValueError("Benchmark documents must be a sequence.")
        if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, (str, bytes)):
            raise ValueError("Benchmark cases must be a sequence.")
        documents = [_canonical_document(value) for value in documents_raw]
        document_ids = [value["document_id"] for value in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Benchmark document IDs must be unique.")
        cases = [_canonical_case(value, set(document_ids)) for value in cases_raw]
        case_ids = [value["case_id"] for value in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case IDs must be unique.")
        unsupported_count = sum(bool(case["unsupported_location"]) for case in cases)
        canonical_dataset = EmbeddingBenchmarkDataset.from_mapping(dataset)
        canonical_fingerprint = canonical_dataset.fingerprint
        declared_fingerprint = str(
            dataset.get("dataset_fingerprint") or ""
        ).strip().lower()
        if (
            declared_fingerprint
            and declared_fingerprint != canonical_fingerprint
        ):
            raise ValueError(
                "Declared dataset_fingerprint does not match canonical "
                "benchmark content."
            )
        identity = GovernedRetrievalBenchmarkDatasetIdentity(
            benchmark_id=canonical_dataset.benchmark_id,
            dataset_version=canonical_dataset.dataset_version,
            dataset_fingerprint=canonical_fingerprint,
            document_count=len(documents),
            case_count=len(cases),
            supported_case_count=len(cases) - unsupported_count,
            unsupported_case_count=unsupported_count,
        )
        return identity, documents, cases

    def _build_taxonomy(
        self,
        dataset: Mapping[str, Any],
        documents: Sequence[Mapping[str, Any]],
    ) -> tuple[GovernedConstraintTaxonomy, str]:
        supplied = dataset.get("governed_constraint_taxonomy")
        if isinstance(supplied, Mapping):
            return _taxonomy_from_mapping(supplied), "dataset_governed_constraint_taxonomy"
        locations = sorted({str(value["location"]) for value in documents})
        categories = sorted({str(value["category"]) for value in documents})
        dayparts = sorted({str(value["daypart"]) for value in documents})
        return (
            GovernedConstraintTaxonomy(
                taxonomy_id="offline-benchmark-derived-canonical-taxonomy",
                version="v1",
                reviewed_by="benchmark_dataset_owner",
                review_status="approved",
                locations=tuple(
                    ConstraintTaxonomyEntry(
                        value,
                        aliases=(value.replace("_", " "),),
                        descriptions=(f"location {value.replace('_', ' ')}",),
                    )
                    for value in locations
                ),
                categories=tuple(
                    ConstraintTaxonomyEntry(
                        value,
                        aliases=(value.replace("_", " "),),
                        descriptions=(
                            f"people visiting {value.replace('_', ' ')}",
                            f"audience associated with {value.replace('_', ' ')} venues",
                        ),
                    )
                    for value in categories
                ),
                dayparts=tuple(
                    ConstraintTaxonomyEntry(
                        value,
                        aliases=(value.replace("_", " "),),
                        descriptions=(
                            f"visits during {value.replace('_', ' ')}",
                            f"activities in the {value.replace('_', ' ')} period",
                        ),
                    )
                    for value in dayparts
                ),
            ),
            "dataset_canonical_fields_without_multilingual_alias_oracle",
        )

    def _summarize(
        self,
        rows: Sequence[Mapping[str, Any]],
        latencies: Sequence[float],
    ) -> dict[str, Any]:
        supported = [row for row in rows if not row["unsupported_location"]]
        unsupported = [row for row in rows if row["unsupported_location"]]
        return {
            "case_count": len(rows),
            "supported_case_count": len(supported),
            "unsupported_case_count": len(unsupported),
            "location_accuracy": _rate(supported, "location_correct"),
            "category_accuracy": _rate(supported, "category_correct"),
            "daypart_accuracy": _rate(supported, "daypart_correct"),
            "full_canonicalization_accuracy": _rate(
                supported, "full_canonicalization_correct"
            ),
            "base_candidate_recall": _rate(supported, "base_candidate_hit"),
            "final_candidate_recall": _rate(supported, "final_candidate_hit"),
            "structured_exact_top1_accuracy": _rate(supported, "exact_top1"),
            "structured_semantic_top1_accuracy": _rate(
                supported, "semantic_top1"
            ),
            "unsupported_rejection_rate": _rate(
                unsupported, "safe_unsupported_rejection"
            ),
            "unsafe_unsupported_ready_rate": (
                1.0 - _rate(unsupported, "safe_unsupported_rejection")
                if unsupported
                else 0.0
            ),
            "clarification_accuracy": _rate(rows, "clarification_correct"),
            "expansion_case_count": sum(
                bool(row["metrics"]["expanded_primary"]) for row in rows
            ),
            "expansion_case_rate": (
                sum(bool(row["metrics"]["expanded_primary"]) for row in rows)
                / len(rows)
                if rows
                else 0.0
            ),
            "latency_ms": {
                "p50": round(_percentile(latencies, 50), 6),
                "p95": round(_percentile(latencies, 95), 6),
                "p99": round(_percentile(latencies, 99), 6),
                "mean": round(statistics.fmean(latencies), 6) if latencies else 0.0,
                "maximum": round(max(latencies), 6) if latencies else 0.0,
            },
        }

    def _language_summary(
        self,
        language_counts: Mapping[str, Counter[str]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for language, counts in sorted(language_counts.items()):
            case_count = counts["case_count"]
            supported = counts["supported_count"]
            unsupported = counts["unsupported_count"]
            output[language] = {
                "case_count": case_count,
                "supported_case_count": supported,
                "unsupported_case_count": unsupported,
                "full_canonicalization_accuracy": (
                    counts["full_canonicalization_correct"] / supported
                    if supported
                    else None
                ),
                "final_candidate_recall": (
                    counts["final_candidate_hit"] / supported if supported else None
                ),
                "structured_semantic_top1_accuracy": (
                    counts["semantic_top1"] / supported if supported else None
                ),
                "unsupported_rejection_rate": (
                    counts["unsupported_rejected"] / unsupported
                    if unsupported
                    else None
                ),
                "case_success_rate": counts["case_success"] / case_count,
            }
        return output

    def _checks(
        self,
        summary: Mapping[str, Any],
        per_language: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, bool]:
        latency = dict(summary["latency_ms"])
        return {
            "location_accuracy": summary["location_accuracy"]
            >= self._policy.min_location_accuracy,
            "category_accuracy": summary["category_accuracy"]
            >= self._policy.min_category_accuracy,
            "daypart_accuracy": summary["daypart_accuracy"]
            >= self._policy.min_daypart_accuracy,
            "full_canonicalization_accuracy": summary[
                "full_canonicalization_accuracy"
            ]
            >= self._policy.min_full_canonicalization_accuracy,
            "base_candidate_recall": summary["base_candidate_recall"]
            >= self._policy.min_base_candidate_recall,
            "final_candidate_recall": summary["final_candidate_recall"]
            >= self._policy.min_final_candidate_recall,
            "structured_semantic_top1_accuracy": summary[
                "structured_semantic_top1_accuracy"
            ]
            >= self._policy.min_structured_semantic_top1_accuracy,
            "unsupported_rejection_rate": summary["unsupported_rejection_rate"]
            >= self._policy.min_unsupported_rejection_rate,
            "clarification_accuracy": summary["clarification_accuracy"]
            >= self._policy.min_clarification_accuracy,
            "unsafe_unsupported_ready_rate": summary[
                "unsafe_unsupported_ready_rate"
            ]
            <= self._policy.max_unsafe_unsupported_ready_rate,
            "per_language_success_rate": all(
                float(metrics["case_success_rate"])
                >= self._policy.min_per_language_success_rate
                for metrics in per_language.values()
            ),
            "p95_latency": latency["p95"] <= self._policy.max_p95_latency_ms,
            "p99_latency": latency["p99"] <= self._policy.max_p99_latency_ms,
        }


def validate_governed_retrieval_benchmark_report(
    report: Mapping[str, Any],
) -> None:
    """Validate report integrity and side-effect/privacy invariants."""

    if report.get("contract_version") != (
        "production-governed-retrieval-benchmark-report-v1"
    ):
        raise ValueError("Unsupported governed retrieval benchmark report.")
    fingerprint = str(report.get("report_fingerprint") or "").strip().lower()
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if fingerprint != stable_digest(unsigned):
        raise ValueError("Governed retrieval benchmark report fingerprint mismatch.")
    required_false = (
        "registration_allowed",
        "production_routing_enabled",
        "eligible_for_automatic_proposal",
        "threshold_changed",
        "activation_or_export_performed",
        "downstream_export_enabled",
        "raw_query_text_stored",
        "document_text_stored",
        "embeddings_stored",
        "raw_identifiers_read",
        "raw_identifiers_stored",
    )
    for field_name in required_false:
        if report.get(field_name) is not False:
            raise ValueError(f"Unsafe benchmark report flag: {field_name}.")
    if dict(report.get("production_certification") or {}).get("ready") is not False:
        raise ValueError("Offline benchmark cannot grant production certification.")
    for case in report.get("case_diagnostics") or ():
        if not isinstance(case, Mapping):
            raise ValueError("Case diagnostics must be mappings.")
        forbidden = {
            "query",
            "query_text",
            "document_text",
            "embedding",
            "embeddings",
            "vector",
            "vectors",
        }
        overlap = forbidden.intersection(str(key) for key in case)
        if overlap:
            raise ValueError(
                "Benchmark case diagnostics contain forbidden payload fields: "
                + ", ".join(sorted(overlap))
            )



def _canonical_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Benchmark document must be a mapping.")
    required = {
        "document_id": str(value.get("document_id") or "").strip(),
        "text": " ".join(str(value.get("text") or "").split()),
        "location": normalize_taxonomy_value(value.get("location")),
        "category": normalize_taxonomy_value(value.get("category")),
        "daypart": normalize_taxonomy_value(value.get("daypart")),
    }
    if not all(required.values()):
        raise ValueError("Benchmark documents require ID, text, and canonical fields.")
    return {
        **required,
        "lookback_bucket": normalize_taxonomy_value(value.get("lookback_bucket"))
        or "benchmark",
        "cohort_size": int(value.get("cohort_size") or 1000),
        "quality_score": _bounded_float(value.get("quality_score"), 0.5),
    }


def _canonical_case(value: Any, document_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Benchmark case must be a mapping.")
    case_id = str(value.get("case_id") or "").strip()
    query = " ".join(str(value.get("query") or "").split())
    language = normalize_taxonomy_value(value.get("language") or "und") or "und"
    expected_locations = _canonical_values(value.get("expected_locations"))
    expected_categories = _canonical_values(value.get("expected_categories"))
    expected_dayparts = _canonical_values(value.get("expected_dayparts"))
    relevant = tuple(
        dict.fromkeys(
            str(item or "").strip()
            for item in (value.get("relevant_document_ids") or ())
            if str(item or "").strip()
        )
    )
    unsupported = bool(value.get("unsupported_location"))
    if not case_id or not query:
        raise ValueError("Benchmark cases require ID and query.")
    unknown = sorted(set(relevant).difference(document_ids))
    if unknown:
        raise ValueError("Benchmark case references unknown documents: " + ", ".join(unknown))
    if unsupported:
        if not expected_locations:
            raise ValueError("Unsupported cases require expected locations.")
        if relevant:
            raise ValueError("Unsupported cases cannot reference relevant documents.")
    elif not expected_locations or not expected_categories or not relevant:
        raise ValueError(
            "Supported cases require expected locations, categories, and "
            "relevant documents."
        )
    return {
        "case_id": case_id,
        "query": query,
        "language": language,
        "expected_locations": expected_locations,
        "expected_categories": expected_categories,
        "expected_dayparts": expected_dayparts,
        "relevant_document_ids": relevant,
        "unsupported_location": unsupported,
    }


def _taxonomy_from_mapping(value: Mapping[str, Any]) -> GovernedConstraintTaxonomy:
    def entries(key: str) -> tuple[ConstraintTaxonomyEntry, ...]:
        raw_entries = value.get(key)
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
            raise ValueError(f"Governed taxonomy {key} must be a sequence.")
        return tuple(
            ConstraintTaxonomyEntry(
                canonical_value=str(item.get("canonical_value") or ""),
                aliases=tuple(item.get("aliases") or ()),
                descriptions=tuple(item.get("descriptions") or ()),
            )
            for item in raw_entries
            if isinstance(item, Mapping)
        )

    return GovernedConstraintTaxonomy(
        taxonomy_id=str(value.get("taxonomy_id") or ""),
        version=str(value.get("version") or ""),
        reviewed_by=str(value.get("reviewed_by") or ""),
        review_status=str(value.get("review_status") or "approved"),
        locations=entries("locations"),
        categories=entries("categories"),
        dayparts=entries("dayparts"),
    )


def _group_search_calls(
    calls: Sequence[_SearchCall],
    bindings: Sequence[ProductionRetrievalModelBinding],
) -> dict[str, tuple[str, ...]]:
    primary_id = bindings[0].feature_set_id
    complementary_id = bindings[1].feature_set_id
    base: list[str] = []
    final: list[str] = []
    for call in calls:
        if call.feature_set_id == complementary_id or (
            call.feature_set_id == primary_id and call.top_k == 10
        ):
            base.extend(call.document_ids)
        final.extend(call.document_ids)
    return {
        "base": tuple(dict.fromkeys(base)),
        "final": tuple(dict.fromkeys(final)),
    }


def _resolution_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(
        normalize_taxonomy_value(item)
        for item in value.get("values", ())
        if normalize_taxonomy_value(item)
    )


def _rate(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    if not rows:
        return 1.0
    return sum(bool(row["metrics"][metric]) for row in rows) / len(rows)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _minimum_zero_failure_sample_count(*, target_rate: float, confidence_level: float) -> int:
    return int(math.ceil(math.log(1.0 - confidence_level) / math.log(1.0 - target_rate)))


def _certification_reasons(
    *,
    engineering_passed: bool,
    native_human_signoff: bool,
    unsupported_count: int,
    required_unsupported: int,
) -> list[str]:
    reasons: list[str] = []
    if not engineering_passed:
        reasons.append("engineering_policy_not_passed")
    if not native_human_signoff:
        reasons.append("native_human_signoff_pending")
    if unsupported_count < required_unsupported:
        reasons.append("unsupported_calibration_sample_insufficient")
    reasons.append("model_registration_and_external_release_gates_pending")
    return reasons


def _canonical_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    return tuple(
        dict.fromkeys(
            normalized
            for item in value
            if (normalized := normalize_taxonomy_value(item))
        )
    )


def _bounded_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return max(0.0, min(parsed, 1.0))


def _assert_no_identifier_keys(value: Any, path: str = "root") -> None:
    blocked = {
        "maid",
        "maids",
        "device_id",
        "device_ids",
        "email",
        "phone",
        "latitude",
        "longitude",
        "raw_identifier",
        "raw_identifiers",
        "raw_observation",
        "raw_observations",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = normalize_taxonomy_value(key)
            if normalized in blocked:
                raise ValueError(f"Raw identifier field is forbidden at {path}.{key}.")
            _assert_no_identifier_keys(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_no_identifier_keys(child, f"{path}[{index}]")
