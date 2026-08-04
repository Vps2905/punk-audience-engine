from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from app.models.audience_feature_contracts import stable_digest
from app.models.production_audience_retrieval_contracts import (
    GovernedAudienceRetrievalRequest,
    MultilingualConstraintResolution,
    ProductionRetrievalModelBinding,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec


class ProductionCandidateFeatureStore(Protocol):
    def get_feature_set(
        self,
        *,
        tenant_id: str,
        feature_set_id: str | None = None,
        version: int | None = None,
        data_use_mode: str | None = None,
    ) -> dict[str, Any]: ...

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
    ) -> list[dict[str, Any]]: ...


class EmbeddingModelApprovalGate(Protocol):
    def require_approved(
        self,
        *,
        tenant_id: str,
        model: EmbeddingModelSpec,
    ) -> dict[str, Any]: ...


class ProductionQueryEncoder(Protocol):
    def encode(
        self,
        *,
        query_text: str,
        model: EmbeddingModelSpec,
    ) -> Sequence[float]: ...


class ApprovedSentenceTransformerQueryEncoder:
    """Pinned query encoding with no hashing or unversioned fallback."""

    def __init__(self, *, device: str = "cpu") -> None:
        self._device = str(device or "cpu")
        self._models: dict[tuple[str, str, str], Any] = {}

    def encode(
        self,
        *,
        query_text: str,
        model: EmbeddingModelSpec,
    ) -> Sequence[float]:
        if model.backend != "sentence_transformers":
            raise RuntimeError(
                "Governed candidate retrieval requires sentence_transformers."
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
        vector = loaded.encode(
            [f"{model.query_prefix}{query_text}"],
            convert_to_numpy=True,
            normalize_embeddings=model.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != model.dimension:
            raise RuntimeError(
                "Query embedding dimension does not match the approved model."
            )
        if not np.isfinite(array).all() or np.linalg.norm(array) <= 0:
            raise RuntimeError("Query embedding is invalid.")
        return array.tolist()


class ProductionDualModelCandidateRetrievalService:
    """
    Approved dual-model candidate generation with rank-only cross-model fusion.

    Raw cosine scores are retained per model as diagnostics but are never
    averaged, compared, or used as a cross-model ordering signal.
    """

    RRF_CONSTANT = 60.0
    PRIMARY_MODEL_NAME = "intfloat/multilingual-e5-small"
    PRIMARY_MODEL_REVISION = (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    )
    COMPLEMENTARY_MODEL_NAME = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    COMPLEMENTARY_MODEL_REVISION = (
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    )

    def __init__(
        self,
        *,
        feature_store: ProductionCandidateFeatureStore,
        model_registry: EmbeddingModelApprovalGate,
        query_encoder: ProductionQueryEncoder | None = None,
    ) -> None:
        self._feature_store = feature_store
        self._model_registry = model_registry
        self._query_encoder = (
            query_encoder or ApprovedSentenceTransformerQueryEncoder()
        )

    def retrieve(
        self,
        *,
        request: GovernedAudienceRetrievalRequest,
        constraints: MultilingualConstraintResolution,
    ) -> dict[str, Any]:
        if constraints.tenant_id != request.tenant_id:
            raise ValueError("Constraint resolution tenant does not match request.")
        if constraints.query_fingerprint != request.query_fingerprint:
            raise ValueError("Constraint resolution does not match the query.")
        if constraints.taxonomy_fingerprint != request.taxonomy.fingerprint:
            raise ValueError("Constraint taxonomy fingerprint does not match request.")
        if not constraints.ready_for_retrieval:
            return self._blocked(
                request=request,
                constraints=constraints,
                reason_code=constraints.reason_code,
                explanation=(
                    "Canonical constraints require clarification before candidate "
                    "retrieval."
                ),
            )

        bindings = (
            request.primary_model,
            request.complementary_model,
        )
        self._validate_governed_model_bundle(bindings)
        feature_sets: dict[str, dict[str, Any]] = {}
        query_vectors: dict[str, Sequence[float]] = {}
        for binding in bindings:
            self._model_registry.require_approved(
                tenant_id=request.tenant_id,
                model=binding.model,
            )
            feature_set = self._feature_store.get_feature_set(
                tenant_id=request.tenant_id,
                feature_set_id=binding.feature_set_id,
                version=binding.feature_set_version,
                data_use_mode=request.execution_mode,
            )
            self._validate_feature_set(
                feature_set=feature_set,
                binding=binding,
                execution_mode=request.execution_mode,
            )
            feature_sets[binding.role] = feature_set
            query_vectors[binding.role] = self._query_encoder.encode(
                query_text=request.query_text,
                model=binding.model,
            )

        pool: dict[str, dict[str, Any]] = {}
        for binding in bindings:
            candidates = self._search(
                request=request,
                binding=binding,
                query_vector=query_vectors[binding.role],
                depth=binding.initial_depth,
            )
            self._merge_model_candidates(
                pool=pool,
                candidates=candidates,
                model_role=binding.role,
            )

        initial_ranked = self._rerank(pool.values(), constraints)
        expanded = not any(
            candidate["full_constraint_match"]
            for candidate in initial_ranked
        )
        if expanded:
            primary = request.primary_model
            expanded_candidates = self._search(
                request=request,
                binding=primary,
                query_vector=query_vectors[primary.role],
                depth=primary.expansion_depth,
            )
            self._merge_model_candidates(
                pool=pool,
                candidates=expanded_candidates,
                model_role=primary.role,
                replace_role_ranks=True,
            )

        ranked = self._rerank(pool.values(), constraints)
        eligible = [
            candidate
            for candidate in ranked
            if candidate["full_constraint_match"]
        ]
        if not eligible:
            return self._blocked(
                request=request,
                constraints=constraints,
                reason_code="no_candidate_satisfies_resolved_constraints",
                explanation=(
                    "No privacy-safe candidate satisfied every resolved "
                    "location, category, and requested daypart constraint after "
                    "the governed expansion step."
                ),
                expanded=expanded,
                candidate_pool_size=len(ranked),
                model_evidence=self._model_evidence(feature_sets, bindings),
            )

        selected = eligible[: request.result_limit]
        return {
            "contract_version": "production-governed-candidate-retrieval-v1",
            "status": "retrieval_ready_for_human_review",
            "reason_code": "verified_full_constraint_match_available",
            "tenant_id": request.tenant_id,
            "query_fingerprint": request.query_fingerprint,
            "language": request.language,
            "execution_mode": request.execution_mode,
            "constraints": constraints.to_safe_dict(),
            "retrieval": {
                "pipeline": (
                    "approved_dual_model_rrf_then_structured_constraint_rerank"
                ),
                "initial_depth": {
                    "primary": request.primary_model.initial_depth,
                    "complementary": (
                        request.complementary_model.initial_depth
                    ),
                },
                "primary_expansion_depth": (
                    request.primary_model.expansion_depth
                ),
                "expanded_primary": expanded,
                "candidate_pool_size": len(ranked),
                "full_match_count": len(eligible),
                "raw_cross_model_cosine_combined": False,
                "rrf_constant": self.RRF_CONSTANT,
                "structured_constraints_rank_before_rrf": True,
            },
            "models": self._model_evidence(feature_sets, bindings),
            "selected_candidates": selected,
            "eligible_for_automatic_proposal": False,
            "human_review_required": True,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
        }

    def _search(
        self,
        *,
        request: GovernedAudienceRetrievalRequest,
        binding: ProductionRetrievalModelBinding,
        query_vector: Sequence[float],
        depth: int,
    ) -> list[dict[str, Any]]:
        return self._feature_store.hybrid_search(
            tenant_id=request.tenant_id,
            feature_set_id=binding.feature_set_id,
            feature_set_version=binding.feature_set_version,
            query_text=request.query_text,
            query_embedding=query_vector,
            execution_mode=request.execution_mode,
            locations=(),
            categories=(),
            dayparts=(),
            exclusions=request.exclusions,
            top_k=depth,
        )

    def _validate_feature_set(
        self,
        *,
        feature_set: Mapping[str, Any],
        binding: ProductionRetrievalModelBinding,
        execution_mode: str,
    ) -> None:
        expected = {
            "feature_set_id": binding.feature_set_id,
            "version": binding.feature_set_version,
            "model_backend": binding.model.backend,
            "model_name": binding.model.model_name,
            "model_version": binding.model.model_revision,
            "embedding_dimension": binding.model.dimension,
        }
        for key, value in expected.items():
            if feature_set.get(key) != value:
                raise RuntimeError(
                    f"Feature set conflicts with approved model binding: {key}."
                )
        if not feature_set.get("eligible_for_retrieval"):
            raise RuntimeError("Feature set is not eligible for retrieval.")
        if execution_mode == "production":
            if feature_set.get("data_use_mode") != "production":
                raise RuntimeError(
                    "Production retrieval cannot use a non-production feature set."
                )
            if feature_set.get("freshness_status") != "fresh":
                raise RuntimeError(
                    "Production retrieval requires a fresh feature set."
                )
        lineage = dict(feature_set.get("lineage") or {})
        stored_spec = dict(lineage.get("embedding_model_spec") or {})
        safe_spec = binding.model.to_safe_dict()
        for key in (
            "backend",
            "model_name",
            "model_revision",
            "dimension",
            "normalize_embeddings",
            "document_prefix",
            "query_prefix",
            "model_fingerprint",
        ):
            if stored_spec.get(key) != safe_spec.get(key):
                raise RuntimeError(
                    "Feature-set lineage does not match the approved model spec."
                )

    def _merge_model_candidates(
        self,
        *,
        pool: dict[str, dict[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        model_role: str,
        replace_role_ranks: bool = False,
    ) -> None:
        if replace_role_ranks:
            for candidate in pool.values():
                candidate["source_ranks"].pop(model_role, None)
                candidate["model_scores"].pop(model_role, None)
        for position, source in enumerate(candidates, start=1):
            identity = self._candidate_identity(source)
            existing = pool.get(identity)
            if existing is not None:
                self._validate_candidate_identity_collision(existing, source)
            candidate = pool.setdefault(
                identity,
                {
                    "candidate_identity": identity,
                    "feature_ids": {},
                    "location_name": source.get("location_name"),
                    "primary_poi_type": source.get("primary_poi_type"),
                    "created_day_part": source.get("created_day_part"),
                    "lookback_bucket": source.get("lookback_bucket"),
                    "cohort_size": int(source.get("cohort_size") or 0),
                    "quality_score": self._bounded_float(
                        source.get("quality_score")
                    ),
                    "privacy_status": source.get("privacy_status"),
                    "rights_status": source.get("rights_status"),
                    "freshness_status": source.get("freshness_status"),
                    "data_use_mode": source.get("data_use_mode"),
                    "source_ranks": {},
                    "model_scores": {},
                },
            )
            candidate["feature_ids"][model_role] = source.get("feature_id")
            candidate["source_ranks"][model_role] = position
            candidate["model_scores"][model_role] = {
                "vector_score": self._optional_float(source.get("vector_score")),
                "lexical_score": self._optional_float(source.get("lexical_score")),
                "within_model_fused_score": self._optional_float(
                    source.get("fused_score")
                ),
            }

        empty = [
            identity
            for identity, candidate in pool.items()
            if not candidate["source_ranks"]
        ]
        for identity in empty:
            pool.pop(identity, None)

    def _rerank(
        self,
        candidates: Sequence[Mapping[str, Any]],
        constraints: MultilingualConstraintResolution,
    ) -> list[dict[str, Any]]:
        expected = constraints.resolved_values()
        requested_dimensions = {
            "location": bool(expected["locations"]),
            "category": bool(expected["categories"]),
            "daypart": bool(expected["dayparts"]),
        }
        output: list[dict[str, Any]] = []
        for raw in candidates:
            candidate = dict(raw)
            location_match = (
                not requested_dimensions["location"]
                or candidate.get("location_name") in expected["locations"]
            )
            category_match = (
                not requested_dimensions["category"]
                or candidate.get("primary_poi_type") in expected["categories"]
            )
            daypart_match = (
                not requested_dimensions["daypart"]
                or candidate.get("created_day_part") in expected["dayparts"]
            )
            match_count = sum(
                (
                    location_match and requested_dimensions["location"],
                    category_match and requested_dimensions["category"],
                    daypart_match and requested_dimensions["daypart"],
                )
            )
            required_count = sum(requested_dimensions.values())
            full_match = (
                required_count > 0
                and match_count == required_count
            )
            source_ranks = dict(candidate.get("source_ranks") or {})
            rrf_score = sum(
                1.0 / (self.RRF_CONSTANT + int(rank))
                for rank in source_ranks.values()
            )
            candidate.update(
                {
                    "constraint_matches": {
                        "location": location_match,
                        "category": category_match,
                        "daypart": daypart_match,
                    },
                    "resolved_constraint_count": required_count,
                    "constraint_match_count": match_count,
                    "full_constraint_match": full_match,
                    "model_agreement_count": len(source_ranks),
                    "rrf_score": round(rrf_score, 10),
                    "raw_cross_model_cosine_combined": False,
                }
            )
            output.append(candidate)
        output.sort(
            key=lambda candidate: (
                -int(candidate["full_constraint_match"]),
                -candidate["constraint_match_count"],
                -candidate["model_agreement_count"],
                -candidate["rrf_score"],
                -candidate["quality_score"],
                candidate["candidate_identity"],
            )
        )
        for rank, candidate in enumerate(output, start=1):
            candidate["rank"] = rank
        return output

    def _candidate_identity(self, source: Mapping[str, Any]) -> str:
        metadata = dict(source.get("metadata") or {})
        for key in (
            "canonical_feature_fingerprint",
            "canonical_row_fingerprint",
            "source_feature_fingerprint",
            "canonical_content_fingerprint",
        ):
            value = str(metadata.get(key) or "").strip().lower()
            if value:
                return f"{key}:{value}"
        return "canonical_candidate_" + stable_digest(
            {
                "location_name": source.get("location_name"),
                "primary_poi_type": source.get("primary_poi_type"),
                "created_day_part": source.get("created_day_part"),
                "lookback_bucket": source.get("lookback_bucket"),
                "cohort_size": int(source.get("cohort_size") or 0),
                "quality_score": self._bounded_float(
                    source.get("quality_score")
                ),
                "purpose": source.get("purpose"),
                "source_latest_at": source.get("source_latest_at"),
                "metadata": metadata,
            }
        )[:32]

    def _validate_governed_model_bundle(
        self,
        bindings: Sequence[ProductionRetrievalModelBinding],
    ) -> None:
        by_role = {binding.role: binding for binding in bindings}
        primary = by_role["primary"].model
        complementary = by_role["complementary"].model
        if (
            primary.model_name != self.PRIMARY_MODEL_NAME
            or primary.model_revision != self.PRIMARY_MODEL_REVISION
            or primary.document_prefix != "passage: "
            or primary.query_prefix != "query: "
            or primary.dimension != 384
        ):
            raise RuntimeError(
                "The primary retrieval model does not match the governed "
                "Module 2.1 E5 revision and prefixes."
            )
        if (
            complementary.model_name != self.COMPLEMENTARY_MODEL_NAME
            or complementary.model_revision
            != self.COMPLEMENTARY_MODEL_REVISION
            or complementary.document_prefix != ""
            or complementary.query_prefix != ""
            or complementary.dimension != 384
        ):
            raise RuntimeError(
                "The complementary retrieval model does not match the "
                "governed Module 2.1 MiniLM revision."
            )

    def _validate_candidate_identity_collision(
        self,
        existing: Mapping[str, Any],
        source: Mapping[str, Any],
    ) -> None:
        expected = {
            "location_name": source.get("location_name"),
            "primary_poi_type": source.get("primary_poi_type"),
            "created_day_part": source.get("created_day_part"),
            "lookback_bucket": source.get("lookback_bucket"),
            "cohort_size": int(source.get("cohort_size") or 0),
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    "Candidate identity collision has conflicting canonical fields."
                )

    def _model_evidence(
        self,
        feature_sets: Mapping[str, Mapping[str, Any]],
        bindings: Sequence[ProductionRetrievalModelBinding],
    ) -> list[dict[str, Any]]:
        output = []
        for binding in bindings:
            feature_set = feature_sets.get(binding.role, {})
            output.append(
                {
                    "role": binding.role,
                    "feature_set_id": binding.feature_set_id,
                    "feature_set_version": binding.feature_set_version,
                    "model_fingerprint": binding.model.fingerprint,
                    "model_name": binding.model.model_name,
                    "model_revision": binding.model.model_revision,
                    "registry_approval_required": True,
                    "feature_set_freshness": feature_set.get(
                        "freshness_status"
                    ),
                }
            )
        return output

    def _blocked(
        self,
        *,
        request: GovernedAudienceRetrievalRequest,
        constraints: MultilingualConstraintResolution,
        reason_code: str,
        explanation: str,
        expanded: bool = False,
        candidate_pool_size: int = 0,
        model_evidence: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return {
            "contract_version": "production-governed-candidate-retrieval-v1",
            "status": "blocked",
            "reason_code": reason_code,
            "tenant_id": request.tenant_id,
            "query_fingerprint": request.query_fingerprint,
            "language": request.language,
            "execution_mode": request.execution_mode,
            "constraints": constraints.to_safe_dict(),
            "retrieval": {
                "pipeline": (
                    "approved_dual_model_rrf_then_structured_constraint_rerank"
                ),
                "expanded_primary": expanded,
                "candidate_pool_size": candidate_pool_size,
                "raw_cross_model_cosine_combined": False,
                "structured_constraints_rank_before_rrf": True,
            },
            "models": [dict(value) for value in model_evidence],
            "selected_candidates": [],
            "eligible_for_automatic_proposal": False,
            "human_review_required": True,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "explanation": explanation,
        }

    def _bounded_float(self, value: Any) -> float:
        parsed = self._optional_float(value)
        if parsed is None:
            return 0.0
        return max(0.0, min(parsed, 1.0))

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed
