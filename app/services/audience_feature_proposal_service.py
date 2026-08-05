from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.services.audience_proposal_request_safety_service import (
    AudienceProposalRequestSafetyService,
)
from app.services.embedding_service import hashing_encode


class AudienceFeatureStore(Protocol):
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


class FeatureQueryEmbeddingService:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], Any] = {}

    def encode(
        self,
        query: str,
        feature_set: dict[str, Any],
    ) -> np.ndarray:
        backend = str(feature_set.get("model_backend") or "").strip().lower()
        dimension = int(feature_set.get("embedding_dimension") or 0)
        if not query.strip():
            raise ValueError("audience_intent cannot be empty.")
        if dimension <= 0:
            raise ValueError("Feature-set embedding dimension is invalid.")

        if backend in {"sklearn_hashing", "hashing"}:
            return np.asarray(
                hashing_encode([query], n_features=dimension)["vectors"][0],
                dtype=float,
            )

        if backend in {"sentence-transformers", "sentence_transformers"}:
            from sentence_transformers import SentenceTransformer

            model_name = str(feature_set.get("model_name") or "").strip()
            model_revision = str(
                feature_set.get("model_version") or ""
            ).strip()
            if not model_name:
                raise ValueError(
                    "Sentence-transformer feature set is missing model_name."
                )
            if (
                not model_revision
                or model_revision.lower() in {
                    "latest",
                    "main",
                    "master",
                }
            ):
                raise ValueError(
                    "Sentence-transformer feature set is missing an "
                    "immutable model revision."
                )
            key = (model_name, model_revision)
            model = self._models.get(key)
            if model is None:
                model = SentenceTransformer(
                    model_name,
                    revision=model_revision,
                    trust_remote_code=False,
                )
                self._models[key] = model
            lineage = dict(feature_set.get("lineage") or {})
            model_spec = dict(
                lineage.get("embedding_model_spec") or {}
            )
            query_prefix = str(
                model_spec.get("query_prefix") or ""
            )
            normalize_embeddings = bool(
                model_spec.get("normalize_embeddings", True)
            )
            vector = model.encode(
                [f"{query_prefix}{query}"],
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            )[0]
            if int(vector.shape[0]) != dimension:
                raise ValueError(
                    "Query embedding dimension does not match feature set."
                )
            return np.asarray(vector, dtype=float)

        raise RuntimeError(
            "Feature-set embedding backend is not available for query encoding."
        )


class AudienceFeatureProposalService:
    """
    Punk AI -> Audience Intelligence proposal boundary.

    Structured fields are hard eligibility filters. Semantic similarity never
    overrides tenant, privacy, rights, freshness, data-mode, or approval gates.
    """

    CONTRACT_VERSION = "2026-07-25"

    def __init__(
        self,
        *,
        feature_store: AudienceFeatureStore,
        query_embedding_service: FeatureQueryEmbeddingService | None = None,
        request_safety_service: (
            AudienceProposalRequestSafetyService | None
        ) = None,
    ) -> None:
        self._feature_store = feature_store
        self._query_embedding_service = (
            query_embedding_service or FeatureQueryEmbeddingService()
        )
        self._request_safety_service = (
            request_safety_service
            or AudienceProposalRequestSafetyService()
        )

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:
        tenant_id = self._required_slug(request.get("tenant_id"), "tenant_id")
        campaign_id = self._required_text(
            request.get("campaign_id"),
            "campaign_id",
        )
        audience_intent = self._required_text(
            request.get("audience_intent"),
            "audience_intent",
        )
        idempotency_key = self._required_text(
            request.get("idempotency_key"),
            "idempotency_key",
        )
        execution_mode = normalize_taxonomy_value(
            request.get("execution_mode") or "historical_preview"
        )
        if execution_mode not in {"historical_preview", "production"}:
            raise ValueError(
                "execution_mode must be historical_preview or production."
            )

        feature_set = self._feature_store.get_feature_set(
            tenant_id=tenant_id,
            feature_set_id=request.get("feature_set_id"),
            version=request.get("feature_set_version"),
            data_use_mode=execution_mode,
        )
        feature_set_id = str(feature_set["feature_set_id"])
        feature_set_version = int(feature_set["version"])
        proposal_id = self._proposal_id(
            tenant_id=tenant_id,
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
            feature_set_id=feature_set_id,
            feature_set_version=feature_set_version,
        )

        safety_decision = self._request_safety_service.evaluate(request)
        if safety_decision.terminal:
            return self._blocked_response(
                request=request,
                feature_set=feature_set,
                proposal_id=proposal_id,
                execution_mode=execution_mode,
                reason_code=str(safety_decision.reason_code),
                explanation=str(safety_decision.explanation),
                audience_request_detected=False,
            )

        hard_block = self._feature_set_block(
            feature_set=feature_set,
            execution_mode=execution_mode,
        )
        if hard_block:
            return self._blocked_response(
                request=request,
                feature_set=feature_set,
                proposal_id=proposal_id,
                execution_mode=execution_mode,
                reason_code=hard_block["reason_code"],
                explanation=hard_block["explanation"],
            )

        structured_filters: dict[str, list[str]] = {}
        invalid_structured_fields: list[str] = []
        for field_name in (
            "locations",
            "categories",
            "dayparts",
            "exclusions",
        ):
            normalized, invalid = self._clean_structured_list(
                request.get(field_name)
            )
            structured_filters[field_name] = normalized
            if invalid:
                invalid_structured_fields.append(field_name)

        if invalid_structured_fields:
            return self._blocked_response(
                request=request,
                feature_set=feature_set,
                proposal_id=proposal_id,
                execution_mode=execution_mode,
                reason_code="blocked_unresolved_structured_filter",
                explanation=(
                    "One or more structured audience filters could not be "
                    "converted into governed canonical values. Clarification "
                    "or multilingual canonicalization is required before "
                    "retrieval."
                ),
            )

        query_embedding = self._query_embedding_service.encode(
            audience_intent,
            feature_set,
        )
        locations = structured_filters["locations"]
        categories = structured_filters["categories"]
        dayparts = structured_filters["dayparts"]
        exclusions = structured_filters["exclusions"]
        top_k = max(1, min(int(request.get("top_k") or 10), 50))

        candidates = self._feature_store.hybrid_search(
            tenant_id=tenant_id,
            feature_set_id=feature_set_id,
            feature_set_version=feature_set_version,
            query_text=audience_intent,
            query_embedding=query_embedding,
            execution_mode=execution_mode,
            locations=locations,
            categories=categories,
            dayparts=dayparts,
            exclusions=exclusions,
            top_k=top_k,
        )
        ranked_candidates = self._ranked_candidates(
            candidates,
            execution_mode=execution_mode,
        )

        if not ranked_candidates:
            return self._blocked_response(
                request=request,
                feature_set=feature_set,
                proposal_id=proposal_id,
                execution_mode=execution_mode,
                reason_code="blocked_no_safe_exact_match",
                explanation=(
                    "No privacy-safe feature passed the requested tenant, "
                    "location, category, daypart, exclusion, and data-mode filters."
                ),
            )

        historical = execution_mode == "historical_preview"
        activation_eligible = bool(
            not historical
            and feature_set.get("eligible_for_activation")
            and feature_set.get("freshness_status") == "fresh"
        )
        approval_status = (
            "blocked_historical_source"
            if historical
            else "pending_manual_approval"
        )
        status = (
            "historical_preview_ready"
            if historical
            else "proposal_ready_for_manual_approval"
        )
        coverage_warnings = []
        if historical:
            coverage_warnings.append(
                "Historical source data is allowed for internal evaluation only. "
                "Activation and downstream export remain blocked."
            )

        return {
            "contract_version": self.CONTRACT_VERSION,
            "proposal_id": proposal_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "campaign_id": campaign_id,
            "status": status,
            "execution_mode": execution_mode,
            "audience_request_detected": True,
            "eligible_for_audience_selection": True,
            "normalized_request": {
                "objective": request.get("objective"),
                "audience_intent": audience_intent,
                "locations": locations,
                "categories": categories,
                "dayparts": dayparts,
                "exclusions": exclusions,
                "destination": request.get("destination"),
                "budget": request.get("budget") or {},
            },
            "feature_set": self._feature_set_summary(feature_set),
            "retrieval": {
                "pipeline": "structured_filters_then_lexical_pgvector_rrf",
                "candidate_count": len(ranked_candidates),
                "top_k": top_k,
                "structured_filters_applied_before_ranking": True,
                "semantic_similarity_can_override_safety": False,
                "confidence_calibration": "provisional_offline_evidence",
            },
            "candidate_cohorts": ranked_candidates,
            "quality_and_confidence": {
                "quality_definition": (
                    "Stored privacy-safe cohort quality from the source feature set."
                ),
                "confidence_definition": (
                    "Offline retrieval evidence from vector similarity, lexical/vector "
                    "agreement, rank fusion, score margin, and stored quality."
                ),
                "campaign_lift_calibrated": False,
            },
            "coverage_warnings": coverage_warnings,
            "approval_required": True,
            "approval_status": approval_status,
            "activation_eligible": activation_eligible,
            "safe_export_eligible": activation_eligible,
            "downstream_export_enabled": False,
            "explanation": (
                "The candidates are suitable for an internal historical-data "
                "review. They cannot be activated or exported."
                if historical
                else (
                    "The candidates passed retrieval eligibility. Human approval "
                    "and the separate safe-activation workflow are still required."
                )
            ),
        }

    def _feature_set_block(
        self,
        *,
        feature_set: dict[str, Any],
        execution_mode: str,
    ) -> dict[str, str] | None:
        if not feature_set.get("eligible_for_retrieval"):
            return {
                "reason_code": "blocked_feature_set_not_retrievable",
                "explanation": (
                    "The feature set is not eligible for audience retrieval."
                ),
            }
        if execution_mode == "production":
            if feature_set.get("data_use_mode") != "production":
                return {
                    "reason_code": "blocked_historical_source",
                    "explanation": (
                        "Historical or offline feature sets cannot be used for "
                        "production audience activation."
                    ),
                }
            if feature_set.get("freshness_status") != "fresh":
                return {
                    "reason_code": "blocked_stale_source",
                    "explanation": (
                        "The source feature set is stale or its freshness is unknown."
                    ),
                }
            if not feature_set.get("eligible_for_activation"):
                return {
                    "reason_code": "blocked_feature_set_not_activation_eligible",
                    "explanation": (
                        "The feature set has not passed production activation eligibility."
                    ),
                }
        return None

    def _ranked_candidates(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        execution_mode: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        vector_scores = [
            max(-1.0, min(float(item.get("vector_score") or 0.0), 1.0))
            for item in candidates
        ]
        for index, item in enumerate(candidates):
            vector_score = vector_scores[index]
            normalized_vector = (vector_score + 1.0) / 2.0
            fused_score = max(float(item.get("fused_score") or 0.0), 0.0)
            normalized_fusion = min(fused_score / (2.0 / 61.0), 1.0)
            agreement = 1.0 if item.get("lexical_rank") is not None else 0.0
            quality = max(
                0.0,
                min(float(item.get("quality_score") or 0.0), 1.0),
            )
            next_vector = (
                vector_scores[index + 1]
                if index + 1 < len(vector_scores)
                else -1.0
            )
            margin = max(0.0, min(vector_score - next_vector, 1.0))
            evidence_confidence = (
                normalized_vector * 0.45
                + normalized_fusion * 0.20
                + agreement * 0.15
                + quality * 0.10
                + margin * 0.10
            )
            output.append(
                {
                    "rank": index + 1,
                    "feature_id": item.get("feature_id"),
                    "location_name": item.get("location_name"),
                    "primary_poi_type": item.get("primary_poi_type"),
                    "created_day_part": item.get("created_day_part"),
                    "lookback_bucket": item.get("lookback_bucket"),
                    "cohort_size": int(item.get("cohort_size") or 0),
                    "quality_score": round(quality, 6),
                    "confidence": round(
                        max(0.0, min(evidence_confidence, 1.0)),
                        6,
                    ),
                    "confidence_calibration": "provisional_offline_evidence",
                    "score_breakdown": {
                        "vector_similarity": round(vector_score, 6),
                        "lexical_score": round(
                            float(item.get("lexical_score") or 0.0),
                            6,
                        ),
                        "rank_fusion_score": round(fused_score, 8),
                        "vector_lexical_agreement": agreement,
                        "next_candidate_margin": round(margin, 6),
                    },
                    "freshness_status": item.get("freshness_status"),
                    "privacy_status": item.get("privacy_status"),
                    "rights_status": item.get("rights_status"),
                    "data_use_mode": item.get("data_use_mode"),
                    "activation_eligible": bool(
                        execution_mode == "production"
                        and item.get("eligible_for_activation")
                        and item.get("freshness_status") == "fresh"
                    ),
                }
            )
        return output

    def _blocked_response(
        self,
        *,
        request: dict[str, Any],
        feature_set: dict[str, Any],
        proposal_id: str,
        execution_mode: str,
        reason_code: str,
        explanation: str,
        audience_request_detected: bool = True,
    ) -> dict[str, Any]:
        return {
            "contract_version": self.CONTRACT_VERSION,
            "proposal_id": proposal_id,
            "idempotency_key": request.get("idempotency_key"),
            "tenant_id": normalize_taxonomy_value(request.get("tenant_id")),
            "campaign_id": request.get("campaign_id"),
            "status": "blocked",
            "execution_mode": execution_mode,
            "reason_code": reason_code,
            "audience_request_detected": audience_request_detected,
            "eligible_for_audience_selection": False,
            "feature_set": self._feature_set_summary(feature_set),
            "retrieval": {
                "pipeline": "structured_filters_then_lexical_pgvector_rrf",
                "candidate_count": 0,
                "structured_filters_applied_before_ranking": True,
                "semantic_similarity_can_override_safety": False,
            },
            "candidate_cohorts": [],
            "coverage_warnings": [explanation],
            "approval_required": True,
            "approval_status": reason_code,
            "activation_eligible": False,
            "safe_export_eligible": False,
            "downstream_export_enabled": False,
            "explanation": explanation,
        }

    def _feature_set_summary(
        self,
        feature_set: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = feature_set.get("source_latest_at")
        if timestamp is not None and hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()
        return {
            "feature_set_id": feature_set.get("feature_set_id"),
            "version": feature_set.get("version"),
            "source_mode": feature_set.get("source_mode"),
            "data_use_mode": feature_set.get("data_use_mode"),
            "source_latest_at": timestamp,
            "freshness_status": feature_set.get("freshness_status"),
            "feature_count": int(feature_set.get("feature_count") or 0),
            "model_backend": feature_set.get("model_backend"),
            "model_name": feature_set.get("model_name"),
            "model_version": feature_set.get("model_version"),
            "privacy_policy_version": feature_set.get(
                "privacy_policy_version"
            ),
            "rights_policy_id": feature_set.get("rights_policy_id"),
            "purpose": feature_set.get("purpose"),
            "eligible_for_activation": bool(
                feature_set.get("eligible_for_activation")
            ),
        }

    def _proposal_id(
        self,
        *,
        tenant_id: str,
        campaign_id: str,
        idempotency_key: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> str:
        payload = json.dumps(
            {
                "tenant_id": tenant_id,
                "campaign_id": campaign_id,
                "idempotency_key": idempotency_key,
                "feature_set_id": feature_set_id,
                "feature_set_version": feature_set_version,
                "contract_version": self.CONTRACT_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "audience_proposal_" + hashlib.sha256(payload).hexdigest()[:24]

    def _clean_structured_list(
        self,
        value: Any,
    ) -> tuple[list[str], bool]:
        if value is None:
            return [], False
        values = [value] if isinstance(value, str) else list(value)
        output: list[str] = []
        invalid = False
        for item in values:
            raw = " ".join(str(item or "").split())
            if not raw:
                continue
            normalized = normalize_taxonomy_value(raw)
            if not normalized:
                # Never silently drop a non-empty structured filter. The
                # legacy feature proposal boundary accepts canonical ASCII
                # slugs only; multilingual text must first pass through the
                # governed canonicalizer. Dropping the value would remove a
                # hard eligibility or exclusion constraint.
                invalid = True
                continue
            if normalized not in output:
                output.append(normalized)
        return output, invalid

    def _required_slug(self, value: Any, label: str) -> str:
        normalized = normalize_taxonomy_value(value)
        if not normalized:
            raise ValueError(f"{label} is required.")
        return normalized

    def _required_text(self, value: Any, label: str) -> str:
        text_value = " ".join(str(value or "").split())
        if not text_value:
            raise ValueError(f"{label} is required.")
        return text_value
