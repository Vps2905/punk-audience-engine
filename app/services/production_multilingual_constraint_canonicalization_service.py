from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from app.models.production_audience_retrieval_contracts import (
    ConstraintDimension,
    ConstraintFieldResolution,
    ConstraintTaxonomyEntry,
    GovernedAudienceRetrievalRequest,
    MultilingualConstraintResolution,
    normalize_multilingual_text,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec


class EmbeddingModelApprovalGate(Protocol):
    def require_approved(
        self,
        *,
        tenant_id: str,
        model: EmbeddingModelSpec,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SemanticConstraintScore:
    canonical_value: str
    score: float


class ConstraintSemanticResolver(Protocol):
    def score(
        self,
        *,
        query_text: str,
        dimension: ConstraintDimension,
        entries: Sequence[ConstraintTaxonomyEntry],
        model: EmbeddingModelSpec,
    ) -> Sequence[SemanticConstraintScore]: ...


class SentenceTransformerConstraintSemanticResolver:
    """Pinned multilingual semantic scoring with no unversioned fallback."""

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

    def __init__(self, *, device: str = "cpu") -> None:
        self._device = str(device or "cpu")
        self._models: dict[tuple[str, str, str], Any] = {}

    def score(
        self,
        *,
        query_text: str,
        dimension: ConstraintDimension,
        entries: Sequence[ConstraintTaxonomyEntry],
        model: EmbeddingModelSpec,
    ) -> Sequence[SemanticConstraintScore]:
        if dimension == "location":
            raise RuntimeError(
                "Semantic location mapping is disabled for fail-closed retrieval."
            )
        if model.backend != "sentence_transformers":
            raise RuntimeError(
                "Constraint canonicalization requires sentence_transformers."
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

        label_texts: list[str] = []
        ranges: dict[str, tuple[int, int]] = {}
        for entry in entries:
            start = len(label_texts)
            values = [
                *entry.aliases,
                *entry.descriptions,
            ]
            label_texts.extend(
                " ".join(str(value or "").split())
                for value in values
                if " ".join(str(value or "").split())
            )
            ranges[entry.canonical_value] = (start, len(label_texts))

        unspecified_start = len(label_texts)
        label_texts.extend(self._UNSPECIFIED_DESCRIPTIONS[dimension])
        ranges["_unspecified"] = (unspecified_start, len(label_texts))

        if not label_texts:
            return []
        encoded = loaded.encode(
            [f"{model.query_prefix}{query_text}"]
            + [f"{model.document_prefix}{text}" for text in label_texts],
            convert_to_numpy=True,
            normalize_embeddings=model.normalize_embeddings,
            show_progress_bar=False,
        )
        matrix = np.asarray(encoded, dtype=np.float64)
        query_vector = matrix[0]
        label_vectors = matrix[1:]
        output: list[SemanticConstraintScore] = []
        for canonical_value, (start, end) in ranges.items():
            if end <= start:
                continue
            similarities = label_vectors[start:end] @ query_vector
            output.append(
                SemanticConstraintScore(
                    canonical_value=canonical_value,
                    score=float(np.max(similarities)),
                )
            )
        return sorted(output, key=lambda item: item.score, reverse=True)


class ProductionMultilingualConstraintCanonicalizationService:
    """
    Resolve governed multilingual constraints without nearest-location guessing.

    The service stores no raw query text and exposes per-field confidence,
    margins, evidence, ambiguity, and clarification requirements.
    """

    def __init__(
        self,
        *,
        model_registry: EmbeddingModelApprovalGate,
        semantic_resolver: ConstraintSemanticResolver | None = None,
        semantic_min_score: float = 0.55,
        semantic_min_margin: float = 0.05,
    ) -> None:
        self._model_registry = model_registry
        self._semantic_resolver = (
            semantic_resolver
            or SentenceTransformerConstraintSemanticResolver()
        )
        self._semantic_min_score = float(semantic_min_score)
        self._semantic_min_margin = float(semantic_min_margin)
        if not 0.0 <= self._semantic_min_score <= 1.0:
            raise ValueError("semantic_min_score must be between 0 and 1.")
        if not 0.0 <= self._semantic_min_margin <= 1.0:
            raise ValueError("semantic_min_margin must be between 0 and 1.")

    def canonicalize(
        self,
        request: GovernedAudienceRetrievalRequest,
    ) -> MultilingualConstraintResolution:
        self._model_registry.require_approved(
            tenant_id=request.tenant_id,
            model=request.canonicalizer_model,
        )
        entries = request.taxonomy.dimension_entries()
        locations = self._resolve_location(
            query_text=request.query_text,
            explicit_values=request.requested_locations,
            entries=entries["location"],
        )
        categories = self._resolve_semantic_dimension(
            query_text=request.query_text,
            explicit_values=request.requested_categories,
            entries=entries["category"],
            dimension="category",
            model=request.canonicalizer_model,
            required=True,
        )
        dayparts = self._resolve_semantic_dimension(
            query_text=request.query_text,
            explicit_values=request.requested_dayparts,
            entries=entries["daypart"],
            dimension="daypart",
            model=request.canonicalizer_model,
            required=False,
        )

        clarification_fields = tuple(
            dimension
            for dimension, resolution in (
                ("location", locations),
                ("category", categories),
                ("daypart", dayparts),
            )
            if resolution.status in {
                "unresolved",
                "ambiguous",
                "unsupported",
            }
        )
        ready = (
            locations.status == "resolved"
            and categories.status == "resolved"
            and dayparts.status in {"resolved", "not_requested"}
            and not clarification_fields
        )
        reason_code = (
            "constraints_ready_for_governed_retrieval"
            if ready
            else self._reason_code(locations, categories, dayparts)
        )
        return MultilingualConstraintResolution(
            tenant_id=request.tenant_id,
            query_fingerprint=request.query_fingerprint,
            language=request.language,
            taxonomy_id=request.taxonomy.taxonomy_id,
            taxonomy_version=request.taxonomy.version,
            taxonomy_fingerprint=request.taxonomy.fingerprint,
            canonicalizer_model_fingerprint=(
                request.canonicalizer_model.fingerprint
            ),
            locations=locations,
            categories=categories,
            dayparts=dayparts,
            ready_for_retrieval=ready,
            reason_code=reason_code,
            clarification_fields=clarification_fields,
        )

    def _resolve_location(
        self,
        *,
        query_text: str,
        explicit_values: Sequence[str],
        entries: Sequence[ConstraintTaxonomyEntry],
    ) -> ConstraintFieldResolution:
        aliases = self._alias_index(entries)
        if explicit_values:
            resolved, unresolved = self._resolve_explicit(
                explicit_values,
                aliases,
            )
            if unresolved:
                return ConstraintFieldResolution(
                    dimension="location",
                    status="unsupported",
                    confidence=0.0,
                    evidence=("explicit_location_not_in_approved_coverage",),
                    unresolved_inputs=tuple(unresolved),
                )
            return ConstraintFieldResolution(
                dimension="location",
                status="resolved",
                values=tuple(resolved),
                confidence=1.0,
                margin=1.0,
                evidence=("approved_exact_location_alias",),
            )

        exact = self._query_alias_matches(query_text, entries)
        if len(exact) == 1:
            return ConstraintFieldResolution(
                dimension="location",
                status="resolved",
                values=tuple(exact),
                confidence=1.0,
                margin=1.0,
                evidence=("approved_exact_location_alias_in_query",),
            )
        if len(exact) > 1:
            return ConstraintFieldResolution(
                dimension="location",
                status="ambiguous",
                confidence=0.0,
                evidence=("multiple_approved_locations_in_query",),
            )
        return ConstraintFieldResolution(
            dimension="location",
            status="unresolved",
            confidence=0.0,
            evidence=("location_requires_exact_approved_alias",),
        )

    def _resolve_semantic_dimension(
        self,
        *,
        query_text: str,
        explicit_values: Sequence[str],
        entries: Sequence[ConstraintTaxonomyEntry],
        dimension: ConstraintDimension,
        model: EmbeddingModelSpec,
        required: bool,
    ) -> ConstraintFieldResolution:
        aliases = self._alias_index(entries)
        if explicit_values:
            resolved, unresolved = self._resolve_explicit(
                explicit_values,
                aliases,
            )
            if unresolved:
                return ConstraintFieldResolution(
                    dimension=dimension,
                    status="unresolved",
                    confidence=0.0,
                    evidence=("explicit_value_not_in_approved_taxonomy",),
                    unresolved_inputs=tuple(unresolved),
                )
            return ConstraintFieldResolution(
                dimension=dimension,
                status="resolved",
                values=tuple(resolved),
                confidence=1.0,
                margin=1.0,
                evidence=("approved_exact_taxonomy_alias",),
            )

        exact = self._query_alias_matches(query_text, entries)
        if len(exact) == 1:
            return ConstraintFieldResolution(
                dimension=dimension,
                status="resolved",
                values=tuple(exact),
                confidence=1.0,
                margin=1.0,
                evidence=("approved_exact_taxonomy_alias_in_query",),
            )
        if len(exact) > 1:
            return ConstraintFieldResolution(
                dimension=dimension,
                status="ambiguous",
                confidence=0.0,
                evidence=("multiple_taxonomy_values_in_query",),
            )

        scores = list(
            self._semantic_resolver.score(
                query_text=query_text,
                dimension=dimension,
                entries=entries,
                model=model,
            )
        )
        if not scores:
            return self._missing_resolution(dimension, required)
        finite_scores = [
            item for item in scores if math.isfinite(float(item.score))
        ]
        if not finite_scores:
            return self._missing_resolution(dimension, required)
        ranked = sorted(
            finite_scores,
            key=lambda item: item.score,
            reverse=True,
        )
        top = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = float(top.score - second_score)
        if top.canonical_value == "_unspecified":
            return self._missing_resolution(dimension, required)
        if (
            top.score < self._semantic_min_score
            or margin < self._semantic_min_margin
        ):
            return ConstraintFieldResolution(
                dimension=dimension,
                status="ambiguous",
                confidence=max(0.0, min(float(top.score), 1.0)),
                margin=max(-1.0, min(margin, 1.0)),
                evidence=("semantic_model_below_governed_margin",),
            )
        return ConstraintFieldResolution(
            dimension=dimension,
            status="resolved",
            values=(top.canonical_value,),
            confidence=max(0.0, min(float(top.score), 1.0)),
            margin=max(-1.0, min(margin, 1.0)),
            evidence=("approved_pinned_semantic_model",),
        )

    def _missing_resolution(
        self,
        dimension: ConstraintDimension,
        required: bool,
    ) -> ConstraintFieldResolution:
        return ConstraintFieldResolution(
            dimension=dimension,
            status="unresolved" if required else "not_requested",
            confidence=0.0,
            evidence=(
                "required_constraint_not_resolved"
                if required
                else "constraint_not_requested"
            ,),
        )

    def _alias_index(
        self,
        entries: Sequence[ConstraintTaxonomyEntry],
    ) -> dict[str, str]:
        return {
            alias: entry.canonical_value
            for entry in entries
            for alias in entry.aliases
        }

    def _resolve_explicit(
        self,
        values: Sequence[str],
        aliases: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        resolved: list[str] = []
        unresolved: list[str] = []
        for raw in values:
            normalized = normalize_multilingual_text(raw)
            canonical = aliases.get(normalized)
            if canonical:
                if canonical not in resolved:
                    resolved.append(canonical)
            elif normalized:
                unresolved.append(normalized)
        return resolved, unresolved

    def _query_alias_matches(
        self,
        query_text: str,
        entries: Sequence[ConstraintTaxonomyEntry],
    ) -> list[str]:
        normalized_query = normalize_multilingual_text(query_text)
        wrapped = f" {normalized_query} "
        matches: list[str] = []
        for entry in entries:
            if any(f" {alias} " in wrapped for alias in entry.aliases):
                matches.append(entry.canonical_value)
        return list(dict.fromkeys(matches))

    def _reason_code(
        self,
        locations: ConstraintFieldResolution,
        categories: ConstraintFieldResolution,
        dayparts: ConstraintFieldResolution,
    ) -> str:
        if locations.status == "unsupported":
            return "unsupported_location_requires_clarification"
        if locations.status != "resolved":
            return "location_requires_clarification"
        if categories.status != "resolved":
            return "category_requires_clarification"
        if dayparts.status not in {"resolved", "not_requested"}:
            return "daypart_requires_clarification"
        return "constraints_require_clarification"
