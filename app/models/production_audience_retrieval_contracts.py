from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    stable_digest,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec

ConstraintDimension = Literal["location", "category", "daypart"]
ConstraintStatus = Literal[
    "resolved",
    "not_requested",
    "unresolved",
    "ambiguous",
    "unsupported",
]


def normalize_multilingual_text(value: Any) -> str:
    """Normalize text without discarding non-Latin scripts."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


@dataclass(frozen=True)
class ConstraintTaxonomyEntry:
    canonical_value: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    descriptions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        canonical = normalize_taxonomy_value(self.canonical_value)
        if not canonical:
            raise ValueError("canonical_value is required.")
        normalized_aliases = tuple(
            dict.fromkeys(
                value
                for raw in (
                    self.canonical_value.replace("_", " "),
                    *self.aliases,
                )
                if (value := normalize_multilingual_text(raw))
            )
        )
        if not normalized_aliases:
            raise ValueError("At least one taxonomy alias is required.")
        descriptions = tuple(
            dict.fromkeys(
                " ".join(str(value or "").split())
                for value in self.descriptions
                if " ".join(str(value or "").split())
            )
        )
        object.__setattr__(self, "canonical_value", canonical)
        object.__setattr__(self, "aliases", normalized_aliases)
        object.__setattr__(self, "descriptions", descriptions)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "canonical_value": self.canonical_value,
            "aliases": list(self.aliases),
            "descriptions": list(self.descriptions),
        }


@dataclass(frozen=True)
class GovernedConstraintTaxonomy:
    taxonomy_id: str
    version: str
    locations: tuple[ConstraintTaxonomyEntry, ...]
    categories: tuple[ConstraintTaxonomyEntry, ...]
    dayparts: tuple[ConstraintTaxonomyEntry, ...]
    reviewed_by: str
    review_status: Literal["approved", "draft", "retired"] = "approved"
    contract_version: str = "production-constraint-taxonomy-v1"

    def __post_init__(self) -> None:
        taxonomy_id = normalize_taxonomy_value(self.taxonomy_id)
        version = " ".join(str(self.version or "").split())
        reviewed_by = normalize_taxonomy_value(self.reviewed_by)
        if not taxonomy_id or not version or not reviewed_by:
            raise ValueError(
                "taxonomy_id, version, and reviewed_by are required."
            )
        if self.review_status not in {"approved", "draft", "retired"}:
            raise ValueError("Unsupported taxonomy review_status.")
        for dimension, entries in self.dimension_entries().items():
            if not entries:
                raise ValueError(f"{dimension} taxonomy cannot be empty.")
            canonical_values = [entry.canonical_value for entry in entries]
            if len(canonical_values) != len(set(canonical_values)):
                raise ValueError(
                    f"{dimension} taxonomy has duplicate canonical values."
                )
            aliases: dict[str, str] = {}
            for entry in entries:
                for alias in entry.aliases:
                    existing = aliases.get(alias)
                    if existing and existing != entry.canonical_value:
                        raise ValueError(
                            f"{dimension} alias {alias!r} is ambiguous."
                        )
                    aliases[alias] = entry.canonical_value
        object.__setattr__(self, "taxonomy_id", taxonomy_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "reviewed_by", reviewed_by)

    def dimension_entries(
        self,
    ) -> dict[ConstraintDimension, tuple[ConstraintTaxonomyEntry, ...]]:
        return {
            "location": tuple(self.locations),
            "category": tuple(self.categories),
            "daypart": tuple(self.dayparts),
        }

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_safe_dict(include_fingerprint=False))

    def to_safe_dict(
        self,
        *,
        include_fingerprint: bool = True,
    ) -> dict[str, Any]:
        result = {
            "contract_version": self.contract_version,
            "taxonomy_id": self.taxonomy_id,
            "version": self.version,
            "reviewed_by": self.reviewed_by,
            "review_status": self.review_status,
            "locations": [entry.to_safe_dict() for entry in self.locations],
            "categories": [entry.to_safe_dict() for entry in self.categories],
            "dayparts": [entry.to_safe_dict() for entry in self.dayparts],
        }
        if include_fingerprint:
            result["taxonomy_fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True)
class ConstraintFieldResolution:
    dimension: ConstraintDimension
    status: ConstraintStatus
    values: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    margin: float = 0.0
    evidence: tuple[str, ...] = field(default_factory=tuple)
    unresolved_inputs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        values = tuple(
            dict.fromkeys(
                value
                for raw in self.values
                if (value := normalize_taxonomy_value(raw))
            )
        )
        unresolved_inputs = tuple(
            dict.fromkeys(
                value
                for raw in self.unresolved_inputs
                if (value := normalize_multilingual_text(raw))
            )
        )
        confidence = float(self.confidence)
        margin = float(self.margin)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not -1.0 <= margin <= 1.0:
            raise ValueError("margin must be between -1 and 1.")
        if self.status == "resolved" and not values:
            raise ValueError("Resolved constraints require at least one value.")
        if self.status != "resolved" and values:
            raise ValueError(
                "Only resolved constraints may contain canonical values."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "unresolved_inputs", unresolved_inputs)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(
            self,
            "evidence",
            tuple(dict.fromkeys(str(value) for value in self.evidence)),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "status": self.status,
            "values": list(self.values),
            "confidence": round(self.confidence, 6),
            "margin": round(self.margin, 6),
            "evidence": list(self.evidence),
            "unresolved_inputs": list(self.unresolved_inputs),
        }


@dataclass(frozen=True)
class MultilingualConstraintResolution:
    tenant_id: str
    query_fingerprint: str
    language: str
    taxonomy_id: str
    taxonomy_version: str
    taxonomy_fingerprint: str
    canonicalizer_model_fingerprint: str
    locations: ConstraintFieldResolution
    categories: ConstraintFieldResolution
    dayparts: ConstraintFieldResolution
    ready_for_retrieval: bool
    reason_code: str
    clarification_fields: tuple[str, ...] = field(default_factory=tuple)
    contract_version: str = "production-multilingual-constraint-resolution-v1"

    def __post_init__(self) -> None:
        tenant_id = normalize_taxonomy_value(self.tenant_id)
        language = normalize_taxonomy_value(self.language or "und") or "und"
        if not tenant_id:
            raise ValueError("tenant_id is required.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.query_fingerprint):
            raise ValueError("query_fingerprint must be a lowercase SHA-256 digest.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.taxonomy_fingerprint):
            raise ValueError(
                "taxonomy_fingerprint must be a lowercase SHA-256 digest."
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            self.canonicalizer_model_fingerprint,
        ):
            raise ValueError(
                "canonicalizer_model_fingerprint must be a SHA-256 digest."
            )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "language", language)
        object.__setattr__(
            self,
            "clarification_fields",
            tuple(
                dict.fromkeys(
                    normalize_taxonomy_value(value)
                    for value in self.clarification_fields
                    if normalize_taxonomy_value(value)
                )
            ),
        )

    def resolved_values(self) -> dict[str, tuple[str, ...]]:
        return {
            "locations": self.locations.values,
            "categories": self.categories.values,
            "dayparts": self.dayparts.values,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "tenant_id": self.tenant_id,
            "query_fingerprint": self.query_fingerprint,
            "language": self.language,
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_version": self.taxonomy_version,
            "taxonomy_fingerprint": self.taxonomy_fingerprint,
            "canonicalizer_model_fingerprint": (
                self.canonicalizer_model_fingerprint
            ),
            "constraints": {
                "locations": self.locations.to_safe_dict(),
                "categories": self.categories.to_safe_dict(),
                "dayparts": self.dayparts.to_safe_dict(),
            },
            "ready_for_retrieval": self.ready_for_retrieval,
            "reason_code": self.reason_code,
            "clarification_fields": list(self.clarification_fields),
            "raw_query_stored": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
        }


@dataclass(frozen=True)
class ProductionRetrievalModelBinding:
    role: Literal["primary", "complementary"]
    feature_set_id: str
    feature_set_version: int
    model: EmbeddingModelSpec
    initial_depth: int = 10
    expansion_depth: int = 10

    def __post_init__(self) -> None:
        feature_set_id = " ".join(str(self.feature_set_id or "").split())
        if not feature_set_id:
            raise ValueError("feature_set_id is required.")
        if int(self.feature_set_version) < 1:
            raise ValueError("feature_set_version must be at least 1.")
        if int(self.initial_depth) != 10:
            raise ValueError("The governed initial candidate depth must be 10.")
        expected_expansion = 30 if self.role == "primary" else 10
        if int(self.expansion_depth) != expected_expansion:
            raise ValueError(
                "Primary expansion depth must be 30 and complementary depth 10."
            )
        object.__setattr__(self, "feature_set_id", feature_set_id)
        object.__setattr__(
            self,
            "feature_set_version",
            int(self.feature_set_version),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "feature_set_id": self.feature_set_id,
            "feature_set_version": self.feature_set_version,
            "model": self.model.to_safe_dict(),
            "initial_depth": self.initial_depth,
            "expansion_depth": self.expansion_depth,
        }


@dataclass(frozen=True)
class GovernedAudienceRetrievalRequest:
    tenant_id: str
    query_text: str
    language: str
    execution_mode: Literal["historical_preview", "production"]
    taxonomy: GovernedConstraintTaxonomy
    canonicalizer_model: EmbeddingModelSpec
    primary_model: ProductionRetrievalModelBinding
    complementary_model: ProductionRetrievalModelBinding
    requested_locations: tuple[str, ...] = field(default_factory=tuple)
    requested_categories: tuple[str, ...] = field(default_factory=tuple)
    requested_dayparts: tuple[str, ...] = field(default_factory=tuple)
    exclusions: tuple[str, ...] = field(default_factory=tuple)
    result_limit: int = 10

    def __post_init__(self) -> None:
        tenant_id = normalize_taxonomy_value(self.tenant_id)
        query_text = " ".join(str(self.query_text or "").split())
        language = normalize_taxonomy_value(self.language or "und") or "und"
        if not tenant_id or not query_text:
            raise ValueError("tenant_id and query_text are required.")
        if self.execution_mode not in {"historical_preview", "production"}:
            raise ValueError(
                "execution_mode must be historical_preview or production."
            )
        if self.taxonomy.review_status != "approved":
            raise ValueError("Production retrieval requires an approved taxonomy.")
        if self.primary_model.role != "primary":
            raise ValueError("primary_model must use the primary role.")
        if self.complementary_model.role != "complementary":
            raise ValueError(
                "complementary_model must use the complementary role."
            )
        if not 1 <= int(self.result_limit) <= 50:
            raise ValueError("result_limit must be between 1 and 50.")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "query_text", query_text)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "result_limit", int(self.result_limit))
        for field_name in (
            "requested_locations",
            "requested_categories",
            "requested_dayparts",
            "exclusions",
        ):
            values: Sequence[str] = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                tuple(
                    dict.fromkeys(
                        " ".join(str(value or "").split())
                        for value in values
                        if " ".join(str(value or "").split())
                    )
                ),
            )

    @property
    def query_fingerprint(self) -> str:
        return stable_digest(
            {
                "tenant_id": self.tenant_id,
                "query_text": self.query_text,
                "language": self.language,
            }
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "query_fingerprint": self.query_fingerprint,
            "language": self.language,
            "execution_mode": self.execution_mode,
            "taxonomy": self.taxonomy.to_safe_dict(),
            "canonicalizer_model": self.canonicalizer_model.to_safe_dict(),
            "primary_model": self.primary_model.to_safe_dict(),
            "complementary_model": self.complementary_model.to_safe_dict(),
            "requested_locations_count": len(self.requested_locations),
            "requested_categories_count": len(self.requested_categories),
            "requested_dayparts_count": len(self.requested_dayparts),
            "exclusion_count": len(self.exclusions),
            "result_limit": self.result_limit,
            "raw_query_stored": False,
        }
