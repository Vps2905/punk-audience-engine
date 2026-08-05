from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from app.models.embedding_benchmark_contracts import EmbeddingBenchmarkDataset
from app.models.production_audience_retrieval_contracts import (
    ConstraintTaxonomyEntry,
    GovernedConstraintTaxonomy,
)

ENGINEERING_LANGUAGE_PACK_SCOPE = "engineering_benchmark_generation_only"
ENGINEERING_TAXONOMY_SCOPE = "engineering_benchmark_only"
PRODUCTION_TAXONOMY_SCOPE = "production_retrieval"
PENDING_NATIVE_SIGNOFF = "pending_native_human_signoff"
PRODUCTION_CERTIFIED = "approved_for_production_retrieval"


@dataclass(frozen=True)
class LoadedGovernedConstraintTaxonomy:
    taxonomy: GovernedConstraintTaxonomy
    lineage: Mapping[str, Any]

    def to_safe_metadata(self) -> dict[str, Any]:
        return {
            "approval_scope": self.lineage["approval_scope"],
            "native_human_review_completed": self.lineage[
                "native_human_review_completed"
            ],
            "production_certification_status": self.lineage[
                "production_certification_status"
            ],
            "oracle_case_constraints_used": self.lineage[
                "oracle_case_constraints_used"
            ],
            "source_language_pack_id": self.lineage.get(
                "source_language_pack_id"
            ),
            "source_language_pack_version": self.lineage.get(
                "source_language_pack_version"
            ),
            "source_review_method": self.lineage.get(
                "source_review_method"
            ),
            "source_language_pack_sha256": self.lineage.get(
                "source_language_pack_sha256"
            ),
            "source_dataset_fingerprint": self.lineage.get(
                "source_dataset_fingerprint"
            ),
        }


@dataclass(frozen=True)
class GovernedTaxonomyBuildArtifacts:
    taxonomy: GovernedConstraintTaxonomy
    taxonomy_payload: dict[str, Any]
    dataset_envelope: dict[str, Any]
    canonical_dataset_fingerprint: str
    location_count: int
    category_count: int
    daypart_count: int


def load_governed_constraint_taxonomy(
    value: Mapping[str, Any],
    *,
    allow_engineering_scope: bool,
    require_native_human_review: bool,
) -> LoadedGovernedConstraintTaxonomy:
    """Load one reviewed taxonomy and fail closed on lineage or fingerprint gaps."""

    if not isinstance(value, Mapping):
        raise ValueError("Governed taxonomy must be a mapping.")

    taxonomy = GovernedConstraintTaxonomy(
        taxonomy_id=str(value.get("taxonomy_id") or ""),
        version=str(value.get("version") or ""),
        reviewed_by=str(value.get("reviewed_by") or ""),
        review_status=str(value.get("review_status") or "draft"),
        locations=_taxonomy_entries(value, "locations"),
        categories=_taxonomy_entries(value, "categories"),
        dayparts=_taxonomy_entries(value, "dayparts"),
    )
    if taxonomy.review_status != "approved":
        raise ValueError("Governed taxonomy must be approved.")

    declared_fingerprint = str(value.get("taxonomy_fingerprint") or "").strip()
    if declared_fingerprint != taxonomy.fingerprint:
        raise ValueError(
            "Declared taxonomy_fingerprint does not match canonical taxonomy."
        )

    lineage_value = value.get("lineage")
    if not isinstance(lineage_value, Mapping):
        raise ValueError("Governed taxonomy lineage is required.")
    lineage = dict(lineage_value)

    approval_scope = str(lineage.get("approval_scope") or "").strip()
    allowed_scopes = {PRODUCTION_TAXONOMY_SCOPE}
    if allow_engineering_scope:
        allowed_scopes.add(ENGINEERING_TAXONOMY_SCOPE)
    if approval_scope not in allowed_scopes:
        raise ValueError("Governed taxonomy approval_scope is not allowed.")

    native_review = lineage.get("native_human_review_completed")
    if not isinstance(native_review, bool):
        raise ValueError(
            "Governed taxonomy native_human_review_completed must be boolean."
        )
    if require_native_human_review and native_review is not True:
        raise ValueError("Governed taxonomy requires native-human review.")
    if approval_scope == PRODUCTION_TAXONOMY_SCOPE and native_review is not True:
        raise ValueError(
            "Production taxonomy scope requires completed native-human review."
        )

    oracle_used = lineage.get("oracle_case_constraints_used")
    if oracle_used is not False:
        raise ValueError(
            "Governed taxonomy cannot use benchmark oracle case constraints."
        )

    certification_status = str(
        lineage.get("production_certification_status") or ""
    ).strip()
    if not certification_status:
        raise ValueError(
            "Governed taxonomy production_certification_status is required."
        )
    if native_review is False and certification_status != PENDING_NATIVE_SIGNOFF:
        raise ValueError(
            "Unreviewed taxonomy must remain pending native-human signoff."
        )
    if (
        approval_scope == PRODUCTION_TAXONOMY_SCOPE
        and certification_status != PRODUCTION_CERTIFIED
    ):
        raise ValueError(
            "Production taxonomy must be explicitly certified for retrieval."
        )

    if approval_scope == ENGINEERING_TAXONOMY_SCOPE:
        _require_text(lineage, "source_language_pack_id")
        _require_text(lineage, "source_language_pack_version")
        _require_text(lineage, "source_review_method")
        _require_sha256(lineage, "source_language_pack_sha256")
        _require_sha256(lineage, "source_dataset_fingerprint")

    return LoadedGovernedConstraintTaxonomy(
        taxonomy=taxonomy,
        lineage=MappingProxyType(deepcopy(lineage)),
    )


def build_engineering_multilingual_taxonomy_artifacts(
    *,
    dataset_payload: Mapping[str, Any],
    language_pack: Mapping[str, Any],
    language_pack_sha256: str,
) -> GovernedTaxonomyBuildArtifacts:
    """Build engineering-only taxonomy artifacts from documents, never case gold."""

    if not isinstance(dataset_payload, Mapping):
        raise ValueError("Benchmark dataset must be a mapping.")
    if not isinstance(language_pack, Mapping):
        raise ValueError("Language pack must be a mapping.")

    canonical_dataset = EmbeddingBenchmarkDataset.from_mapping(dataset_payload)
    _validate_engineering_language_pack(language_pack)
    pack_id = _require_text(language_pack, "pack_id")
    pack_version = _require_text(language_pack, "pack_version")
    reviewed_by = _require_text(language_pack, "reviewed_by")
    source_lineage = dict(language_pack.get("lineage") or {})
    source_review_method = _require_text(source_lineage, "review_method")
    pack_sha256 = _normalized_sha256(
        language_pack_sha256,
        "language_pack_sha256",
    )

    languages_value = language_pack.get("languages")
    if not isinstance(languages_value, Mapping):
        raise ValueError("Language-pack languages must be a mapping.")
    languages = dict(languages_value)
    language_order = tuple(
        " ".join(str(item).split())
        for item in language_pack["language_order"]
    )

    locations = sorted({item.location for item in canonical_dataset.documents})
    categories = sorted({item.category for item in canonical_dataset.documents})
    dayparts = sorted({item.daypart for item in canonical_dataset.documents})

    lineage = source_lineage
    observed_category_count = lineage.get("observed_category_count")
    if observed_category_count is not None and int(observed_category_count) != len(
        categories
    ):
        raise ValueError(
            "Language-pack observed_category_count does not match dataset."
        )
    observed_daypart_count = lineage.get("observed_daypart_count")
    if observed_daypart_count is not None and int(observed_daypart_count) != len(
        dayparts
    ):
        raise ValueError(
            "Language-pack observed_daypart_count does not match dataset."
        )

    def reviewed_labels(
        *,
        dimension_key: str,
        canonical_value: str,
    ) -> tuple[str, ...]:
        labels: list[str] = []
        for language in language_order:
            definition = languages.get(language)
            if not isinstance(definition, Mapping):
                raise ValueError(
                    f"Language-pack definition is invalid: {language}."
                )
            mapping = definition.get(dimension_key)
            if not isinstance(mapping, Mapping):
                raise ValueError(
                    f"Language-pack {dimension_key} is invalid: {language}."
                )
            label = " ".join(
                str(mapping.get(canonical_value) or "").split()
            )
            if not label:
                raise ValueError(
                    "Missing reviewed taxonomy label for "
                    f"{dimension_key}.{canonical_value}.{language}."
                )
            labels.append(label)
        return tuple(dict.fromkeys(labels))

    location_entries = tuple(
        ConstraintTaxonomyEntry(
            canonical_value=value,
            aliases=(value.replace("_", " "),),
            descriptions=(
                f"approved covered location {value.replace('_', ' ')}",
            ),
        )
        for value in locations
    )
    category_entries = tuple(
        ConstraintTaxonomyEntry(
            canonical_value=value,
            aliases=(
                value.replace("_", " "),
                *reviewed_labels(
                    dimension_key="category_labels",
                    canonical_value=value,
                ),
            ),
            descriptions=reviewed_labels(
                dimension_key="category_labels",
                canonical_value=value,
            ),
        )
        for value in categories
    )
    daypart_entries = tuple(
        ConstraintTaxonomyEntry(
            canonical_value=value,
            aliases=(
                value.replace("_", " "),
                *reviewed_labels(
                    dimension_key="daypart_labels",
                    canonical_value=value,
                ),
            ),
            descriptions=reviewed_labels(
                dimension_key="daypart_labels",
                canonical_value=value,
            ),
        )
        for value in dayparts
    )

    taxonomy = GovernedConstraintTaxonomy(
        taxonomy_id=(
            "punk-module2-engineering-multilingual-constraint-taxonomy"
        ),
        version=(
            f"{pack_version}"
            "-engineering-exact-alias-evaluation-v2"
        ),
        locations=location_entries,
        categories=category_entries,
        dayparts=daypart_entries,
        reviewed_by=reviewed_by,
        review_status="approved",
    )
    taxonomy_payload = taxonomy.to_safe_dict()
    taxonomy_payload["lineage"] = {
        "approval_scope": ENGINEERING_TAXONOMY_SCOPE,
        "source_language_pack_id": pack_id,
        "source_language_pack_version": pack_version,
        "source_review_method": source_review_method,
        "source_language_pack_sha256": pack_sha256,
        "source_dataset_fingerprint": canonical_dataset.fingerprint,
        "native_human_review_completed": False,
        "production_certification_status": PENDING_NATIVE_SIGNOFF,
        "oracle_case_constraints_used": False,
    }

    loaded = load_governed_constraint_taxonomy(
        taxonomy_payload,
        allow_engineering_scope=True,
        require_native_human_review=False,
    )
    if loaded.taxonomy.fingerprint != taxonomy.fingerprint:
        raise ValueError("Built taxonomy failed canonical reload validation.")

    envelope = deepcopy(dict(dataset_payload))
    envelope["governed_constraint_taxonomy"] = deepcopy(taxonomy_payload)
    envelope_dataset = EmbeddingBenchmarkDataset.from_mapping(envelope)
    if envelope_dataset.fingerprint != canonical_dataset.fingerprint:
        raise ValueError("Taxonomy envelope changed canonical dataset identity.")

    return GovernedTaxonomyBuildArtifacts(
        taxonomy=taxonomy,
        taxonomy_payload=taxonomy_payload,
        dataset_envelope=envelope,
        canonical_dataset_fingerprint=canonical_dataset.fingerprint,
        location_count=len(location_entries),
        category_count=len(category_entries),
        daypart_count=len(daypart_entries),
    )


def attach_governed_taxonomy_to_dataset(
    *,
    dataset_payload: Mapping[str, Any],
    taxonomy_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach one verified engineering taxonomy without changing dataset identity."""

    canonical_dataset = EmbeddingBenchmarkDataset.from_mapping(dataset_payload)
    loaded = load_governed_constraint_taxonomy(
        taxonomy_payload,
        allow_engineering_scope=True,
        require_native_human_review=False,
    )
    if loaded.lineage.get("source_dataset_fingerprint") != (
        canonical_dataset.fingerprint
    ):
        raise ValueError(
            "Governed taxonomy was built for a different benchmark dataset."
        )
    envelope = deepcopy(dict(dataset_payload))
    envelope["governed_constraint_taxonomy"] = deepcopy(
        dict(taxonomy_payload)
    )
    envelope_dataset = EmbeddingBenchmarkDataset.from_mapping(envelope)
    if envelope_dataset.fingerprint != canonical_dataset.fingerprint:
        raise ValueError("Taxonomy attachment changed canonical dataset identity.")
    return envelope


def _taxonomy_entries(
    value: Mapping[str, Any],
    key: str,
) -> tuple[ConstraintTaxonomyEntry, ...]:
    raw_entries = value.get(key)
    if not isinstance(raw_entries, Sequence) or isinstance(
        raw_entries,
        (str, bytes),
    ):
        raise ValueError(f"Governed taxonomy {key} must be a sequence.")
    entries: list[ConstraintTaxonomyEntry] = []
    for position, item in enumerate(raw_entries):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Governed taxonomy {key}[{position}] must be a mapping."
            )
        entries.append(
            ConstraintTaxonomyEntry(
                canonical_value=str(item.get("canonical_value") or ""),
                aliases=_string_sequence(item.get("aliases"), f"{key}.aliases"),
                descriptions=_string_sequence(
                    item.get("descriptions"),
                    f"{key}.descriptions",
                ),
            )
        )
    return tuple(entries)


def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence.")
    return tuple(str(item) for item in value)


def _validate_engineering_language_pack(value: Mapping[str, Any]) -> None:
    if value.get("review_status") != "approved":
        raise ValueError("Language pack is not approved for engineering use.")
    if value.get("translation_review_completed") is not True:
        raise ValueError("Language-pack translation review is incomplete.")
    _require_text(value, "pack_id")
    _require_text(value, "pack_version")
    _require_text(value, "reviewed_by")

    lineage_value = value.get("lineage")
    if not isinstance(lineage_value, Mapping):
        raise ValueError("Language-pack lineage is required.")
    lineage = dict(lineage_value)
    if lineage.get("approval_scope") != ENGINEERING_LANGUAGE_PACK_SCOPE:
        raise ValueError("Language-pack approval scope is invalid.")
    if lineage.get("native_human_review_completed") is not False:
        raise ValueError("Engineering language pack must disclose pending review.")
    if lineage.get("machine_translation_auto_approved") is not False:
        raise ValueError("Machine translation cannot be auto-approved.")
    if lineage.get("requires_native_language_review") is not True:
        raise ValueError("Language pack must require native-language review.")
    if lineage.get("requires_named_human_review") is not True:
        raise ValueError("Language pack must require named human review.")
    if lineage.get("production_certification_status") != PENDING_NATIVE_SIGNOFF:
        raise ValueError("Language pack must remain pending native-human signoff.")
    _require_text(lineage, "review_method")

    language_order = value.get("language_order")
    languages = value.get("languages")
    if not isinstance(language_order, Sequence) or isinstance(
        language_order,
        (str, bytes),
    ):
        raise ValueError("Language-pack language_order must be a sequence.")
    if not isinstance(languages, Mapping):
        raise ValueError("Language-pack languages must be a mapping.")
    normalized_order = tuple(" ".join(str(item).split()) for item in language_order)
    if (
        not normalized_order
        or any(not item for item in normalized_order)
        or len(normalized_order) != len(set(normalized_order))
    ):
        raise ValueError("Language-pack language_order is empty or duplicated.")
    if set(normalized_order) != {str(key) for key in languages}:
        raise ValueError("Language order and language definitions differ.")
    language_count = lineage.get("language_count")
    if language_count is not None and int(language_count) != len(normalized_order):
        raise ValueError("Language-pack language_count does not match definitions.")


def _require_text(value: Mapping[str, Any], key: str) -> str:
    output = " ".join(str(value.get(key) or "").split())
    if not output:
        raise ValueError(f"{key} is required.")
    return output


def _require_sha256(value: Mapping[str, Any], key: str) -> str:
    return _normalized_sha256(value.get(key), key)


def _normalized_sha256(value: Any, field_name: str) -> str:
    output = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", output):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return output
