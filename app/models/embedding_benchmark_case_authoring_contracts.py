from __future__ import annotations

import json
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timezone
from typing import Any

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    parse_utc_datetime,
    stable_digest,
)

REQUIRED_SUPPORTED_TEMPLATE_FIELDS = {
    "category",
    "daypart",
    "location",
}
REQUIRED_UNSUPPORTED_TEMPLATE_FIELDS = {"location"}
FORBIDDEN_UNSUPPORTED_LOCATION_TOKENS = {
    "example",
    "placeholder",
    "test",
    "unknown",
    "unsupported",
}


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def _json_safe_mapping(
    payload: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(dict(payload), allow_nan=False)
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must contain finite JSON values."
        ) from None


def _template_fields(value: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(value):
        if field_name:
            fields.add(field_name)
    return fields


def _validated_template(
    value: Any,
    *,
    label: str,
    required_fields: set[str],
) -> str:
    template = _required_text(value, label)
    fields = _template_fields(template)
    if fields != required_fields:
        raise ValueError(
            f"{label} must contain exactly these placeholders: "
            + ", ".join(sorted(required_fields))
        )
    try:
        template.format(
            category="category",
            daypart="daypart",
            location="location",
        )
    except (KeyError, ValueError):
        raise ValueError(f"{label} is not a valid format template.") from None
    return template


def _normalized_label_mapping(
    payload: Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{label} must be a non-empty object.")
    values: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        key = normalize_taxonomy_value(raw_key)
        if not key:
            raise ValueError(f"{label} contains an invalid taxonomy key.")
        if key in values:
            raise ValueError(f"{label} contains duplicate normalized keys.")
        values[key] = _required_text(raw_value, f"{label}.{key}")
    return {key: values[key] for key in sorted(values)}


@dataclass(frozen=True)
class EmbeddingBenchmarkLanguageDefinition:
    language: str
    supported_query_template: str
    unsupported_query_template: str
    category_labels: Mapping[str, str]
    daypart_labels: Mapping[str, str]

    def __post_init__(self) -> None:
        language = normalize_taxonomy_value(self.language)
        if not language:
            raise ValueError("language is required.")
        object.__setattr__(self, "language", language)
        object.__setattr__(
            self,
            "supported_query_template",
            _validated_template(
                self.supported_query_template,
                label=(
                    f"languages.{language}.supported_query_template"
                ),
                required_fields=REQUIRED_SUPPORTED_TEMPLATE_FIELDS,
            ),
        )
        object.__setattr__(
            self,
            "unsupported_query_template",
            _validated_template(
                self.unsupported_query_template,
                label=(
                    f"languages.{language}.unsupported_query_template"
                ),
                required_fields=REQUIRED_UNSUPPORTED_TEMPLATE_FIELDS,
            ),
        )
        object.__setattr__(
            self,
            "category_labels",
            _normalized_label_mapping(
                self.category_labels,
                label=f"languages.{language}.category_labels",
            ),
        )
        object.__setattr__(
            self,
            "daypart_labels",
            _normalized_label_mapping(
                self.daypart_labels,
                label=f"languages.{language}.daypart_labels",
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        *,
        language: str,
        payload: Mapping[str, Any],
    ) -> EmbeddingBenchmarkLanguageDefinition:
        if not isinstance(payload, Mapping):
            raise TypeError("Benchmark language definition must be an object.")
        return cls(
            language=language,
            supported_query_template=payload.get(
                "supported_query_template",
                "",
            ),
            unsupported_query_template=payload.get(
                "unsupported_query_template",
                "",
            ),
            category_labels=(
                payload.get("category_labels")
                if isinstance(payload.get("category_labels"), Mapping)
                else {}
            ),
            daypart_labels=(
                payload.get("daypart_labels")
                if isinstance(payload.get("daypart_labels"), Mapping)
                else {}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_query_template": self.supported_query_template,
            "unsupported_query_template": self.unsupported_query_template,
            "category_labels": dict(self.category_labels),
            "daypart_labels": dict(self.daypart_labels),
        }


@dataclass(frozen=True)
class EmbeddingBenchmarkLanguagePack:
    pack_id: str
    pack_version: str
    review_status: str
    reviewed_by: str
    reviewed_at: Any
    translation_review_completed: bool
    language_order: tuple[str, ...]
    languages: Mapping[str, EmbeddingBenchmarkLanguageDefinition]
    unsupported_location_labels: Mapping[str, Mapping[str, str]]
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pack_id",
            _required_text(self.pack_id, "pack_id"),
        )
        object.__setattr__(
            self,
            "pack_version",
            _required_text(self.pack_version, "pack_version"),
        )
        if normalize_taxonomy_value(self.review_status) != "approved":
            raise ValueError("Benchmark language pack review is not approved.")
        object.__setattr__(self, "review_status", "approved")
        reviewer = normalize_taxonomy_value(self.reviewed_by)
        if not reviewer:
            raise ValueError("Benchmark language pack requires a named reviewer.")
        object.__setattr__(self, "reviewed_by", reviewer)
        reviewed_at = parse_utc_datetime(self.reviewed_at)
        if reviewed_at is None:
            raise ValueError("Benchmark language pack requires reviewed_at.")
        object.__setattr__(
            self,
            "reviewed_at",
            reviewed_at.astimezone(timezone.utc),
        )
        if not self.translation_review_completed:
            raise ValueError(
                "Benchmark language pack translation review is incomplete."
            )
        object.__setattr__(self, "translation_review_completed", True)

        language_order = tuple(
            normalize_taxonomy_value(value)
            for value in self.language_order
            if normalize_taxonomy_value(value)
        )
        if not language_order:
            raise ValueError("Benchmark language_order cannot be empty.")
        if len(language_order) != len(set(language_order)):
            raise ValueError("Benchmark language_order must be unique.")
        object.__setattr__(self, "language_order", language_order)

        languages = dict(self.languages)
        if set(languages) != set(language_order):
            raise ValueError(
                "Benchmark language definitions must match language_order."
            )
        for language, definition in languages.items():
            if language != definition.language:
                raise ValueError(
                    "Benchmark language definition key does not match language."
                )
        object.__setattr__(
            self,
            "languages",
            {key: languages[key] for key in language_order},
        )

        unsupported = self._validated_unsupported_locations(
            self.unsupported_location_labels,
            language_order=language_order,
        )
        object.__setattr__(self, "unsupported_location_labels", unsupported)
        object.__setattr__(
            self,
            "lineage",
            _json_safe_mapping(self.lineage, "language pack lineage"),
        )

    @staticmethod
    def _validated_unsupported_locations(
        payload: Mapping[str, Mapping[str, str]],
        *,
        language_order: Sequence[str],
    ) -> dict[str, dict[str, str]]:
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError(
                "Benchmark unsupported_location_labels must be non-empty."
            )
        normalized: dict[str, dict[str, str]] = {}
        for raw_location, raw_labels in payload.items():
            location = normalize_taxonomy_value(raw_location)
            if not location:
                raise ValueError("Unsupported location key is invalid.")
            tokens = set(location.split("_"))
            if tokens.intersection(FORBIDDEN_UNSUPPORTED_LOCATION_TOKENS):
                raise ValueError(
                    "Unsupported locations cannot contain marker tokens."
                )
            if location in normalized:
                raise ValueError("Unsupported location keys must be unique.")
            if not isinstance(raw_labels, Mapping):
                raise TypeError(
                    "Unsupported location labels must be language objects."
                )
            labels = {
                normalize_taxonomy_value(language): _required_text(
                    value,
                    f"unsupported_location_labels.{location}.{language}",
                )
                for language, value in raw_labels.items()
            }
            if set(labels) != set(language_order):
                raise ValueError(
                    "Every unsupported location requires one label per language."
                )
            for value in labels.values():
                marker_tokens = {
                    normalize_taxonomy_value(token)
                    for token in value.replace("-", " ").split()
                }
                if marker_tokens.intersection(
                    FORBIDDEN_UNSUPPORTED_LOCATION_TOKENS
                ):
                    raise ValueError(
                        "Unsupported location labels cannot leak marker tokens."
                    )
            normalized[location] = {
                language: labels[language]
                for language in language_order
            }
        return {
            key: normalized[key]
            for key in sorted(normalized)
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> EmbeddingBenchmarkLanguagePack:
        if not isinstance(payload, Mapping):
            raise TypeError("Benchmark language pack must be an object.")
        languages_payload = payload.get("languages")
        if not isinstance(languages_payload, Mapping):
            raise TypeError("Benchmark languages must be an object.")
        definitions = {
            normalize_taxonomy_value(language): (
                EmbeddingBenchmarkLanguageDefinition.from_mapping(
                    language=language,
                    payload=value,
                )
            )
            for language, value in languages_payload.items()
            if isinstance(value, Mapping)
        }
        if len(definitions) != len(languages_payload):
            raise ValueError("Every benchmark language must be an object.")
        order_payload = payload.get("language_order")
        if not isinstance(order_payload, list):
            raise TypeError("Benchmark language_order must be an array.")
        unsupported = payload.get("unsupported_location_labels")
        return cls(
            pack_id=payload.get("pack_id", ""),
            pack_version=payload.get("pack_version", ""),
            review_status=payload.get("review_status", ""),
            reviewed_by=payload.get("reviewed_by", ""),
            reviewed_at=payload.get("reviewed_at"),
            translation_review_completed=bool(
                payload.get("translation_review_completed", False)
            ),
            language_order=tuple(order_payload),
            languages=definitions,
            unsupported_location_labels=(
                unsupported
                if isinstance(unsupported, Mapping)
                else {}
            ),
            lineage=(
                payload.get("lineage")
                if isinstance(payload.get("lineage"), Mapping)
                else {}
            ),
        )

    @classmethod
    def from_json(
        cls,
        content: str,
    ) -> EmbeddingBenchmarkLanguagePack:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(
                "Benchmark language pack is not valid JSON."
            ) from None
        return cls.from_mapping(payload)

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat(),
            "translation_review_completed": True,
            "language_order": list(self.language_order),
            "languages": {
                key: self.languages[key].to_dict()
                for key in self.language_order
            },
            "unsupported_location_labels": {
                key: dict(self.unsupported_location_labels[key])
                for key in sorted(self.unsupported_location_labels)
            },
            "lineage": dict(self.lineage),
        }
