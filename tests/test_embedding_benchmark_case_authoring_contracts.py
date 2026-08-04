from __future__ import annotations

import copy

import pytest

from app.models.embedding_benchmark_case_authoring_contracts import (
    EmbeddingBenchmarkLanguagePack,
)


def _language_pack_payload() -> dict:
    languages = {}
    for language in ("en", "fr", "es", "hi", "bn"):
        languages[language] = {
            "supported_query_template": (
                "Find {category} in {location} during {daypart}."
            ),
            "unsupported_query_template": "Find in {location}.",
            "category_labels": {
                "restaurant": f"{language}-restaurant",
            },
            "daypart_labels": {
                "morning": f"{language}-morning",
                "afternoon": f"{language}-afternoon",
                "evening": f"{language}-evening",
                "night": f"{language}-night",
            },
        }
    return {
        "pack_id": "language-pack",
        "pack_version": "immutable-v1",
        "review_status": "approved",
        "reviewed_by": "language-reviewer",
        "reviewed_at": "2026-08-04T12:00:00+00:00",
        "translation_review_completed": True,
        "language_order": ["en", "fr", "es", "hi", "bn"],
        "languages": languages,
        "unsupported_location_labels": {
            "brindlehaven": {
                language: "Brindlehaven"
                for language in languages
            },
            "cedar_harbor": {
                language: "Cedar Harbor"
                for language in languages
            },
            "dovewood": {
                language: "Dovewood"
                for language in languages
            },
            "rivermere": {
                language: "Rivermere"
                for language in languages
            },
        },
        "lineage": {"purpose": "unit-test"},
    }


def test_language_pack_requires_named_approved_translation_review():
    pending = _language_pack_payload()
    pending["review_status"] = "pending"
    with pytest.raises(ValueError, match="not approved"):
        EmbeddingBenchmarkLanguagePack.from_mapping(pending)

    missing_reviewer = _language_pack_payload()
    missing_reviewer["reviewed_by"] = ""
    with pytest.raises(ValueError, match="named reviewer"):
        EmbeddingBenchmarkLanguagePack.from_mapping(missing_reviewer)

    incomplete = _language_pack_payload()
    incomplete["translation_review_completed"] = False
    with pytest.raises(ValueError, match="incomplete"):
        EmbeddingBenchmarkLanguagePack.from_mapping(incomplete)


def test_language_pack_rejects_template_and_marker_token_leakage():
    invalid_template = _language_pack_payload()
    invalid_template["languages"]["en"][
        "supported_query_template"
    ] = "Find {category} in {location}."
    with pytest.raises(ValueError, match="exactly these placeholders"):
        EmbeddingBenchmarkLanguagePack.from_mapping(invalid_template)

    marker = copy.deepcopy(_language_pack_payload())
    marker["unsupported_location_labels"][
        "unsupported_test_city"
    ] = {
        language: "Unsupported Test City"
        for language in marker["language_order"]
    }
    with pytest.raises(ValueError, match="marker tokens"):
        EmbeddingBenchmarkLanguagePack.from_mapping(marker)


def test_language_pack_is_content_addressed_and_normalized():
    pack = EmbeddingBenchmarkLanguagePack.from_mapping(
        _language_pack_payload()
    )

    assert pack.review_status == "approved"
    assert pack.reviewed_by == "language_reviewer"
    assert len(pack.fingerprint) == 64
    assert pack.to_dict()["translation_review_completed"] is True
