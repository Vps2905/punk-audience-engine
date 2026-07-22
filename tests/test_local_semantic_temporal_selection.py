"""
Focused temporal-selection regression tests.

These validate that the semantic intent service correctly:
  - infers evening for phrases describing end-of-work scenarios;
  - abstains when no temporal intent is present;
  - preserves explicit temporal keywords;
  - handles mixed temporal dimensions;
  - does not infer time from venue-category correlations alone.

No exact phrase tables or prompt-specific synonyms are used.
"""

import pandas as pd
import pytest

from app.services.local_semantic_intent_service import (
    LocalSemanticIntentService,
)


# ── helpers ────────────────────────────────────────────────

def _rag_context(
    locations=None,
    pois=None,
    dayparts=None,
):
    return {
        "available_locations": locations or [
            "montreal",
            "vancouver",
            "oslo",
            "toronto",
            "london",
            "new york",
        ],
        "requested_location_candidates": [],
        "available_poi_types": pois or [
            "cafe",
            "restaurant",
            "shopping_mall",
            "gym",
            "bar",
            "office",
        ],
        "available_dayparts": dayparts or [
            "morning",
            "afternoon",
            "evening",
            "night",
            "weekday",
            "weekend",
        ],
    }


def _resolve(prompt, **rag_overrides):
    svc = LocalSemanticIntentService()
    return svc.resolve(
        prompt=prompt,
        rag_context=_rag_context(**rag_overrides),
    )


# ── Positive semantic temporal cases ──────────────────────

class TestEndOfWorkdayInference:
    """Phrases that semantically describe the end of the working
    day should resolve to evening without exact-phrase rules."""

    def test_heading_home_after_a_long_day_at_work(self):
        result = _resolve(
            "Find people heading home after a long day "
            "at work near Vancouver."
        )
        assert "evening" in result["dayparts"]

    def test_once_the_workday_is_done(self):
        result = _resolve(
            "Target shoppers once the workday is done "
            "in Toronto."
        )
        assert "evening" in result["dayparts"]

    def test_post_office_hours_shopping(self):
        result = _resolve(
            "Find customers who stop at malls after "
            "office hours in New York."
        )
        assert "evening" in result["dayparts"]


# ── Negative abstention cases ─────────────────────────────

class TestTemporalAbstention:
    """Requests without temporal meaning must return empty
    dayparts.  A venue type alone must not imply a time."""

    def test_coffee_customers_in_montreal(self):
        result = _resolve(
            "Find coffee shop customers near Montreal."
        )
        assert result["dayparts"] == []

    def test_shoppers_in_vancouver(self):
        result = _resolve(
            "Find shoppers visiting malls in Vancouver."
        )
        assert result["dayparts"] == []

    def test_gym_visitors_near_oslo(self):
        result = _resolve(
            "Find people who visit the gym near Oslo."
        )
        assert result["dayparts"] == []

    def test_restaurant_visitors_in_toronto(self):
        result = _resolve(
            "Reach restaurant visitors in Toronto."
        )
        assert result["dayparts"] == []


# ── Explicit temporal cases ───────────────────────────────

class TestExplicitTemporalKeywords:
    """When the user explicitly names a time period, the
    resolver should preserve it via direct extraction."""

    def test_morning_coffee(self):
        result = _resolve(
            "coffee visitors during the morning in Montreal"
        )
        assert "morning" in result["dayparts"]

    def test_weekend_shoppers(self):
        result = _resolve(
            "shoppers on weekends in Vancouver"
        )
        assert "weekend" in result["dayparts"]

    def test_night_restaurant(self):
        result = _resolve(
            "restaurant visitors at night in Toronto"
        )
        assert "night" in result["dayparts"]

    def test_afternoon_gym(self):
        result = _resolve(
            "gym visitors in the afternoon near Oslo"
        )
        assert "afternoon" in result["dayparts"]

    def test_weekday_office(self):
        result = _resolve(
            "weekday office visitors in London"
        )
        assert "weekday" in result["dayparts"]


# ── Mixed temporal dimensions ─────────────────────────────

class TestMixedTemporalDimensions:
    """When both a clock-time and a calendar qualifier are
    explicitly stated, both should be preserved."""

    def test_weekday_evening(self):
        result = _resolve(
            "weekday evening coffee customers in Montreal"
        )
        assert "weekday" in result["dayparts"]
        assert "evening" in result["dayparts"]

    def test_weekend_morning(self):
        result = _resolve(
            "weekend morning shoppers in Vancouver"
        )
        assert "weekend" in result["dayparts"]
        assert "morning" in result["dayparts"]

    def test_weekday_night(self):
        result = _resolve(
            "weekday night restaurant visitors in Toronto"
        )
        assert "weekday" in result["dayparts"]
        assert "night" in result["dayparts"]


# ── False-positive protection ─────────────────────────────

class TestVenueCategoryDoesNotImplyTime:
    """Category context alone must not create time intent.
    These use realistic full-sentence prompts to validate
    that venue associations do not leak through."""

    def test_cafe_alone(self):
        result = _resolve(
            "Find cafe visitors in Montreal."
        )
        assert result["dayparts"] == []

    def test_bar_alone(self):
        result = _resolve(
            "Find bar visitors in Toronto."
        )
        assert result["dayparts"] == []


# ── Safe environment parsing ──────────────────────────────

class TestSafeEnvParsing:
    """Malformed environment variables must not crash the
    service."""

    def test_malformed_float_env(self, monkeypatch):
        monkeypatch.setenv(
            "LOCAL_SEMANTIC_TEMPORAL_OVERRIDE_MARGIN",
            "not_a_number",
        )
        result = _resolve(
            "Find coffee shop visitors near Montreal."
        )
        assert result["dayparts"] == []

    def test_malformed_int_env(self, monkeypatch):
        monkeypatch.setenv(
            "LOCAL_SEMANTIC_TEMPORAL_MIN_WORDS",
            "oops",
        )
        # Should not crash; falls back to default.
        result = _resolve(
            "Find coffee shop visitors near Montreal."
        )
        assert isinstance(result["dayparts"], list)
