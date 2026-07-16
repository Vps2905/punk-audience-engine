import json

import pandas as pd

from app.agents.hybrid_semantic_intent_agent import (
    HybridSemanticIntentAgent,
)


def _cohorts() -> pd.DataFrame:
    rows = []

    for index in range(60):
        rows.append(
            {
                "location_name": f"city_{index}",
                "primary_poi_type": "restaurant",
                "created_day_part": "morning",
                "quality_score": 0.7,
                "total_maid_volume": 2000,
                "privacy_status": "passed",
            }
        )

    rows.append(
        {
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
            "quality_score": 0.9,
            "total_maid_volume": 3000,
            "privacy_status": "passed",
        }
    )

    return pd.DataFrame(rows)


def test_llm_context_is_compact_and_prompt_relevant():
    agent = HybridSemanticIntentAgent()
    full_context = agent._build_safe_rag_context(
        _cohorts()
    )

    compact = agent._build_llm_rag_context(
        prompt=(
            "Reach professionals grabbing espresso "
            "near Montreal in the evening."
        ),
        rag_context=full_context,
        max_context_items=12,
    )

    assert len(
        full_context["safe_available_combinations"]
    ) == 61
    assert len(
        compact["safe_available_combinations"]
    ) <= 12

    assert any(
        combo["location_name"] == "montreal"
        and combo["primary_poi_type"] == "cafe"
        and combo["created_day_part"] == "evening"
        for combo in compact[
            "safe_available_combinations"
        ]
    )

    assert (
        compact["context_compaction"][
            "full_safe_combination_count"
        ]
        == 61
    )
    assert (
        compact["context_compaction"][
            "sent_safe_combination_count"
        ]
        <= 12
    )


def test_resolve_sends_compact_context_but_uses_llm(
    monkeypatch,
):
    captured = {}

    def fake_llm(messages, config):
        captured["messages"] = messages
        captured["config"] = config
        return json.dumps(
            {
                "business_intent": "cafe",
                "audience_goal": "reach cafe visitors",
                "locations": ["montreal"],
                "requested_categories": ["cafe"],
                "matched_available_poi_types": ["cafe"],
                "poi_terms": ["espresso", "cafe"],
                "dayparts": ["evening"],
                "quality_intent": "high",
                "fallback_tolerance": "strict_exact_first",
                "data_gap_likely": False,
                "confidence_score": 0.95,
                "reasoning_summary": (
                    "Montreal cafe intent in evening."
                ),
            }
        )

    monkeypatch.setenv("ENABLE_LLM_INTENT", "true")
    monkeypatch.setenv(
        "LLM_INTENT_PROVIDER",
        "openrouter",
    )
    monkeypatch.setenv(
        "LLM_INTENT_MODEL",
        "test-model",
    )
    monkeypatch.setenv(
        "LLM_INTENT_MAX_CONTEXT_ITEMS",
        "10",
    )
    monkeypatch.setenv(
        "LLM_INTENT_MIN_CONFIDENCE",
        "0.65",
    )

    result = HybridSemanticIntentAgent(
        llm_client=fake_llm
    ).resolve(
        prompt=(
            "Reach professionals grabbing espresso "
            "near Montreal in the evening."
        ),
        safe_cohorts=_cohorts(),
    )

    assert result["llm_used"] is True
    assert result["resolver_mode"] == "llm_rag_primary"
    assert result["locations"] == ["montreal"]
    assert result["canonical_categories"] == ["cafe"]
    assert result["dayparts"] == ["evening"]

    user_payload = json.loads(
        captured["messages"][1]["content"]
    )
    sent_context = user_payload["safe_rag_context"]

    assert len(
        sent_context["safe_available_combinations"]
    ) <= 10
    assert (
        sent_context["context_compaction"][
            "full_safe_combination_count"
        ]
        == 61
    )
