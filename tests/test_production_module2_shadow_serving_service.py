from __future__ import annotations

from dataclasses import dataclass

from app.models.audience_feature_contracts import stable_digest
from app.models.production_module2_completion_contracts import (
    Module2ShadowObservation,
    Module2ShadowPolicy,
)
from app.services.production_module2_shadow_serving_service import (
    InMemoryShadowObservationSink,
    ProductionModule2ReleaseGateService,
    ProductionModule2ShadowServingService,
    evaluate_module2_shadow_readiness,
)


@dataclass
class Request:
    tenant_id: str = "tenant-a"
    query_fingerprint: str = "a" * 64
    query_text: str = "raw query must not be stored"


class Adapter:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def retrieve(self, _request):
        self.calls += 1
        return self.result


def _ready_result():
    return {
        "status": "retrieval_ready_for_human_review",
        "reason_code": "retrieval_ready",
        "constraints": {
            "constraints": {
                "locations": {"status": "resolved", "values": ["montreal"]},
                "categories": {"status": "resolved", "values": ["cafe"]},
                "dayparts": {"status": "resolved", "values": ["evening"]},
            }
        },
        "selected_candidates": [
            {
                "location_name": "montreal",
                "primary_poi_type": "cafe",
                "created_day_part": "evening",
                "retrieval_rank": 1,
                "feature_ids": {"primary": "must-not-be-stored"},
            }
        ],
    }


def test_shadow_disabled_calls_neither_adapter():
    incumbent = Adapter(_ready_result())
    candidate = Adapter(_ready_result())
    result = ProductionModule2ShadowServingService(
        incumbent=incumbent,
        candidate=candidate,
        sink=InMemoryShadowObservationSink(),
        enabled=False,
    ).compare(Request())
    assert result["status"] == "shadow_serving_disabled"
    assert incumbent.calls == 0
    assert candidate.calls == 0


def test_shadow_comparison_stores_only_safe_fingerprints_and_never_routes_candidate():
    sink = InMemoryShadowObservationSink()
    service = ProductionModule2ShadowServingService(
        incumbent=Adapter(_ready_result()),
        candidate=Adapter(_ready_result()),
        sink=sink,
        enabled=True,
        timer_fn=iter([1.0, 1.01, 2.0, 2.02]).__next__,
    )
    result = service.compare(Request())
    assert result["agreement"] is True
    assert result["exact_signature_match"] is True
    assert result["candidate_selection_changed"] is False
    assert result["candidate_output_returned_to_user"] is False
    assert result["production_routing_enabled"] is False
    assert len(sink.observations) == 1
    text = str(sink.observations[0].to_safe_dict())
    assert "raw query must not be stored" not in text
    assert "must-not-be-stored" not in text


def test_shadow_readiness_and_release_gate_are_recommendation_only():
    observations = tuple(
        Module2ShadowObservation(
            tenant_id="tenant-a",
            request_fingerprint=stable_digest({"case": index}),
            observed_at="2026-08-05T00:00:00+00:00",
            incumbent_signature="b" * 64,
            candidate_signature="b" * 64,
            incumbent_status="ready",
            candidate_status="ready",
            incumbent_latency_ms=10.0,
            candidate_latency_ms=20.0,
            candidate_error=False,
            safety_divergence=False,
            agreement=True,
        )
        for index in range(20)
    )
    shadow = evaluate_module2_shadow_readiness(
        observations,
        policy=Module2ShadowPolicy(minimum_samples=20),
    )
    assert shadow["shadow_release_ready"] is True
    assert shadow["production_routing_enabled"] is False

    release = ProductionModule2ReleaseGateService().evaluate(
        certification_report={
            "production_certification_ready": True,
            "report_fingerprint": "c" * 64,
        },
        index_record={"status": "shadow", "manifest_fingerprint": "d" * 64},
        shadow_report=shadow,
        operator_approval_recorded=True,
    )
    assert release["release_recommendation_ready"] is True
    assert release["index_activation_performed"] is False
    assert release["production_routing_enabled"] is False



def test_shadow_decision_agreement_allows_safe_reranking_drift():
    incumbent = _ready_result()
    candidate = _ready_result()
    candidate["reason_code"] = "verified_full_constraint_match_available"
    candidate["selected_candidates"][0]["retrieval_rank"] = 2

    result = ProductionModule2ShadowServingService(
        incumbent=Adapter(incumbent),
        candidate=Adapter(candidate),
        sink=InMemoryShadowObservationSink(),
        enabled=True,
    ).compare(Request())

    assert result["agreement"] is True
    assert result["exact_signature_match"] is False
    assert result["candidate_selection_changed"] is True
    assert result["safety_divergence"] is False


def test_shadow_blocked_reason_codes_share_safe_decision_agreement():
    incumbent = {
        "status": "blocked",
        "reason_code": "blocked_no_safe_exact_match",
        "constraints": {
            "constraints": {
                "locations": {"status": "unsupported", "values": []},
                "categories": {"status": "not_requested", "values": []},
                "dayparts": {"status": "not_requested", "values": []},
            }
        },
        "selected_candidates": [],
    }
    candidate = {
        "status": "blocked",
        "reason_code": "location_requires_clarification",
        "constraints": {
            "constraints": {
                "locations": {
                    "status": "requires_clarification",
                    "values": [],
                },
                "categories": {"status": "not_requested", "values": []},
                "dayparts": {"status": "not_requested", "values": []},
            }
        },
        "selected_candidates": [],
    }

    result = ProductionModule2ShadowServingService(
        incumbent=Adapter(incumbent),
        candidate=Adapter(candidate),
        sink=InMemoryShadowObservationSink(),
        enabled=True,
    ).compare(Request())

    assert result["agreement"] is True
    assert result["exact_signature_match"] is False
    assert result["candidate_selection_changed"] is True
    assert result["safety_divergence"] is False


def test_shadow_decision_disagreement_still_blocks_release_agreement():
    blocked = {
        "status": "blocked",
        "reason_code": "location_requires_clarification",
        "constraints": {
            "constraints": {
                "locations": {
                    "status": "requires_clarification",
                    "values": [],
                },
                "categories": {"status": "not_requested", "values": []},
                "dayparts": {"status": "not_requested", "values": []},
            }
        },
        "selected_candidates": [],
    }

    candidate_blocked = ProductionModule2ShadowServingService(
        incumbent=Adapter(_ready_result()),
        candidate=Adapter(blocked),
        sink=InMemoryShadowObservationSink(),
        enabled=True,
    ).compare(Request())
    assert candidate_blocked["agreement"] is False
    assert candidate_blocked["safety_divergence"] is False

    unsafe_candidate_ready = ProductionModule2ShadowServingService(
        incumbent=Adapter(blocked),
        candidate=Adapter(_ready_result()),
        sink=InMemoryShadowObservationSink(),
        enabled=True,
    ).compare(Request())
    assert unsafe_candidate_ready["agreement"] is False
    assert unsafe_candidate_ready["safety_divergence"] is True
