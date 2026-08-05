from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    stable_digest,
)
from app.models.production_module2_completion_contracts import (
    Module2ShadowObservation,
    Module2ShadowPolicy,
)


class RetrievalAdapter(Protocol):
    def retrieve(self, request: Any) -> Mapping[str, Any]: ...


class ShadowObservationSink(Protocol):
    def write(self, observation: Module2ShadowObservation) -> None: ...


class InMemoryShadowObservationSink:
    def __init__(self) -> None:
        self.observations: list[Module2ShadowObservation] = []

    def write(self, observation: Module2ShadowObservation) -> None:
        self.observations.append(observation)


class ProductionModule2ShadowServingService:
    """Run retrieval-only candidate logic beside the incumbent without routing.

    Raw query text is passed only to retrieval adapters and is never written to
    the observation sink. Candidate output cannot replace incumbent output here.
    """

    def __init__(
        self,
        *,
        incumbent: RetrievalAdapter,
        candidate: RetrievalAdapter,
        sink: ShadowObservationSink,
        enabled: bool = False,
        now_fn=None,
        timer_fn=None,
    ) -> None:
        self._incumbent = incumbent
        self._candidate = candidate
        self._sink = sink
        self._enabled = bool(enabled)
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._timer_fn = timer_fn or time.perf_counter

    def compare(self, request: Any) -> dict[str, Any]:
        request_fingerprint = getattr(request, "query_fingerprint", None)
        tenant_id = getattr(request, "tenant_id", None)
        if not request_fingerprint or not tenant_id:
            raise ValueError(
                "Shadow serving requires tenant_id and query_fingerprint."
            )
        if not self._enabled:
            return {
                "status": "shadow_serving_disabled",
                "tenant_id": str(tenant_id),
                "request_fingerprint": str(request_fingerprint),
                "incumbent_called": False,
                "candidate_called": False,
                "production_routing_enabled": False,
                "automatic_proposal_creation_enabled": False,
                "activation_or_export_performed": False,
            }

        incumbent_started = self._timer_fn()
        incumbent_result = dict(self._incumbent.retrieve(request))
        incumbent_latency = max(
            0.0,
            (self._timer_fn() - incumbent_started) * 1000.0,
        )

        candidate_error = False
        candidate_started = self._timer_fn()
        try:
            candidate_result = dict(self._candidate.retrieve(request))
        except Exception:
            candidate_error = True
            candidate_result = {
                "status": "shadow_candidate_error",
                "reason_code": "shadow_candidate_exception",
            }
        candidate_latency = max(
            0.0,
            (self._timer_fn() - candidate_started) * 1000.0,
        )

        incumbent_signature = self._result_signature(incumbent_result)
        candidate_signature = self._result_signature(candidate_result)
        incumbent_status = self._status(incumbent_result)
        candidate_status = self._status(candidate_result)
        exact_signature_match = (
            incumbent_signature == candidate_signature
        )
        incumbent_decision_signature = self._decision_signature(
            incumbent_result
        )
        candidate_decision_signature = self._decision_signature(
            candidate_result
        )
        agreement = (
            incumbent_decision_signature
            == candidate_decision_signature
        )
        safety_divergence = self._safety_divergence(
            incumbent_result,
            candidate_result,
            candidate_error=candidate_error,
        )

        observation = Module2ShadowObservation(
            tenant_id=str(tenant_id),
            request_fingerprint=str(request_fingerprint),
            observed_at=self._now_fn(),
            incumbent_signature=incumbent_signature,
            candidate_signature=candidate_signature,
            incumbent_status=incumbent_status,
            candidate_status=candidate_status,
            incumbent_latency_ms=incumbent_latency,
            candidate_latency_ms=candidate_latency,
            candidate_error=candidate_error,
            safety_divergence=safety_divergence,
            agreement=agreement,
        )
        self._sink.write(observation)
        return {
            "status": "shadow_comparison_recorded",
            "tenant_id": observation.tenant_id,
            "request_fingerprint": observation.request_fingerprint,
            "agreement": agreement,
            "exact_signature_match": exact_signature_match,
            "candidate_selection_changed": (
                agreement and not exact_signature_match
            ),
            "safety_divergence": safety_divergence,
            "candidate_error": candidate_error,
            "incumbent_latency_ms": round(incumbent_latency, 6),
            "candidate_latency_ms": round(candidate_latency, 6),
            "candidate_output_returned_to_user": False,
            "production_routing_enabled": False,
            "automatic_proposal_creation_enabled": False,
            "activation_or_export_performed": False,
            "raw_query_stored": False,
            "raw_identifiers_stored": False,
        }

    @staticmethod
    def _status(result: Mapping[str, Any]) -> str:
        return str(result.get("status") or result.get("reason_code") or "unknown")

    @classmethod
    def _result_signature(cls, result: Mapping[str, Any]) -> str:
        constraints = dict(result.get("constraints") or {})
        nested_constraints = dict(constraints.get("constraints") or {})
        selected = []
        for candidate in list(result.get("selected_candidates") or ())[:5]:
            if not isinstance(candidate, Mapping):
                continue
            selected.append(
                {
                    "location_name": candidate.get("location_name"),
                    "primary_poi_type": candidate.get("primary_poi_type"),
                    "created_day_part": candidate.get("created_day_part"),
                    "retrieval_rank": candidate.get("retrieval_rank"),
                }
            )
        safe = {
            "status": cls._status(result),
            "reason_code": result.get("reason_code"),
            "constraints": {
                dimension: {
                    "status": dict(nested_constraints.get(dimension) or {}).get(
                        "status"
                    ),
                    "values": list(
                        dict(nested_constraints.get(dimension) or {}).get(
                            "values"
                        )
                        or ()
                    ),
                }
                for dimension in ("locations", "categories", "dayparts")
            },
            "selected": selected,
        }
        return stable_digest(safe)

    @classmethod
    def _decision_signature(cls, result: Mapping[str, Any]) -> str:
        """Fingerprint routing decision and canonical constraints only.

        Ranking and safe reason-code changes remain visible through the exact
        result signatures, but they do not make a replacement model fail the
        production agreement gate when both systems make the same safe routing
        decision over the same governed constraints.
        """

        constraints = dict(result.get("constraints") or {})
        nested_constraints = dict(
            constraints.get("constraints") or {}
        )
        safe = {
            "outcome": cls._decision_outcome(result),
            "constraints": {
                dimension: cls._constraint_decision_state(
                    nested_constraints.get(dimension)
                )
                for dimension in (
                    "locations",
                    "categories",
                    "dayparts",
                )
            },
        }
        return stable_digest(safe)

    @staticmethod
    def _constraint_decision_state(value: Any) -> dict[str, Any]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        values = sorted(
            {
                normalized
                for raw in list(payload.get("values") or ())
                if (
                    normalized := normalize_taxonomy_value(raw)
                )
            }
        )
        status = normalize_taxonomy_value(payload.get("status"))

        if values:
            decision_status = "resolved"
        elif status in {
            "not_requested",
            "not_applicable",
            "not_evaluated",
        }:
            decision_status = "not_requested"
        else:
            decision_status = "unresolved"

        return {
            "status": decision_status,
            "values": values,
        }

    @classmethod
    def _decision_outcome(cls, result: Mapping[str, Any]) -> str:
        if cls._is_ready(result):
            return "ready"

        status = normalize_taxonomy_value(cls._status(result))
        reason = normalize_taxonomy_value(result.get("reason_code"))
        combined = f"{status} {reason}".strip()

        if "error" in combined or "exception" in combined:
            return "error"
        if any(
            token in combined
            for token in (
                "blocked",
                "clarification",
                "unsupported",
                "no_safe",
                "skipped",
            )
        ):
            return "blocked"
        return status or "unknown"

    @staticmethod
    def _is_ready(result: Mapping[str, Any]) -> bool:
        return str(result.get("status") or "") in {
            "retrieval_ready_for_human_review",
            "completed",
            "ready",
        }

    @classmethod
    def _safety_divergence(
        cls,
        incumbent: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        candidate_error: bool,
    ) -> bool:
        if candidate_error:
            return False
        incumbent_ready = cls._is_ready(incumbent)
        candidate_ready = cls._is_ready(candidate)
        return candidate_ready and not incumbent_ready


def evaluate_module2_shadow_readiness(
    observations: Sequence[Module2ShadowObservation],
    *,
    policy: Module2ShadowPolicy | None = None,
) -> dict[str, Any]:
    selected_policy = policy or Module2ShadowPolicy()
    values = list(observations)
    sample_count = len(values)
    agreement_rate = (
        sum(value.agreement for value in values) / sample_count
        if sample_count
        else 0.0
    )
    safety_divergence_rate = (
        sum(value.safety_divergence for value in values) / sample_count
        if sample_count
        else 1.0
    )
    candidate_error_rate = (
        sum(value.candidate_error for value in values) / sample_count
        if sample_count
        else 1.0
    )
    overheads = sorted(value.latency_overhead_ms for value in values)
    p95_overhead = _percentile(overheads, 0.95) if overheads else float("inf")
    checks = {
        "minimum_samples": sample_count >= selected_policy.minimum_samples,
        "agreement_rate": (
            agreement_rate >= selected_policy.minimum_agreement_rate
        ),
        "safety_divergence_rate": (
            safety_divergence_rate
            <= selected_policy.maximum_safety_divergence_rate
        ),
        "candidate_error_rate": (
            candidate_error_rate
            <= selected_policy.maximum_candidate_error_rate
        ),
        "p95_latency_overhead": (
            p95_overhead
            <= selected_policy.maximum_p95_latency_overhead_ms
        ),
    }
    ready = all(checks.values())
    report = {
        "contract_version": "module2-shadow-readiness-report-v2",
        "status": "shadow_release_ready" if ready else "shadow_release_blocked",
        "shadow_release_ready": ready,
        "sample_count": sample_count,
        "agreement_rate": round(agreement_rate, 12),
        "safety_divergence_rate": round(safety_divergence_rate, 12),
        "candidate_error_rate": round(candidate_error_rate, 12),
        "p95_latency_overhead_ms": (
            round(p95_overhead, 6) if values else None
        ),
        "checks": checks,
        "policy": {
            **selected_policy.__dict__,
            "policy_fingerprint": selected_policy.fingerprint,
        },
        "raw_query_stored": False,
        "raw_identifiers_stored": False,
        "candidate_output_routed_to_user": False,
        "production_routing_enabled": False,
        "automatic_proposal_creation_enabled": False,
        "activation_or_export_performed": False,
    }
    report["report_fingerprint"] = stable_digest(report)
    return report


def validate_module2_shadow_readiness_report(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    import json

    safe = json.loads(json.dumps(dict(payload), allow_nan=False))
    declared = str(safe.pop("report_fingerprint", "")).strip().lower()
    if declared != stable_digest(safe):
        raise ValueError("Module 2 shadow report fingerprint mismatch.")
    for flag in (
        "raw_query_stored",
        "raw_identifiers_stored",
        "candidate_output_routed_to_user",
        "production_routing_enabled",
        "automatic_proposal_creation_enabled",
        "activation_or_export_performed",
    ):
        if safe.get(flag) is not False:
            raise ValueError(f"Shadow report must keep {flag} disabled.")
    safe["report_fingerprint"] = declared
    return safe


class ProductionModule2ReleaseGateService:
    """Produce a recommendation only; never mutate routing or index state."""

    def evaluate(
        self,
        *,
        certification_report: Mapping[str, Any],
        index_record: Mapping[str, Any],
        shadow_report: Mapping[str, Any],
        operator_approval_recorded: bool = False,
    ) -> dict[str, Any]:
        checks = {
            "production_certification": bool(
                certification_report.get("production_certification_ready")
            ),
            "active_index_candidate": index_record.get("status") == "shadow",
            "shadow_release": bool(shadow_report.get("shadow_release_ready")),
            "operator_approval": bool(operator_approval_recorded),
        }
        recommendation_ready = all(checks.values())
        report = {
            "contract_version": "module2-release-gate-report-v1",
            "status": (
                "eligible_for_explicit_operator_activation"
                if recommendation_ready
                else "release_blocked"
            ),
            "release_recommendation_ready": recommendation_ready,
            "checks": checks,
            "certification_report_fingerprint": certification_report.get(
                "report_fingerprint"
            ),
            "index_manifest_fingerprint": index_record.get(
                "manifest_fingerprint"
            ),
            "shadow_report_fingerprint": shadow_report.get(
                "report_fingerprint"
            ),
            "model_registration_performed": False,
            "index_activation_performed": False,
            "production_routing_enabled": False,
            "automatic_proposal_creation_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }
        report["report_fingerprint"] = stable_digest(report)
        return report


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("inf")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * fraction)
