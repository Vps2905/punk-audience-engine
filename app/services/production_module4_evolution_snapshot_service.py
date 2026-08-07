from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    finite_unit_interval,
    stable_fingerprint,
)
from app.models.production_module4_evolution_contracts import (
    MODULE4_EVOLUTION_POLICY_VERSION,
    GovernedEvolutionCohortSnapshot,
    GovernedEvolutionSnapshotReport,
    Module4EvolutionSnapshotRequest,
)


class ProductionModule4EvolutionSnapshotService:
    """Create immutable aggregate-only cohort monitoring snapshots."""

    def __init__(self) -> None:
        self._minimum_quality = 0.70

    def build(
        self,
        *,
        request: Module4EvolutionSnapshotRequest,
        cohort_rows: Sequence[Mapping[str, Any]],
    ) -> GovernedEvolutionSnapshotReport:
        rows = [dict(value) for value in cohort_rows]
        self._assert_safe_input(rows)
        assert_no_raw_identifier_fields(rows)
        if not rows:
            raise ValueError("Module 4.1 requires at least one aggregate cohort.")
        if len(rows) > 5000:
            raise ValueError("Module 4.1 cohort input exceeds the safety limit.")
        ids = [str(value.get("export_cohort_id") or "").strip() for value in rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Module 4.1 cohort identities must be unique.")

        canonical_rows = [self._canonical_row(value) for value in rows]
        canonical_rows.sort(key=lambda value: value["export_cohort_id"])
        snapshot_fingerprint = stable_fingerprint(
            {
                "policy_version": MODULE4_EVOLUTION_POLICY_VERSION,
                "minimum_monitoring_quality": self._minimum_quality,
                "request": request.to_record(),
                "cohorts": canonical_rows,
            }
        )
        snapshots = [
            self._snapshot(
                request=request,
                snapshot_fingerprint=snapshot_fingerprint,
                row=value,
            )
            for value in canonical_rows
        ]
        statuses = [value.monitoring_status for value in snapshots]
        safety = {
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "raw_identifiers_returned": False,
            "audience_membership_read": False,
            "individual_behavior_inferred": False,
            "cohort_lifecycle_mutated": False,
            "automatic_evolution_performed": False,
            "automatic_approval_performed": False,
            "manual_approval_required": True,
            "routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }
        return GovernedEvolutionSnapshotReport(
            status="engineering_preview_ready",
            policy_version=MODULE4_EVOLUTION_POLICY_VERSION,
            request=request,
            snapshot_fingerprint=snapshot_fingerprint,
            source_cohort_count=len(snapshots),
            monitoring_count=statuses.count("monitoring_baseline"),
            review_required_count=statuses.count("review_required_quality"),
            paused_count=statuses.count("paused_stale_source"),
            blocked_count=sum(value.startswith("blocked_") for value in statuses),
            historical_count=statuses.count("historical_baseline"),
            cohort_snapshots=snapshots,
            safety=safety,
        )

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        request_value = payload.get("request")
        values = payload.get("cohort_snapshots")
        safety = payload.get("safety")
        if not isinstance(request_value, Mapping) or not isinstance(values, list):
            raise ValueError("Module 4.1 request and snapshots are required.")
        if not isinstance(safety, Mapping):
            raise ValueError("Module 4.1 safety evidence is required.")
        request = Module4EvolutionSnapshotRequest(**dict(request_value))
        snapshots = [
            GovernedEvolutionCohortSnapshot(**dict(value))
            for value in values
            if isinstance(value, Mapping)
        ]
        if len(snapshots) != len(values):
            raise ValueError("Module 4.1 snapshots contain invalid values.")
        canonical_rows = [
            {
                "export_cohort_id": value.export_cohort_id,
                "quality_score": value.quality_score,
                "freshness_status": value.freshness_status,
                "approval_status": value.approval_status,
                "data_safety_status": value.data_safety_status,
                "risk_decision": value.risk_decision,
            }
            for value in snapshots
        ]
        canonical_rows.sort(key=lambda value: value["export_cohort_id"])
        expected_snapshot = stable_fingerprint({
            "policy_version": MODULE4_EVOLUTION_POLICY_VERSION,
            "minimum_monitoring_quality": self._minimum_quality,
            "request": request.to_record(),
            "cohorts": canonical_rows,
        })
        if payload.get("snapshot_fingerprint") != expected_snapshot:
            raise ValueError("Module 4.1 snapshot fingerprint mismatch.")
        for value in snapshots:
            identity = value.to_record()
            fingerprint = identity.pop("cohort_snapshot_fingerprint")
            if stable_fingerprint(identity) != fingerprint:
                raise ValueError("Module 4.1 cohort fingerprint mismatch.")
        for field in (
            "raw_identifiers_read", "raw_identifiers_stored",
            "raw_identifiers_returned", "audience_membership_read",
            "individual_behavior_inferred", "cohort_lifecycle_mutated",
            "automatic_evolution_performed", "automatic_approval_performed",
            "routing_enabled", "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe Module 4.1 safety field: {field}.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Module 4.1 manual approval must remain required.")
        statuses = [value.monitoring_status for value in snapshots]
        expected_counts = {
            "source_cohort_count": len(statuses),
            "monitoring_count": statuses.count("monitoring_baseline"),
            "review_required_count": statuses.count("review_required_quality"),
            "paused_count": statuses.count("paused_stale_source"),
            "blocked_count": sum(value.startswith("blocked_") for value in statuses),
            "historical_count": statuses.count("historical_baseline"),
        }
        for field, expected in expected_counts.items():
            if int(payload.get(field) or 0) != expected:
                raise ValueError(f"Module 4.1 {field} is inconsistent.")
        return payload

    def _canonical_row(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "export_cohort_id": str(value.get("export_cohort_id") or "").strip(),
            "quality_score": finite_unit_interval(
                value.get("management_quality_score")
                if value.get("management_quality_score") is not None
                else value.get("quality_score", 0.0),
                label="cohort quality_score",
            ),
            "freshness_status": normalize_taxonomy_value(
                value.get("freshness_status") or "unknown"
            ),
            "approval_status": normalize_taxonomy_value(
                value.get("approval_status") or "unknown"
            ),
            "data_safety_status": normalize_taxonomy_value(
                value.get("data_safety_status") or "unknown"
            ),
            "risk_decision": normalize_taxonomy_value(
                value.get("risk_decision") or "unknown"
            ),
        }

    def _snapshot(
        self,
        *,
        request: Module4EvolutionSnapshotRequest,
        snapshot_fingerprint: str,
        row: Mapping[str, Any],
    ) -> GovernedEvolutionCohortSnapshot:
        freshness = str(row["freshness_status"])
        approval = str(row["approval_status"])
        risk = str(row["risk_decision"])
        quality = float(row["quality_score"])
        if risk.startswith("blocked") or risk in {"block", "deny", "rejected"}:
            status, reasons = "blocked_policy", ["aggregate_policy_risk_block"]
        elif approval.startswith("blocked") or approval in {"rejected", "denied"}:
            status, reasons = "blocked_approval", ["approval_state_blocks_evolution"]
        elif freshness != "fresh":
            status, reasons = "paused_stale_source", ["source_freshness_not_fresh"]
        elif request.execution_mode != "production":
            status, reasons = "historical_baseline", ["non_production_monitoring_only"]
        elif quality < self._minimum_quality:
            status, reasons = "review_required_quality", ["quality_below_monitoring_floor"]
        else:
            status, reasons = "monitoring_baseline", ["aggregate_baseline_recorded"]
        identity = {
            "tenant_id": request.tenant_id,
            "snapshot_fingerprint": snapshot_fingerprint,
            "source_run_id": request.source_run_id,
            **dict(row),
            "monitoring_status": status,
            "reason_codes": reasons,
            "lifecycle_mutated": False,
            "automatic_approval_performed": False,
            "routing_enabled": False,
            "eligible_for_activation": False,
            "eligible_for_export": False,
        }
        return GovernedEvolutionCohortSnapshot(
            cohort_snapshot_fingerprint=stable_fingerprint(identity),
            **identity,
        )

    def _assert_safe_input(self, value: Any) -> None:
        blocked = (
            "raw_identifier", "individual_id", "device_id", "maid",
            "email", "phone", "latitude", "longitude", "hashed_identifier",
        )
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = normalize_taxonomy_value(key)
                if any(token in normalized for token in blocked):
                    raise ValueError(
                        "Module 4.1 input contains a prohibited identifier field."
                    )
                self._assert_safe_input(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                self._assert_safe_input(nested)
