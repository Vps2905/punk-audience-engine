from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models.production_infrastructure_contracts import (
    INFRASTRUCTURE_ASSESSMENT_POLICY_VERSION,
    INFRASTRUCTURE_REVIEW_POLICY_VERSION,
    ProductionInfrastructureAssessmentRequest,
)
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)


REQUIRED_INFRASTRUCTURE_CONTROLS = (
    "alarm_routes_verified",
    "autoscaling_limits_verified",
    "change_set_reviewed",
    "cloudformation_template_validated",
    "cost_budget_reviewed",
    "data_plane_dependencies_verified",
    "database_backup_policy_verified",
    "database_restore_plan_verified",
    "immutable_image_digest_verified",
    "least_privilege_iam_reviewed",
    "multi_az_capacity_verified",
    "private_network_boundaries_verified",
    "rollback_plan_verified",
    "secret_manager_injection_verified",
    "security_group_paths_verified",
    "service_quotas_verified",
    "tls_listener_verified",
)


def _safety() -> dict[str, Any]:
    return {
        "secret_values_stored": False,
        "credentials_returned": False,
        "resource_identifiers_returned": False,
        "environment_values_returned": False,
        "cloud_resources_mutated": False,
        "change_set_executed": False,
        "automatic_deployment_performed": False,
        "production_traffic_enabled": False,
        "production_release_authorized": False,
        "manual_approval_required": True,
    }


def _validate_safety(report: Mapping[str, Any], label: str) -> None:
    safety = report.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError(f"{label} safety evidence is required.")
    for field in (
        "secret_values_stored",
        "credentials_returned",
        "resource_identifiers_returned",
        "environment_values_returned",
        "cloud_resources_mutated",
        "change_set_executed",
        "automatic_deployment_performed",
        "production_traffic_enabled",
        "production_release_authorized",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"Unsafe {label} safety field: {field}.")
    if safety.get("manual_approval_required") is not True:
        raise ValueError(f"{label} manual approval must remain required.")


class ProductionInfrastructureAssessmentService:
    """Create non-secret IaC readiness evidence without changing AWS state."""

    def assess(
        self,
        *,
        request: ProductionInfrastructureAssessmentRequest,
        attestations: Mapping[str, Any],
    ) -> dict[str, Any]:
        unknown = sorted(set(attestations) - set(REQUIRED_INFRASTRUCTURE_CONTROLS))
        if unknown:
            raise ValueError("Unknown infrastructure control attestations.")
        controls = [
            {
                "control": name,
                "passed": attestations.get(name) is True,
                "evidence_code": (
                    "operator_attested" if attestations.get(name) is True
                    else "attestation_missing"
                ),
            }
            for name in REQUIRED_INFRASTRUCTURE_CONTROLS
        ]
        failed = [value["control"] for value in controls if not value["passed"]]
        fingerprint = stable_fingerprint({
            "policy_version": INFRASTRUCTURE_ASSESSMENT_POLICY_VERSION,
            "request": request.to_record(),
            "controls": controls,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": INFRASTRUCTURE_ASSESSMENT_POLICY_VERSION,
            "request": request.to_record(),
            "infrastructure_assessment_fingerprint": fingerprint,
            "control_count": len(controls),
            "passed_control_count": len(controls) - len(failed),
            "failed_control_count": len(failed),
            "assessment_status": "pass" if not failed else "fail_closed",
            "failed_control_codes": failed,
            "controls": controls,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported infrastructure assessment status.")
        if payload.get("policy_version") != INFRASTRUCTURE_ASSESSMENT_POLICY_VERSION:
            raise ValueError("Unsupported infrastructure assessment policy.")
        request = ProductionInfrastructureAssessmentRequest(
            **dict(payload.get("request") or {})
        )
        controls = payload.get("controls")
        if not isinstance(controls, list):
            raise ValueError("Infrastructure controls are required.")
        if [value.get("control") for value in controls] != list(
            REQUIRED_INFRASTRUCTURE_CONTROLS
        ):
            raise ValueError("Infrastructure control set is inconsistent.")
        for value in controls:
            if value.get("passed") not in {True, False}:
                raise ValueError("Infrastructure control result must be boolean.")
            expected_code = (
                "operator_attested" if value["passed"] else "attestation_missing"
            )
            if value.get("evidence_code") != expected_code:
                raise ValueError("Infrastructure evidence code is inconsistent.")
        failed = [value["control"] for value in controls if not value["passed"]]
        expected_counts = {
            "control_count": len(controls),
            "passed_control_count": len(controls) - len(failed),
            "failed_control_count": len(failed),
        }
        for field, expected in expected_counts.items():
            if int(payload.get(field) or 0) != expected:
                raise ValueError(f"Infrastructure {field} is inconsistent.")
        if payload.get("failed_control_codes") != failed:
            raise ValueError("Infrastructure failed controls are inconsistent.")
        expected_status = "pass" if not failed else "fail_closed"
        if payload.get("assessment_status") != expected_status:
            raise ValueError("Infrastructure assessment result is inconsistent.")
        expected_fingerprint = stable_fingerprint({
            "policy_version": INFRASTRUCTURE_ASSESSMENT_POLICY_VERSION,
            "request": request.to_record(),
            "controls": controls,
        })
        if payload.get("infrastructure_assessment_fingerprint") != (
            expected_fingerprint
        ):
            raise ValueError("Infrastructure assessment fingerprint mismatch.")
        _validate_safety(payload, "infrastructure assessment")
        return payload


class ProductionInfrastructureChangeSetReviewService:
    """Record review approval without creating or executing a change set."""

    def review(
        self,
        *,
        assessment_report: Mapping[str, Any],
        manual_review: Mapping[str, Any],
    ) -> dict[str, Any]:
        assessment = ProductionInfrastructureAssessmentService().validate_report(
            assessment_report
        )
        decision = str(manual_review.get("decision") or "").strip()
        review_reference = required_slug(
            manual_review.get("review_reference"),
            label="review_reference",
        )
        if decision not in {
            "approved_for_preproduction_change_set",
            "requires_changes",
            "rejected",
        }:
            raise ValueError("Unsupported infrastructure review decision.")
        passed = assessment["assessment_status"] == "pass"
        if decision == "approved_for_preproduction_change_set" and not passed:
            raise ValueError("Failed infrastructure controls cannot be approved.")
        review = {
            "decision": decision,
            "review_reference": review_reference,
            "infrastructure_controls_passed": passed,
            "change_set_execution_authorized": False,
            "production_traffic_authorized": False,
            "production_release_authorized": False,
            "manual_approval_required": True,
        }
        tenant_id = assessment["request"]["tenant_id"]
        source = assessment["infrastructure_assessment_fingerprint"]
        fingerprint = stable_fingerprint({
            "policy_version": INFRASTRUCTURE_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_infrastructure_assessment_fingerprint": source,
            "manual_review": review,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": INFRASTRUCTURE_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_infrastructure_assessment_fingerprint": source,
            "infrastructure_review_fingerprint": fingerprint,
            "manual_review": review,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported infrastructure review status.")
        if payload.get("policy_version") != INFRASTRUCTURE_REVIEW_POLICY_VERSION:
            raise ValueError("Unsupported infrastructure review policy.")
        tenant_id = required_slug(payload.get("tenant_id"), label="tenant_id")
        source = required_sha256_digest(
            payload.get("source_infrastructure_assessment_fingerprint"),
            label="source_infrastructure_assessment_fingerprint",
        )
        review = payload.get("manual_review")
        if not isinstance(review, Mapping):
            raise ValueError("Infrastructure manual review is required.")
        if review.get("decision") not in {
            "approved_for_preproduction_change_set",
            "requires_changes",
            "rejected",
        }:
            raise ValueError("Infrastructure review decision is invalid.")
        required_slug(review.get("review_reference"), label="review_reference")
        if review.get("infrastructure_controls_passed") not in {True, False}:
            raise ValueError("Infrastructure review result must be boolean.")
        if (
            review.get("decision") == "approved_for_preproduction_change_set"
            and review.get("infrastructure_controls_passed") is not True
        ):
            raise ValueError("Failed infrastructure controls cannot be approved.")
        for field in (
            "change_set_execution_authorized",
            "production_traffic_authorized",
            "production_release_authorized",
        ):
            if review.get(field) is not False:
                raise ValueError(f"Infrastructure review cannot set {field}.")
        if review.get("manual_approval_required") is not True:
            raise ValueError("Infrastructure manual approval remains required.")
        expected = stable_fingerprint({
            "policy_version": INFRASTRUCTURE_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_infrastructure_assessment_fingerprint": source,
            "manual_review": dict(review),
        })
        if payload.get("infrastructure_review_fingerprint") != expected:
            raise ValueError("Infrastructure review fingerprint mismatch.")
        _validate_safety(payload, "infrastructure review")
        return payload
