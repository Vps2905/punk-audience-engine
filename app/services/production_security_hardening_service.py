from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    required_slug,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    required_sha256_digest,
)
from app.models.production_security_contracts import (
    SECURITY_POSTURE_POLICY_VERSION,
    SECURITY_REVIEW_POLICY_VERSION,
    ProductionSecurityAssessmentRequest,
)


_RELEASE_AFFECTING_FLAGS = (
    "MODULE2_MODEL_REGISTRATION_ENABLED",
    "MODULE2_PRODUCTION_ROUTING_ENABLED",
    "MODULE3_COHORT_GENERATION_ENABLED",
    "MODULE3_COHORT_PERSISTENCE_ENABLED",
    "MODULE3_OVERLAP_DEDUPLICATION_ENABLED",
    "MODULE3_LOOKALIKE_GENERATION_ENABLED",
    "MODULE3_PRODUCTION_ROUTING_ENABLED",
    "MODULE4_AUTOMATIC_EVOLUTION_ENABLED",
    "MODULE4_PRODUCTION_ROUTING_ENABLED",
    "MODULE5_AUTONOMOUS_MUTATION_ENABLED",
    "MODULE5_PRODUCTION_ROUTING_ENABLED",
    "QUALITY_AUTOMATIC_REMEDIATION_ENABLED",
    "QUALITY_PRODUCTION_ROUTING_ENABLED",
    "OBSERVABILITY_AUTOMATIC_REMEDIATION_ENABLED",
    "OBSERVABILITY_PRODUCTION_ROUTING_ENABLED",
    "SECURITY_PRODUCTION_RELEASE_ENABLED",
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safety() -> dict[str, Any]:
    return {
        "secret_values_stored": False,
        "credentials_returned": False,
        "database_urls_returned": False,
        "environment_values_returned": False,
        "vulnerability_exploit_attempted": False,
        "security_configuration_mutated": False,
        "automatic_remediation_performed": False,
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
        "database_urls_returned",
        "environment_values_returned",
        "vulnerability_exploit_attempted",
        "security_configuration_mutated",
        "automatic_remediation_performed",
        "production_release_authorized",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"Unsafe {label} safety field: {field}.")
    if safety.get("manual_approval_required") is not True:
        raise ValueError(f"{label} manual approval must remain required.")


class ProductionSecurityPostureService:
    """Assess non-secret deployment posture without returning environment values."""

    def assess(
        self,
        *,
        request: ProductionSecurityAssessmentRequest,
        environment: Mapping[str, str],
    ) -> dict[str, Any]:
        controls = self._controls(environment)
        passed_count = sum(value["passed"] for value in controls)
        failed = [value["control"] for value in controls if not value["passed"]]
        posture = "pass" if not failed else "fail_closed"
        fingerprint = stable_fingerprint({
            "policy_version": SECURITY_POSTURE_POLICY_VERSION,
            "request": request.to_record(),
            "controls": controls,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": SECURITY_POSTURE_POLICY_VERSION,
            "request": request.to_record(),
            "security_posture_fingerprint": fingerprint,
            "control_count": len(controls),
            "passed_control_count": passed_count,
            "failed_control_count": len(failed),
            "posture_status": posture,
            "failed_control_codes": failed,
            "controls": controls,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported security posture status.")
        if payload.get("policy_version") != SECURITY_POSTURE_POLICY_VERSION:
            raise ValueError("Unsupported security posture policy version.")
        request = ProductionSecurityAssessmentRequest(
            **dict(payload.get("request") or {})
        )
        controls = payload.get("controls")
        if not isinstance(controls, list) or not controls:
            raise ValueError("Security posture controls are required.")
        if controls != sorted(controls, key=lambda value: value.get("control", "")):
            raise ValueError("Security posture controls must be ordered.")
        if len({value.get("control") for value in controls}) != len(controls):
            raise ValueError("Security posture control names must be unique.")
        for value in controls:
            required_slug(value.get("control"), label="security_control")
            if value.get("passed") not in {True, False}:
                raise ValueError("Security posture result must be boolean.")
            required_slug(value.get("evidence_code"), label="evidence_code")
        passed = sum(value["passed"] for value in controls)
        failed = [value["control"] for value in controls if not value["passed"]]
        expected = {
            "control_count": len(controls),
            "passed_control_count": passed,
            "failed_control_count": len(failed),
        }
        for field, count in expected.items():
            if int(payload.get(field) or 0) != count:
                raise ValueError(f"Security posture {field} is inconsistent.")
        if payload.get("failed_control_codes") != failed:
            raise ValueError("Security posture failed controls are inconsistent.")
        expected_status = "pass" if not failed else "fail_closed"
        if payload.get("posture_status") != expected_status:
            raise ValueError("Security posture result is inconsistent.")
        expected_fingerprint = stable_fingerprint({
            "policy_version": SECURITY_POSTURE_POLICY_VERSION,
            "request": request.to_record(),
            "controls": controls,
        })
        if payload.get("security_posture_fingerprint") != expected_fingerprint:
            raise ValueError("Security posture fingerprint mismatch.")
        _validate_safety(payload, "security posture")
        return payload

    def _controls(self, environment: Mapping[str, str]):
        production = _truthy(environment.get("PRODUCTION_MODE")) or str(
            environment.get("APP_ENV") or ""
        ).strip().lower() == "production"
        api_key = str(environment.get("AUDIENCE_API_KEY") or "")
        tenant_secret = str(environment.get("PUNK_AI_TENANT_AUTH_SECRET") or "")
        allowed_hosts = [
            value.strip().lower()
            for value in str(environment.get("PRODUCTION_ALLOWED_HOSTS") or "").split(",")
            if value.strip()
        ]
        cors_origins = [
            value.strip().lower()
            for value in str(environment.get("PRODUCTION_CORS_ORIGINS") or "").split(",")
            if value.strip()
        ]
        max_body = self._safe_int(environment.get("MAX_REQUEST_BODY_BYTES"))
        release_flags_disabled = not any(
            _truthy(environment.get(key)) for key in _RELEASE_AFFECTING_FLAGS
        )
        results = {
            "api_docs_disabled": production
            and not _truthy(environment.get("EXPOSE_API_DOCS")),
            "api_key_auth_required": _truthy(
                environment.get("REQUIRE_AUDIENCE_API_KEY")
            ),
            "api_key_strength_attested": self._strong_secret(api_key),
            "container_non_root": _truthy(
                environment.get("CONTAINER_RUNTIME_NON_ROOT")
            ),
            "cors_origins_restricted": "*" not in cors_origins
            and all(value.startswith("https://") for value in cors_origins),
            "database_tls_required": _truthy(
                environment.get("REQUIRE_DATABASE_TLS")
            ),
            "debug_disabled": not _truthy(environment.get("DEBUG")),
            "demo_routes_disabled": not _truthy(
                environment.get("ALLOW_DEMO_ROUTES")
            ),
            "local_file_storage_disabled": not _truthy(
                environment.get("ALLOW_LOCAL_FILE_STORAGE")
            ),
            "no_new_privileges": _truthy(
                environment.get("CONTAINER_NO_NEW_PRIVILEGES")
            ),
            "outbound_tls_required": _truthy(
                environment.get("REQUIRE_OUTBOUND_TLS")
            ),
            "production_mode_enabled": production,
            "production_release_flags_disabled": release_flags_disabled,
            "request_body_limit_bounded": max_body is not None
            and 1024 <= max_body <= 20_971_520,
            "root_filesystem_read_only": _truthy(
                environment.get("CONTAINER_ROOT_FILESYSTEM_READ_ONLY")
            ),
            "secret_manager_injection_attested": _truthy(
                environment.get("SECRETS_INJECTED_BY_SECRET_MANAGER")
            ),
            "security_headers_enabled": _truthy(
                environment.get("SECURITY_HEADERS_ENABLED")
            ),
            "tenant_signature_required": production
            and _truthy(environment.get("REQUIRE_PUNK_AI_TENANT_SIGNATURE")),
            "tenant_signing_secret_strength_attested": self._strong_secret(
                tenant_secret
            ),
            "trusted_hosts_restricted": bool(allowed_hosts)
            and "*" not in allowed_hosts,
        }
        return [
            {
                "control": name,
                "passed": bool(passed),
                "evidence_code": "control_satisfied" if passed else "control_missing",
            }
            for name, passed in sorted(results.items())
        ]

    def _strong_secret(self, value: str) -> bool:
        normalized = value.strip().lower()
        if len(value) < 32 or len(set(value)) < 8:
            return False
        return not any(
            marker in normalized
            for marker in ("change_me", "changeme", "example", "password", "test-key")
        )

    def _safe_int(self, value: Any) -> int | None:
        try:
            return int(str(value or "").strip())
        except ValueError:
            return None


class ProductionSecurityManualReviewService:
    """Record a manual security review without authorizing production release."""

    def review(
        self,
        *,
        posture_report: Mapping[str, Any],
        manual_review: Mapping[str, Any],
    ) -> dict[str, Any]:
        posture = ProductionSecurityPostureService().validate_report(posture_report)
        decision = str(manual_review.get("decision") or "").strip()
        reference = required_slug(
            manual_review.get("review_reference"),
            label="review_reference",
        )
        if decision not in {
            "approved_for_preproduction_review",
            "needs_remediation",
            "rejected",
        }:
            raise ValueError("Unsupported security manual review decision.")
        if (
            decision == "approved_for_preproduction_review"
            and posture["posture_status"] != "pass"
        ):
            raise ValueError("Security posture failures cannot be approved.")
        review = {
            "decision": decision,
            "review_reference": reference,
            "security_controls_passed": posture["posture_status"] == "pass",
            "production_release_authorized": False,
            "automatic_remediation_performed": False,
            "manual_approval_required": True,
        }
        tenant_id = posture["request"]["tenant_id"]
        source_fingerprint = posture["security_posture_fingerprint"]
        fingerprint = stable_fingerprint({
            "policy_version": SECURITY_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_security_posture_fingerprint": source_fingerprint,
            "manual_review": review,
        })
        report = {
            "status": "engineering_preview_ready",
            "policy_version": SECURITY_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_security_posture_fingerprint": source_fingerprint,
            "security_review_fingerprint": fingerprint,
            "manual_review": review,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported security review status.")
        if payload.get("policy_version") != SECURITY_REVIEW_POLICY_VERSION:
            raise ValueError("Unsupported security review policy version.")
        tenant_id = required_slug(payload.get("tenant_id"), label="tenant_id")
        source = required_sha256_digest(
            payload.get("source_security_posture_fingerprint"),
            label="source_security_posture_fingerprint",
        )
        review = payload.get("manual_review")
        if not isinstance(review, Mapping):
            raise ValueError("Security manual review is required.")
        if review.get("decision") not in {
            "approved_for_preproduction_review",
            "needs_remediation",
            "rejected",
        }:
            raise ValueError("Security manual review decision is invalid.")
        required_slug(review.get("review_reference"), label="review_reference")
        if review.get("security_controls_passed") not in {True, False}:
            raise ValueError("Security review control result must be boolean.")
        if (
            review.get("decision") == "approved_for_preproduction_review"
            and review.get("security_controls_passed") is not True
        ):
            raise ValueError("Failed security controls cannot be approved.")
        for field in (
            "production_release_authorized",
            "automatic_remediation_performed",
        ):
            if review.get(field) is not False:
                raise ValueError(f"Security review cannot set {field}.")
        if review.get("manual_approval_required") is not True:
            raise ValueError("Security review manual approval is required.")
        expected = stable_fingerprint({
            "policy_version": SECURITY_REVIEW_POLICY_VERSION,
            "tenant_id": tenant_id,
            "source_security_posture_fingerprint": source,
            "manual_review": dict(review),
        })
        if payload.get("security_review_fingerprint") != expected:
            raise ValueError("Security review fingerprint mismatch.")
        _validate_safety(payload, "security review")
        return payload
