from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.models.production_preproduction_deployment_contracts import (
    PREPRODUCTION_DEPLOYMENT_CERTIFICATION_POLICY_VERSION,
    PreproductionDeploymentCertificationRequest,
)


ALLOWED_STACK_STATUSES = {"CREATE_COMPLETE", "UPDATE_COMPLETE"}

OBSERVATION_FIELDS = (
    "stack_status",
    "review_fingerprint",
    "running_image_digest",
    "api_desired_count",
    "api_running_count",
    "api_healthy_target_count",
    "api_availability_zone_count",
    "worker_desired_count",
    "worker_running_count",
    "worker_availability_zone_count",
    "database_storage_encrypted",
    "database_multi_az",
    "database_publicly_accessible",
    "database_deletion_protection",
    "database_tls_required",
    "database_backup_retention_days",
    "applied_migration_head",
    "pending_migration_count",
    "migration_checksum_mismatch_count",
    "runtime_database_roles_separated",
    "database_admin_credentials_in_runtime",
    "secret_injection_verified",
    "plaintext_secret_count",
    "alarm_route_verified",
    "alarm_count",
    "alarms_in_alarm_state_count",
    "deployment_rollback_enabled",
    "production_traffic_enabled",
    "downstream_export_enabled",
    "live_provider_data_used",
)

CONTROL_NAMES = (
    "isolated_preproduction_stack_complete",
    "reviewed_change_set_binding_verified",
    "immutable_candidate_image_running",
    "api_service_healthy_multi_az",
    "provider_worker_healthy_multi_az",
    "database_private_encrypted_multi_az",
    "database_tls_backup_and_deletion_protection",
    "database_migrations_complete_and_immutable",
    "runtime_database_roles_separated",
    "secret_injection_without_plaintext",
    "alarm_routes_and_runtime_alarms_verified",
    "deployment_rollback_enabled",
    "production_traffic_disabled",
    "downstream_export_disabled",
    "no_live_provider_data_required",
)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return parsed


def _boolean(value: Any, *, label: str) -> bool:
    if value not in {True, False}:
        raise ValueError(f"{label} must be boolean.")
    return bool(value)


def _safety() -> dict[str, Any]:
    return {
        "inspection_read_only": True,
        "secret_values_read": False,
        "secret_values_returned": False,
        "resource_identifiers_returned": False,
        "data_rows_read": False,
        "live_provider_data_required": False,
        "production_resources_mutated": False,
        "production_traffic_enabled": False,
        "audience_activation_performed": False,
        "downstream_export_performed": False,
        "live_production_certified": False,
        "manual_production_approval_required": True,
    }


def _validate_safety(report: Mapping[str, Any]) -> None:
    safety = report.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("Deployment certification safety evidence is required.")
    if safety.get("inspection_read_only") is not True:
        raise ValueError("Deployment certification inspection must be read-only.")
    for field in (
        "secret_values_read",
        "secret_values_returned",
        "resource_identifiers_returned",
        "data_rows_read",
        "live_provider_data_required",
        "production_resources_mutated",
        "production_traffic_enabled",
        "audience_activation_performed",
        "downstream_export_performed",
        "live_production_certified",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"Unsafe deployment certification field: {field}.")
    if safety.get("manual_production_approval_required") is not True:
        raise ValueError("Manual production approval must remain required.")


class ProductionPreproductionDeploymentCertificationService:
    """Certify a measured staging deployment without reading audience data."""

    def certify(
        self,
        *,
        request: PreproductionDeploymentCertificationRequest,
        observations: Mapping[str, Any],
    ) -> dict[str, Any]:
        unknown = sorted(set(observations) - set(OBSERVATION_FIELDS))
        missing = sorted(set(OBSERVATION_FIELDS) - set(observations))
        if unknown:
            raise ValueError("Unknown preproduction deployment observations.")
        if missing:
            raise ValueError("Missing preproduction deployment observations.")

        normalized = self._normalize_observations(observations)
        controls = self._controls(request=request, observations=normalized)
        failed = [control["control"] for control in controls if not control["passed"]]
        request_record = request.to_record()
        fingerprint = stable_fingerprint(
            {
                "policy_version": (
                    PREPRODUCTION_DEPLOYMENT_CERTIFICATION_POLICY_VERSION
                ),
                "request": request_record,
                "controls": controls,
            }
        )
        report = {
            "status": (
                "preproduction_deployment_certified"
                if not failed
                else "preproduction_deployment_certification_failed_closed"
            ),
            "policy_version": (
                PREPRODUCTION_DEPLOYMENT_CERTIFICATION_POLICY_VERSION
            ),
            "request": request_record,
            "deployment_certification_fingerprint": fingerprint,
            "control_count": len(controls),
            "passed_control_count": len(controls) - len(failed),
            "failed_control_count": len(failed),
            "failed_control_codes": failed,
            "controls": controls,
            "preproduction_deployment_certified": not failed,
            "fresh_provider_data_required": False,
            "live_production_certified": False,
            "safety": _safety(),
        }
        return self.validate_report(report)

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("policy_version") != (
            PREPRODUCTION_DEPLOYMENT_CERTIFICATION_POLICY_VERSION
        ):
            raise ValueError("Unsupported deployment certification policy.")
        request = PreproductionDeploymentCertificationRequest(
            **dict(payload.get("request") or {})
        )
        controls = payload.get("controls")
        if not isinstance(controls, list):
            raise ValueError("Deployment certification controls are required.")
        if [control.get("control") for control in controls] != list(CONTROL_NAMES):
            raise ValueError("Deployment certification control set is inconsistent.")
        for control in controls:
            if control.get("passed") not in {True, False}:
                raise ValueError("Deployment certification result must be boolean.")
            expected = "measured_pass" if control["passed"] else "measured_fail"
            if control.get("evidence_code") != expected:
                raise ValueError("Deployment evidence code is inconsistent.")
        failed = [control["control"] for control in controls if not control["passed"]]
        expected_counts = {
            "control_count": len(controls),
            "passed_control_count": len(controls) - len(failed),
            "failed_control_count": len(failed),
        }
        for field, expected in expected_counts.items():
            if payload.get(field) != expected:
                raise ValueError(f"Deployment {field} is inconsistent.")
        if payload.get("failed_control_codes") != failed:
            raise ValueError("Deployment failed controls are inconsistent.")
        certified = not failed
        expected_status = (
            "preproduction_deployment_certified"
            if certified
            else "preproduction_deployment_certification_failed_closed"
        )
        if payload.get("status") != expected_status:
            raise ValueError("Deployment certification status is inconsistent.")
        if payload.get("preproduction_deployment_certified") is not certified:
            raise ValueError("Deployment certification outcome is inconsistent.")
        if payload.get("fresh_provider_data_required") is not False:
            raise ValueError("Fresh provider data is not required for this gate.")
        if payload.get("live_production_certified") is not False:
            raise ValueError("This gate cannot certify live production.")
        expected_fingerprint = stable_fingerprint(
            {
                "policy_version": (
                    PREPRODUCTION_DEPLOYMENT_CERTIFICATION_POLICY_VERSION
                ),
                "request": request.to_record(),
                "controls": controls,
            }
        )
        if payload.get("deployment_certification_fingerprint") != (
            expected_fingerprint
        ):
            raise ValueError("Deployment certification fingerprint mismatch.")
        _validate_safety(payload)
        return payload

    def _normalize_observations(
        self, observations: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = {
            "stack_status": str(observations["stack_status"] or "").strip(),
            "review_fingerprint": str(
                observations["review_fingerprint"] or ""
            ).strip(),
            "running_image_digest": str(
                observations["running_image_digest"] or ""
            ).strip(),
            "applied_migration_head": str(
                observations["applied_migration_head"] or ""
            ).strip(),
        }
        for field in (
            "api_desired_count",
            "api_running_count",
            "api_healthy_target_count",
            "api_availability_zone_count",
            "worker_desired_count",
            "worker_running_count",
            "worker_availability_zone_count",
            "database_backup_retention_days",
            "pending_migration_count",
            "migration_checksum_mismatch_count",
            "plaintext_secret_count",
            "alarm_count",
            "alarms_in_alarm_state_count",
        ):
            result[field] = _integer(observations[field], label=field)
        for field in (
            "database_storage_encrypted",
            "database_multi_az",
            "database_publicly_accessible",
            "database_deletion_protection",
            "database_tls_required",
            "runtime_database_roles_separated",
            "database_admin_credentials_in_runtime",
            "secret_injection_verified",
            "alarm_route_verified",
            "deployment_rollback_enabled",
            "production_traffic_enabled",
            "downstream_export_enabled",
            "live_provider_data_used",
        ):
            result[field] = _boolean(observations[field], label=field)
        return result

    def _controls(
        self,
        *,
        request: PreproductionDeploymentCertificationRequest,
        observations: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        api_desired = observations["api_desired_count"]
        worker_desired = observations["worker_desired_count"]
        decisions = {
            "isolated_preproduction_stack_complete": (
                observations["stack_status"] in ALLOWED_STACK_STATUSES
            ),
            "reviewed_change_set_binding_verified": (
                observations["review_fingerprint"]
                == request.infrastructure_review_fingerprint
            ),
            "immutable_candidate_image_running": (
                observations["running_image_digest"]
                == request.candidate_image_digest
            ),
            "api_service_healthy_multi_az": (
                api_desired >= 2
                and observations["api_running_count"] == api_desired
                and observations["api_healthy_target_count"] == api_desired
                and observations["api_availability_zone_count"] >= 2
            ),
            "provider_worker_healthy_multi_az": (
                worker_desired >= 2
                and observations["worker_running_count"] == worker_desired
                and observations["worker_availability_zone_count"] >= 2
            ),
            "database_private_encrypted_multi_az": (
                observations["database_storage_encrypted"]
                and observations["database_multi_az"]
                and not observations["database_publicly_accessible"]
            ),
            "database_tls_backup_and_deletion_protection": (
                observations["database_tls_required"]
                and observations["database_deletion_protection"]
                and observations["database_backup_retention_days"] >= 7
            ),
            "database_migrations_complete_and_immutable": (
                observations["applied_migration_head"]
                == request.expected_migration_head
                and observations["pending_migration_count"] == 0
                and observations["migration_checksum_mismatch_count"] == 0
            ),
            "runtime_database_roles_separated": (
                observations["runtime_database_roles_separated"]
                and not observations["database_admin_credentials_in_runtime"]
            ),
            "secret_injection_without_plaintext": (
                observations["secret_injection_verified"]
                and observations["plaintext_secret_count"] == 0
            ),
            "alarm_routes_and_runtime_alarms_verified": (
                observations["alarm_route_verified"]
                and observations["alarm_count"] >= 4
                and observations["alarms_in_alarm_state_count"] == 0
            ),
            "deployment_rollback_enabled": observations[
                "deployment_rollback_enabled"
            ],
            "production_traffic_disabled": not observations[
                "production_traffic_enabled"
            ],
            "downstream_export_disabled": not observations[
                "downstream_export_enabled"
            ],
            "no_live_provider_data_required": not observations[
                "live_provider_data_used"
            ],
        }
        return [
            {
                "control": name,
                "passed": bool(decisions[name]),
                "evidence_code": (
                    "measured_pass" if decisions[name] else "measured_fail"
                ),
            }
            for name in CONTROL_NAMES
        ]
