from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from app.services.production_infrastructure_governance_service import (
    ProductionInfrastructureAssessmentService,
    ProductionInfrastructureChangeSetReviewService,
)
from app.services.production_preproduction_deployment_certification_service import (
    ProductionPreproductionDeploymentCertificationService,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionInfrastructureStatusService:
    """Report IaC engineering readiness without modifying cloud resources."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        assessment = self._validated(
            "INFRASTRUCTURE_ASSESSMENT_EVIDENCE_PATH",
            ProductionInfrastructureAssessmentService().validate_report,
        )
        review = self._validated(
            "INFRASTRUCTURE_REVIEW_EVIDENCE_PATH",
            ProductionInfrastructureChangeSetReviewService().validate_report,
        )
        assessment_valid = bool(
            assessment and assessment.get("assessment_status") == "pass"
        )
        review_valid = bool(
            assessment_valid
            and review
            and review.get("tenant_id")
            == assessment.get("request", {}).get("tenant_id")
            and review.get("source_infrastructure_assessment_fingerprint")
            == assessment.get("infrastructure_assessment_fingerprint")
            and review.get("manual_review", {}).get("decision")
            == "approved_for_preproduction_change_set"
            and review.get("manual_review", {}).get(
                "infrastructure_controls_passed"
            ) is True
        )
        deployment_path_key = "PREPRODUCTION_DEPLOYMENT_CERTIFICATION_PATH"
        deployment_configured = bool(
            str(self._environment.get(deployment_path_key) or "").strip()
        )
        deployment = self._validated(
            deployment_path_key,
            ProductionPreproductionDeploymentCertificationService().validate_report,
        )
        deployment_valid = bool(
            review_valid
            and deployment
            and deployment.get("preproduction_deployment_certified") is True
            and deployment.get("request", {}).get("tenant_id")
            == assessment.get("request", {}).get("tenant_id")
            and deployment.get("request", {}).get(
                "infrastructure_review_fingerprint"
            )
            == review.get("infrastructure_review_fingerprint")
        )
        validation_enabled = self._flag("INFRASTRUCTURE_VALIDATION_ENABLED")
        apply_enabled = self._flag("INFRASTRUCTURE_APPLY_ENABLED")
        traffic_enabled = self._flag("INFRASTRUCTURE_PRODUCTION_TRAFFIC_ENABLED")
        if apply_enabled or traffic_enabled:
            readiness = "unsafe_configuration_infrastructure_mutation_blocked"
        elif deployment_configured and not deployment_valid:
            readiness = (
                "preproduction_deployment_certification_failed_closed"
            )
        elif deployment_valid:
            readiness = "preproduction_deployment_certified"
        elif validation_enabled and not assessment_valid:
            readiness = "unsafe_configuration_infrastructure_evidence_missing"
        elif assessment_valid and not review_valid:
            readiness = "infrastructure_change_set_review_pending"
        elif review_valid:
            readiness = "infrastructure_engineering_evidence_ready"
        else:
            readiness = "infrastructure_evidence_pending"
        return {
            "module": "production_infrastructure_as_code",
            "status": readiness,
            "components": {
                "runtime_plane_assessment": assessment_valid,
                "manual_change_set_review": review_valid,
            },
            "certification_components": {
                "measured_preproduction_deployment": deployment_valid,
            },
            "infrastructure_engineering_evidence_ready": review_valid,
            "preproduction_deployment_certified": deployment_valid,
            "live_production_certified": False,
            "evidence_presence": {
                "infrastructure_assessment": assessment is not None,
                "infrastructure_review": review is not None,
                "preproduction_deployment_certification": (
                    deployment is not None
                ),
            },
            "feature_flags": {
                "validation_enabled": validation_enabled,
                "apply_enabled": apply_enabled,
                "production_traffic_enabled": traffic_enabled,
            },
            "safety": {
                "secret_values_exposed": False,
                "resource_identifiers_exposed": False,
                "cloud_resources_mutated": False,
                "change_set_executed": False,
                "automatic_deployment_performed": False,
                "manual_approval_required": True,
                "production_traffic_enabled": False,
                "production_release_authorized": False,
            },
        }

    def _flag(self, key: str) -> bool:
        return _truthy(self._environment.get(key))

    def _validated(
        self,
        key: str,
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any] | None:
        value = self._read_json(key)
        if value is None:
            return None
        try:
            return validator(value)
        except (TypeError, ValueError):
            return None

    def _read_json(self, key: str) -> dict[str, Any] | None:
        raw = str(self._environment.get(key) or "").strip()
        if not raw or not Path(raw).is_file():
            return None
        try:
            value = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, Mapping) else None
