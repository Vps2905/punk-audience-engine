from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from app.services.production_security_hardening_service import (
    ProductionSecurityManualReviewService,
    ProductionSecurityPostureService,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionSecurityStatusService:
    """Report security engineering readiness without disclosing secrets."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        posture = self._validated(
            "SECURITY_POSTURE_EVIDENCE_PATH",
            ProductionSecurityPostureService().validate_report,
        )
        review = self._validated(
            "SECURITY_REVIEW_EVIDENCE_PATH",
            ProductionSecurityManualReviewService().validate_report,
        )
        posture_valid = bool(posture and posture.get("posture_status") == "pass")
        review_valid = bool(
            posture_valid
            and review
            and review.get("tenant_id") == posture.get("request", {}).get("tenant_id")
            and review.get("source_security_posture_fingerprint")
            == posture.get("security_posture_fingerprint")
            and review.get("manual_review", {}).get("decision")
            == "approved_for_preproduction_review"
            and review.get("manual_review", {}).get("security_controls_passed") is True
        )
        hardening_enabled = self._flag("SECURITY_HARDENING_ENABLED")
        release_enabled = self._flag("SECURITY_PRODUCTION_RELEASE_ENABLED")
        if release_enabled:
            readiness = "unsafe_configuration_production_release_blocked"
        elif hardening_enabled and not posture_valid:
            readiness = "unsafe_configuration_security_posture_missing_or_failed"
        elif posture_valid and not review_valid:
            readiness = "security_manual_review_pending"
        elif review_valid:
            readiness = "security_engineering_evidence_ready"
        else:
            readiness = "security_evidence_pending"
        return {
            "module": "production_application_and_deployment_security",
            "status": readiness,
            "components": {
                "non_secret_security_posture": posture_valid,
                "manual_preproduction_review": review_valid,
            },
            "security_engineering_evidence_ready": review_valid,
            "live_production_certified": False,
            "evidence_presence": {
                "security_posture": posture is not None,
                "security_review": review is not None,
            },
            "feature_flags": {
                "security_hardening_enabled": hardening_enabled,
                "production_release_enabled": release_enabled,
            },
            "safety": {
                "secret_values_exposed": False,
                "credentials_exposed": False,
                "environment_values_exposed": False,
                "automatic_remediation_performed": False,
                "manual_approval_required": True,
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
