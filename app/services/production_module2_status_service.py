from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from app.services.production_module2_certification_service import (
    validate_module2_certification_report,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


class ProductionModule2StatusService:
    """Return configuration/evidence presence without leaking paths or secrets."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        certification = self._read_json("MODULE2_CERTIFICATION_REPORT_PATH")
        shadow = self._read_json("MODULE2_SHADOW_REPORT_PATH")
        index = self._read_json("MODULE2_INDEX_RECORD_PATH")

        certification_valid = False
        if certification is not None:
            try:
                certification = validate_module2_certification_report(
                    certification
                )
                certification_valid = True
            except (TypeError, ValueError):
                certification = None

        engineering_complete = bool(
            certification_valid
            and certification.get("module2_evidence_ready")
        )
        production_ready = bool(
            certification_valid
            and certification.get("production_certification_ready")
        )
        shadow_ready = bool(shadow and shadow.get("shadow_release_ready"))
        index_status = str((index or {}).get("status") or "not_configured")

        routing_enabled = _truthy(
            self._environment.get("MODULE2_PRODUCTION_ROUTING_ENABLED")
        )
        shadow_enabled = _truthy(
            self._environment.get("MODULE2_SHADOW_SERVING_ENABLED")
        )
        registration_enabled = _truthy(
            self._environment.get("MODULE2_MODEL_REGISTRATION_ENABLED")
        )

        if routing_enabled and not production_ready:
            readiness = "unsafe_configuration_production_routing_blocked"
        elif production_ready and index_status == "shadow" and shadow_ready:
            readiness = "ready_for_explicit_operator_activation"
        elif engineering_complete:
            readiness = "engineering_complete_external_release_gates_pending"
        else:
            readiness = "module2_completion_evidence_pending"

        return {
            "module": "module_2_governed_retrieval_and_index_lifecycle",
            "status": readiness,
            "components": {
                "module_2_1_governed_multilingual_retrieval": True,
                "module_2_2_governed_multilingual_taxonomy": True,
                "module_2_3_certification_and_calibration": True,
                "module_2_4_index_lifecycle": True,
                "module_2_5_shadow_serving_and_release_gate": True,
            },
            "evidence_presence": {
                "certification_report": certification is not None,
                "shadow_report": shadow is not None,
                "index_record": index is not None,
            },
            "engineering_evidence_ready": engineering_complete,
            "production_certification_ready": production_ready,
            "shadow_release_ready": shadow_ready,
            "index_status": index_status,
            "feature_flags": {
                "shadow_serving_enabled": shadow_enabled,
                "model_registration_enabled": registration_enabled,
                "production_routing_enabled": routing_enabled,
            },
            "safety": {
                "automatic_proposal_creation_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "raw_identifiers_exposed": False,
                "secret_values_returned": False,
            },
            "remaining_external_gates": [
                "native-language human signoff",
                "299-case zero-failure unsupported calibration",
                "staging deployment with fresh provider data",
                "scale, recovery, privacy, cost, and capacity validation",
                "explicit model-registration and routing approval",
            ],
        }

    def _read_json(self, key: str) -> dict[str, Any] | None:
        raw_path = str(self._environment.get(key) or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None
