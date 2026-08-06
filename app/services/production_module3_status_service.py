from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


class ProductionModule3StatusService:
    """Report Module 3.1-3.3 readiness without returning paths or secrets."""

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def status(self) -> dict[str, Any]:
        evidence = self._read_json("MODULE3_COHORT_EVIDENCE_PATH")
        evidence_valid = bool(
            evidence
            and evidence.get("status") == "engineering_preview_ready"
            and evidence.get("safety", {}).get("raw_identifiers_returned") is False
            and evidence.get("safety", {}).get("activation_or_export_performed")
            is False
        )
        overlap_evidence = self._read_json("MODULE3_OVERLAP_EVIDENCE_PATH")
        overlap_evidence_valid = bool(
            overlap_evidence
            and overlap_evidence.get("status") == "engineering_preview_ready"
            and overlap_evidence.get("safety", {}).get("raw_identifiers_returned")
            is False
            and overlap_evidence.get("safety", {}).get(
                "membership_intersection_read"
            )
            is False
            and overlap_evidence.get("safety", {}).get("overlap_rate_computed")
            is False
            and overlap_evidence.get("safety", {}).get("unique_reach_claimed")
            is False
            and overlap_evidence.get("safety", {}).get("cohort_sizes_summed")
            is False
            and overlap_evidence.get("safety", {}).get(
                "candidate_lifecycle_mutated"
            )
            is False
            and overlap_evidence.get("safety", {}).get(
                "activation_or_export_performed"
            )
            is False
        )

        generation_enabled = _truthy(
            self._environment.get("MODULE3_COHORT_GENERATION_ENABLED")
        )
        persistence_enabled = _truthy(
            self._environment.get("MODULE3_COHORT_PERSISTENCE_ENABLED")
        )
        overlap_enabled = _truthy(
            self._environment.get("MODULE3_OVERLAP_DEDUPLICATION_ENABLED")
        )
        lookalike_enabled = _truthy(
            self._environment.get("MODULE3_LOOKALIKE_GENERATION_ENABLED")
        )
        routing_enabled = _truthy(
            self._environment.get("MODULE3_PRODUCTION_ROUTING_ENABLED")
        )

        if routing_enabled or lookalike_enabled:
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif overlap_enabled and not (evidence_valid and overlap_evidence_valid):
            readiness = "unsafe_configuration_overlap_evidence_missing"
        elif evidence_valid and overlap_evidence_valid:
            readiness = "module3_3_engineering_evidence_ready"
        elif evidence_valid:
            readiness = "module3_1_2_engineering_evidence_ready"
        else:
            readiness = "module3_1_2_evidence_pending"

        return {
            "module": "module_3_governed_cohort_intelligence",
            "status": readiness,
            "components": {
                "module_3_1_cohort_contracts_and_schema": True,
                "module_3_2_candidate_generation_and_quality_scoring": True,
                "module_3_3_overlap_and_deduplication": True,
                "module_3_4_governed_lookalike_generation": False,
                "module_3_5_lifecycle_approval_and_monitoring": False,
                "module_3_6_punk_ai_shadow_integration": False,
            },
            "engineering_evidence_ready": evidence_valid,
            "module3_3_engineering_evidence_ready": overlap_evidence_valid,
            "evidence_presence": {
                "cohort_candidate_report": evidence is not None,
                "overlap_deduplication_report": overlap_evidence is not None,
            },
            "feature_flags": {
                "cohort_generation_enabled": generation_enabled,
                "cohort_persistence_enabled": persistence_enabled,
                "overlap_deduplication_enabled": overlap_enabled,
                "lookalike_generation_enabled": lookalike_enabled,
                "production_routing_enabled": routing_enabled,
            },
            "safety": {
                "minimum_cohort_size": 1000,
                "raw_identifiers_exposed": False,
                "overlap_or_unique_reach_claimed": False,
                "membership_intersection_read": False,
                "overlap_rate_computed": False,
                "cohort_sizes_summed": False,
                "lookalike_generation_performed": False,
                "automatic_proposal_creation_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "secret_values_returned": False,
            },
            "remaining_engineering_work": [
                "Module 3.3 historical overlap evidence review",
                "governed lookalike generation",
                "cohort lifecycle approval and monitoring",
                "Punk AI shadow integration",
                "fresh-data staging validation",
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
