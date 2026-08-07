from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


class ProductionModule3StatusService:
    """Report Module 3.1-3.6 readiness without returning paths or secrets."""

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

        lookalike_evidence = self._read_json(
            "MODULE3_LOOKALIKE_EVIDENCE_PATH"
        )
        lookalike_evidence_valid = bool(
            lookalike_evidence
            and lookalike_evidence.get("status")
            == "engineering_preview_ready"
            and lookalike_evidence.get("safety", {}).get(
                "raw_identifiers_returned"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "audience_membership_read"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "membership_similarity_computed"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "audience_membership_generated"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "overlap_rate_computed"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "unique_reach_claimed"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "cohort_sizes_summed"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "candidate_lifecycle_mutated"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "activation_or_export_performed"
            )
            is False
            and lookalike_evidence.get("safety", {}).get(
                "downstream_export_enabled"
            )
            is False
        )
        module3_3_evidence_ready = bool(
            evidence_valid and overlap_evidence_valid
        )
        module3_4_evidence_ready = bool(
            module3_3_evidence_ready and lookalike_evidence_valid
        )
        lifecycle_evidence = self._read_json(
            "MODULE3_LIFECYCLE_EVIDENCE_PATH"
        )
        lifecycle_evidence_valid = bool(
            lifecycle_evidence
            and lifecycle_evidence.get("status")
            == "engineering_preview_ready"
            and lifecycle_evidence.get("safety", {}).get(
                "raw_identifiers_returned"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "audience_membership_read"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "membership_intersection_read"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "overlap_rate_computed"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "unique_reach_claimed"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "cohort_sizes_summed"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "candidate_lifecycle_mutated"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "automatic_approval_performed"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "manual_approval_required"
            )
            is True
            and lifecycle_evidence.get("safety", {}).get(
                "monitoring_required"
            )
            is True
            and lifecycle_evidence.get("safety", {}).get(
                "shadow_routing_enabled"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "activation_or_export_performed"
            )
            is False
            and lifecycle_evidence.get("safety", {}).get(
                "downstream_export_enabled"
            )
            is False
        )
        module3_5_evidence_ready = bool(
            module3_4_evidence_ready and lifecycle_evidence_valid
        )
        shadow_evidence = self._read_json("MODULE3_SHADOW_EVIDENCE_PATH")
        shadow_evidence_valid = bool(
            shadow_evidence
            and shadow_evidence.get("status") == "engineering_preview_ready"
            and int(shadow_evidence.get("observation_count") or 0) > 0
            and shadow_evidence.get("shadow_alignment_passed") is True
            and shadow_evidence.get("safety", {}).get(
                "raw_identifiers_returned"
            )
            is False
            and shadow_evidence.get("safety", {}).get(
                "audience_membership_read"
            )
            is False
            and shadow_evidence.get("safety", {}).get("proposal_created")
            is False
            and shadow_evidence.get("safety", {}).get("proposal_modified")
            is False
            and shadow_evidence.get("safety", {}).get(
                "candidate_lifecycle_mutated"
            )
            is False
            and shadow_evidence.get("safety", {}).get(
                "automatic_approval_performed"
            )
            is False
            and shadow_evidence.get("safety", {}).get(
                "manual_approval_required"
            )
            is True
            and shadow_evidence.get("safety", {}).get(
                "monitoring_required"
            )
            is True
            and shadow_evidence.get("safety", {}).get(
                "shadow_routing_enabled"
            )
            is False
            and shadow_evidence.get("safety", {}).get(
                "production_routing_enabled"
            )
            is False
            and shadow_evidence.get("safety", {}).get(
                "activation_or_export_performed"
            )
            is False
            and shadow_evidence.get("safety", {}).get(
                "downstream_export_enabled"
            )
            is False
        )
        module3_6_evidence_ready = bool(
            module3_5_evidence_ready and shadow_evidence_valid
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
        lifecycle_evaluation_enabled = _truthy(
            self._environment.get("MODULE3_LIFECYCLE_EVALUATION_ENABLED")
        )
        lifecycle_mutation_enabled = _truthy(
            self._environment.get("MODULE3_LIFECYCLE_MUTATION_ENABLED")
        )
        shadow_observation_enabled = _truthy(
            self._environment.get("MODULE3_SHADOW_OBSERVATION_ENABLED")
        )
        routing_enabled = _truthy(
            self._environment.get("MODULE3_PRODUCTION_ROUTING_ENABLED")
        )

        if routing_enabled or lookalike_enabled or lifecycle_mutation_enabled:
            readiness = "unsafe_configuration_release_affecting_feature_blocked"
        elif overlap_enabled and not module3_3_evidence_ready:
            readiness = "unsafe_configuration_overlap_evidence_missing"
        elif lifecycle_evaluation_enabled and not module3_5_evidence_ready:
            readiness = "unsafe_configuration_lifecycle_evidence_missing"
        elif shadow_observation_enabled and not module3_6_evidence_ready:
            readiness = "unsafe_configuration_shadow_evidence_missing"
        elif module3_6_evidence_ready:
            readiness = "module3_6_engineering_evidence_ready"
        elif module3_5_evidence_ready:
            readiness = "module3_5_engineering_evidence_ready"
        elif module3_4_evidence_ready:
            readiness = "module3_4_engineering_evidence_ready"
        elif module3_3_evidence_ready:
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
                "module_3_4_governed_lookalike_generation": True,
                "module_3_5_lifecycle_approval_and_monitoring": (
                    module3_5_evidence_ready
                ),
                "module_3_6_punk_ai_shadow_integration": (
                    module3_6_evidence_ready
                ),
            },
            "engineering_evidence_ready": evidence_valid,
            "module3_3_engineering_evidence_ready": (
                module3_3_evidence_ready
            ),
            "module3_4_engineering_evidence_ready": (
                module3_4_evidence_ready
            ),
            "module3_5_engineering_evidence_ready": (
                module3_5_evidence_ready
            ),
            "module3_6_engineering_evidence_ready": (
                module3_6_evidence_ready
            ),
            "evidence_presence": {
                "cohort_candidate_report": evidence is not None,
                "overlap_deduplication_report": overlap_evidence is not None,
                "lookalike_review_report": lookalike_evidence is not None,
                "lifecycle_monitoring_report": (
                    lifecycle_evidence is not None
                ),
                "punk_ai_shadow_observation_report": (
                    shadow_evidence is not None
                ),
            },
            "feature_flags": {
                "cohort_generation_enabled": generation_enabled,
                "cohort_persistence_enabled": persistence_enabled,
                "overlap_deduplication_enabled": overlap_enabled,
                "lookalike_generation_enabled": lookalike_enabled,
                "lifecycle_evaluation_enabled": lifecycle_evaluation_enabled,
                "lifecycle_mutation_enabled": lifecycle_mutation_enabled,
                "shadow_observation_enabled": shadow_observation_enabled,
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
                "proposal_modified": False,
                "candidate_lifecycle_mutated": False,
                "automatic_approval_performed": False,
                "manual_approval_required": True,
                "monitoring_required": True,
                "shadow_routing_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "secret_values_returned": False,
            },
            "remaining_engineering_work": [
                "Module 3.3 historical overlap evidence review",
                "Module 3.4 historical lookalike evidence review",
                "Module 3.5 historical lifecycle evidence review",
                "Module 3.6 historical Punk AI shadow evidence review",
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
