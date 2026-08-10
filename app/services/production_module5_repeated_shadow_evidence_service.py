from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_bounded_autonomy_certification_contracts import (
    ShadowCertificationCase,
)
from app.models.production_bounded_autonomy_contracts import AutonomyGoal
from app.services.production_bounded_autonomy_certification_service import (
    ProductionBoundedAutonomyCertificationService,
)
from app.services.production_bounded_autonomy_shadow_service import (
    LegacyOrchestratorObservationAdapter,
)


_MAX_RESULT_BYTES = 10_000_000
_RELEASE_EFFECT_FLAGS = (
    "MODULE3_PRODUCTION_ROUTING_ENABLED",
    "MODULE4_PRODUCTION_ROUTING_ENABLED",
    "MODULE5_PRODUCTION_ROUTING_ENABLED",
    "MODULE5_AGENT_PRODUCTION_EFFECT_AUTHORIZATION_ENABLED",
    "MODULE5_SCALE_RECOVERY_PRODUCTION_CUTOVER_ENABLED",
    "SECURITY_PRODUCTION_RELEASE_ENABLED",
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class ProductionModule5RepeatedShadowEvidenceService:
    """Build Module 5.8 evidence from real, persisted safe run summaries."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        certification_service: Any | None = None,
        observation_adapter: LegacyOrchestratorObservationAdapter | None = None,
    ) -> None:
        self.environment = dict(os.environ if environment is None else environment)
        self.certification_service = (
            certification_service or ProductionBoundedAutonomyCertificationService(
                environment=self.environment
            )
        )
        self.observation_adapter = (
            observation_adapter or LegacyOrchestratorObservationAdapter()
        )

    def run(
        self,
        *,
        tenant_id: str,
        result_paths: Sequence[Path],
        maximum_cases: int = 10_000,
    ) -> dict[str, Any]:
        clean_tenant = normalize_taxonomy_value(tenant_id)
        if not clean_tenant:
            raise ValueError("tenant_id is required.")
        self._assert_release_effects_disabled()
        bounded_maximum = int(maximum_cases)
        if not 1 <= bounded_maximum <= 10_000:
            raise ValueError("maximum_cases must be between 1 and 10000.")

        paths = self._unique_paths(result_paths)
        if not paths:
            raise ValueError("At least one historical result is required.")
        if len(paths) > bounded_maximum:
            raise ValueError("Historical result count exceeds maximum_cases.")

        cases = tuple(
            self._case_from_path(
                path=path,
                tenant_id=clean_tenant,
                case_index=index,
            )
            for index, path in enumerate(paths)
        )
        report = self.certification_service.run(
            tenant_id=clean_tenant,
            cases=cases,
        )
        return self.certification_service.validate_report(report)

    def inventory(
        self,
        *,
        tenant_id: str,
        result_paths: Sequence[Path],
        maximum_cases: int = 10_000,
    ) -> dict[str, Any]:
        clean_tenant = normalize_taxonomy_value(tenant_id)
        if not clean_tenant:
            raise ValueError("tenant_id is required.")
        self._assert_release_effects_disabled()
        bounded_maximum = int(maximum_cases)
        if not 1 <= bounded_maximum <= 10_000:
            raise ValueError("maximum_cases must be between 1 and 10000.")
        paths = self._unique_paths(result_paths)
        if not paths:
            raise ValueError("At least one historical result is required.")
        if len(paths) > bounded_maximum:
            raise ValueError("Historical result count exceeds maximum_cases.")
        cases = [
            self._case_from_path(
                path=path,
                tenant_id=clean_tenant,
                case_index=index,
            )
            for index, path in enumerate(paths)
        ]
        observations = [
            self.observation_adapter.adapt(
                goal=case.goal,
                legacy_result=case.legacy_result,
            )
            for case in cases
        ]
        return {
            "historical_result_count": len(cases),
            "unique_objective_count": len(
                {case.goal.objective_sha256 for case in cases}
            ),
            "terminal_case_count": sum(value.terminal for value in observations),
            "nonterminal_review_case_count": sum(
                not value.terminal for value in observations
            ),
            "unsafe_result_count": sum(
                value.raw_identifiers_returned
                or value.activation_or_export_performed
                or value.downstream_export_enabled
                for value in observations
            ),
            "prompt_content_stored": False,
            "source_paths_stored": False,
        }

    def _assert_release_effects_disabled(self) -> None:
        unsafe_flags = [
            name
            for name in _RELEASE_EFFECT_FLAGS
            if _truthy(self.environment.get(name))
        ]
        if unsafe_flags:
            raise RuntimeError(
                "Module 5.8 evidence generation requires every release-effect "
                "flag to remain disabled."
            )

    def _case_from_path(
        self,
        *,
        path: Path,
        tenant_id: str,
        case_index: int,
    ) -> ShadowCertificationCase:
        payload = self._read_result(path)
        declared_tenant = normalize_taxonomy_value(payload.get("tenant_id"))
        if declared_tenant and declared_tenant != tenant_id:
            raise ValueError("Historical result tenant lineage mismatch.")
        objective = " ".join(str(payload.get("prompt") or "").split())
        if not objective:
            raise ValueError(
                "Historical result does not contain its original prompt."
            )
        digest = hashlib.sha256(
            (
                f"{tenant_id}|{case_index}|{path.name}|"
                f"{payload.get('run_id') or ''}|{objective}"
            ).encode("utf-8")
        ).hexdigest()
        request_id = f"shadow-request-{digest[:32]}"
        goal = AutonomyGoal(
            tenant_id=tenant_id,
            request_id=request_id,
            goal_id=f"shadow-goal-{digest[32:64]}",
            objective=objective,
            requested_outcomes=("governed_recommendation",),
            execution_mode="shadow",
            constraints=(
                "historical_evidence_only",
                "manual_approval_required",
            ),
        )
        minimized = dict(payload)
        minimized["run_id"] = f"historical-run-{digest[:32]}"
        observation = self.observation_adapter.adapt(
            goal=goal,
            legacy_result=minimized,
        )
        if (
            observation.raw_identifiers_returned
            or observation.activation_or_export_performed
            or observation.downstream_export_enabled
        ):
            raise ValueError("Historical result contains an unsafe effect signal.")
        return ShadowCertificationCase(
            goal=goal,
            legacy_result=minimized,
            legacy_latency_ms=self._latency(payload),
        )

    @staticmethod
    def _read_result(path: Path) -> dict[str, Any]:
        candidate = Path(path)
        if candidate.is_symlink():
            raise ValueError("Historical result cannot be a symbolic link.")
        if not candidate.is_file():
            raise FileNotFoundError(f"Historical result was not found: {candidate}")
        if candidate.stat().st_size > _MAX_RESULT_BYTES:
            raise ValueError("Historical result exceeds the 10 MB safety limit.")
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Historical result must contain a JSON object.")
        return value

    @staticmethod
    def _latency(payload: Mapping[str, Any]) -> float | None:
        for field, multiplier in (
            ("total_latency_ms", 1.0),
            ("latency_ms", 1.0),
            ("duration_ms", 1.0),
            ("duration_seconds", 1000.0),
        ):
            value = payload.get(field)
            if value is None:
                continue
            try:
                parsed = float(value) * multiplier
            except (TypeError, ValueError):
                continue
            if 0.0 <= parsed <= 3_600_000.0:
                return parsed
        return None

    @staticmethod
    def _unique_paths(result_paths: Sequence[Path]) -> tuple[Path, ...]:
        output: list[Path] = []
        seen: set[Path] = set()
        for raw in result_paths:
            path = Path(raw).expanduser().absolute()
            if path not in seen:
                output.append(path)
                seen.add(path)
        return tuple(sorted(output, key=str))


def discover_historical_prompt_results(
    roots: Sequence[Path],
) -> tuple[Path, ...]:
    """Discover only canonical final prompt summaries below explicit roots."""

    output: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().absolute()
        if root.is_symlink():
            raise ValueError("Historical run root cannot be a symbolic link.")
        if not root.is_dir():
            raise FileNotFoundError(f"Historical run root was not found: {root}")
        output.extend(root.rglob("final_prompt_summary.json"))
    return ProductionModule5RepeatedShadowEvidenceService._unique_paths(output)
