from __future__ import annotations

import inspect
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.agents.privacy_layer_agent import PrivacyLayerAgent
from app.agents.synthetic_engine_agent import SyntheticEngineAgent
from app.agents.embedding_feature_store_agent import EmbeddingFeatureStoreAgent
from app.agents.cohort_management_agent import CohortManagementAgent
from app.agents.safe_export_agent import SafeExportAgent
from app.utils.location_matcher import location_matches_request
from app.agents.autonomous_v2_swarm_review_agent import AutonomousV2SwarmReviewAgent
from app.services.autonomous_audience_intelligence_v2_service import (
    AutonomousAudienceIntelligenceV2Service,
)
from app.services.embedding_service import embed_records
from app.services.vector_store_service import load_vector_store
from app.core.production_guardrails import local_file_storage_allowed
from app.services.audience_proposal_request_safety_service import (
    AudienceProposalRequestSafetyService,
)


class AudienceIntelligenceOrchestratorAgent:
    """
    Runs the Audience Intelligence pipeline from a business prompt.

    v1 flow remains:
    Postgres/safe input
      -> PrivacyLayerAgent
      -> prompt cohort selection
      -> SyntheticEngineAgent
      -> EmbeddingFeatureStoreAgent
      -> CohortManagementAgent
      -> SafeExportAgent

    v2 sidecar:
      -> all privacy-safe cohorts
      -> semantic prompt intent
      -> all-safe-cohort 384-dim embeddings
      -> dynamic ranking
      -> mutation/data-gap suggestions
      -> swarm review

    v2-guided selection:
      The export path can now use v2-ranked relevant cohorts so broad prompts do
      not export unrelated high-quality cohorts such as gas_station/casino/car_wash.
    """

    def run(
        self,
        prompt: str,
        output_root: str | Path = "data/prompt_runs",
        source: str = "postgres",
        safe_cohort_path: Optional[str | Path] = None,
        postgres_limit: int = 10000,
        k_min: int = 1000,
        epsilon: float = 1.0,
        synthetic_rows: int = 1000,
        max_export_cohorts: int = 25,
        min_export_quality: float = 0.25,
        approval_required: bool = True,
        semantic_intent: Optional[Dict[str, Any]] = None,
        persist_artifacts: Optional[bool] = None,
        certification_evaluation: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        run_id = self._build_run_id(prompt)
        run_dir = Path(output_root) / run_id
        persist_artifacts = (
            local_file_storage_allowed()
            if persist_artifacts is None
            else bool(persist_artifacts)
        )
        if persist_artifacts and not local_file_storage_allowed():
            raise RuntimeError(
                "Local artifact persistence is disabled in this environment."
            )
        if certification_evaluation and (
            persist_artifacts or approval_required is not True
        ):
            raise ValueError(
                "Certification evaluation requires in-memory execution and "
                "manual approval."
            )

        if persist_artifacts:
            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        run_reference = (
            str(run_dir)
            if persist_artifacts
            else (
                "postgres://audience_run_history"
                f"?run_id={run_id}"
            )
        )
        final_summary_path = run_dir / "final_prompt_summary.json"
        final_summary_reference = (
            str(final_summary_path)
            if persist_artifacts
            else (
                "postgres://audience_run_history.final_summary"
                f"?run_id={run_id}"
            )
        )

        print("RUN ID:", run_id)
        print("PROMPT:", prompt)
        print("RUN DIR:", run_dir)
        print()

        safety_decision = (
            AudienceProposalRequestSafetyService().evaluate(
                {
                    "audience_intent": prompt,
                }
            )
        )
        if safety_decision.terminal:
            prompt_filter_report = (
                self._build_preflight_prompt_filter_report(
                    reason_code=str(
                        safety_decision.reason_code or ""
                    ),
                    explanation=str(
                        safety_decision.explanation or ""
                    ),
                )
            )
            return self._build_terminal_prompt_safety_result(
                prompt=prompt,
                run_id=run_id,
                run_reference=run_reference,
                final_summary_reference=final_summary_reference,
                final_summary_path=final_summary_path,
                persist_artifacts=persist_artifacts,
                source_mode="not_evaluated",
                source_rows=None,
                source_columns=[],
                privacy_cohorts=pd.DataFrame(),
                prompt_filter_report=prompt_filter_report,
            )

        if source == "postgres":
            safe_raw_input = self._load_safe_rows_from_postgres(
                prompt=prompt,
                limit=postgres_limit,
            )
            source_mode = "postgres_safe_derived"
        elif source == "safe_artifact":
            if not safe_cohort_path:
                raise ValueError("--safe-cohort-path is required when source=safe_artifact")
            safe_raw_input = pd.read_csv(safe_cohort_path)
            source_mode = "existing_safe_artifact"
        else:
            raise ValueError("source must be postgres or safe_artifact")

        print("SOURCE MODE:", source_mode)
        print("SOURCE ROWS:", len(safe_raw_input))
        print("SOURCE COLUMNS:", list(safe_raw_input.columns))
        print()

        privacy_dir = run_dir / "01_privacy"
        synthetic_dir = run_dir / "02_synthetic"
        embedding_dir = run_dir / "03_embeddings"
        cohort_dir = run_dir / "04_cohort_management"
        export_dir = run_dir / "05_safe_export"
        v2_dir = run_dir / "06_v2_autonomous_preview"

        if source == "postgres":
            privacy_result = self._run_privacy_layer(
                safe_raw_input=safe_raw_input,
                output_dir=privacy_dir,
                run_id=f"{run_id}_privacy",
                k_min=k_min,
                epsilon=epsilon,
                persist_artifacts=persist_artifacts,
                record_privacy_budget=not certification_evaluation,
            )

            privacy_records = (
                privacy_result.get("records") or {}
            ).get("safe_cohorts")

            if isinstance(privacy_records, list):
                privacy_cohorts = pd.DataFrame(privacy_records)
            else:
                privacy_feature_path = self._resolve_privacy_feature_path(
                    privacy_result=privacy_result,
                    privacy_dir=privacy_dir,
                )
                privacy_cohorts = pd.read_csv(privacy_feature_path)
        else:
            privacy_feature_path = privacy_dir / "clean_feature_table.csv"
            if persist_artifacts:
                privacy_dir.mkdir(parents=True, exist_ok=True)
                safe_raw_input.to_csv(privacy_feature_path, index=False)
            privacy_result = {
                "status": "skipped_existing_safe_artifact",
                "feature_path": str(privacy_feature_path),
            }
            privacy_cohorts = safe_raw_input.copy()

        selected_cohorts, prompt_filter_report = self._select_cohorts_for_prompt(
            prompt=prompt,
            cohorts=privacy_cohorts,
            semantic_intent=semantic_intent,
        )

        selected_path = privacy_dir / "prompt_selected_cohorts.csv"

        if persist_artifacts:
            selected_cohorts.to_csv(
                selected_path,
                index=False,
            )

        print("PRIVACY COHORTS:", len(privacy_cohorts))
        print("PROMPT SELECTED COHORTS:", len(selected_cohorts))
        print("PROMPT FILTER MODE:", prompt_filter_report["filter_mode"])
        print()

        if self._is_terminal_prompt_safety_report(prompt_filter_report):
            return self._build_terminal_prompt_safety_result(
                prompt=prompt,
                run_id=run_id,
                run_reference=run_reference,
                final_summary_reference=final_summary_reference,
                final_summary_path=final_summary_path,
                persist_artifacts=persist_artifacts,
                source_mode=source_mode,
                source_rows=len(safe_raw_input),
                source_columns=list(safe_raw_input.columns),
                privacy_cohorts=privacy_cohorts,
                prompt_filter_report=prompt_filter_report,
            )

        try:
            v2_result = AutonomousAudienceIntelligenceV2Service().run(
                prompt=prompt,
                safe_cohorts=privacy_cohorts,
                output_dir=v2_dir,
                freshness_source_df=safe_raw_input,
                persist_artifacts=persist_artifacts,
                semantic_intent=semantic_intent,
            )
            print("V2 AUTONOMOUS STATUS:", v2_result.get("status"))
            print("V2 VECTOR COUNT:", v2_result.get("embedding_manifest", {}).get("vector_count"))
            print("V2 VECTOR DIM:", v2_result.get("embedding_manifest", {}).get("vector_dimension"))
            print("V2 RANKED MATCHES:", v2_result.get("ranked_match_count"))
            print("V2 MUTATION SUGGESTIONS:", v2_result.get("mutation", {}).get("suggestion_count"))
            print()
        except Exception as exc:
            v2_result = {
                "status": "failed",
                "pipeline_version": "v2_autonomous_preview",
                "error": str(exc),
                "approval_status": "blocked_v2_failure",
                "block_export": True,
                "approval_required": True,
                "downstream_export_enabled": False,
                "coverage_warnings": [
                    (
                        "Autonomous Audience Intelligence v2 failed. "
                        "Audience approval and downstream delivery are "
                        "blocked until the V2 stage completes successfully."
                    )
                ],
            }
            print("V2 AUTONOMOUS STATUS: failed")
            print("V2 ERROR:", exc)
            print()

        if semantic_intent:
            prompt_intent = v2_result.setdefault(
                "prompt_intent",
                {},
            )

            existing_quality = str(
                prompt_intent.get("quality_intent") or ""
            ).strip().lower()

            fallback_quality = str(
                semantic_intent.get("quality_intent") or ""
            ).strip().lower()

            if (
                existing_quality
                not in {"high", "balanced", "broad"}
                and fallback_quality
                in {"high", "balanced", "broad"}
            ):
                prompt_intent["quality_intent"] = (
                    fallback_quality
                )

        prompt_filter_report = self._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report=prompt_filter_report,
            v2_result=v2_result,
        )

        freshness_guardrail = self._build_freshness_guardrail(v2_result)

        v2_guided_selection_report = {
            "enabled": False,
            "reason": "not_attempted",
            "rows": 0,
        }

        try:
            v2_selected_cohorts, v2_guided_selection_report = self._select_cohorts_from_v2_ranked(
                privacy_cohorts=privacy_cohorts,
                v2_result=v2_result,
                prompt_filter_report=prompt_filter_report,
                max_rows=max_export_cohorts,
            )

            prompt_filter_report[
                "v2_guided_selection"
            ] = v2_guided_selection_report

            v2_selection_authoritative = bool(
                v2_guided_selection_report.get(
                    "enabled"
                )
                or v2_guided_selection_report.get(
                    "block_export"
                )
                or v2_guided_selection_report.get(
                    "filter_mode"
                )
                in {
                    "needs_clarification",
                    "location_category_gap_no_export",
                    "broad_location_no_export",
                }
            )

            if v2_selection_authoritative:
                selected_cohorts = (
                    v2_selected_cohorts
                )

                if persist_artifacts:
                    selected_cohorts.to_csv(
                        selected_path,
                        index=False,
                    )

            if len(v2_selected_cohorts) >= 1:
                print(
                    "V2 GUIDED SELECTION:",
                    v2_guided_selection_report,
                )
                print(
                    "V2 GUIDED SELECTED COHORTS:",
                    len(selected_cohorts),
                )
            else:
                print(
                    "V2 GUIDED SELECTION BLOCKED/SKIPPED:",
                    v2_guided_selection_report,
                )

            print()
        except Exception as exc:
            v2_guided_selection_report = {
                "enabled": False,
                "reason": "failed",
                "error": str(exc),
                "rows": 0,
            }
            prompt_filter_report["v2_guided_selection"] = v2_guided_selection_report
            print("V2 GUIDED SELECTION FAILED:", exc)
            print()

        synthetic_agent = SyntheticEngineAgent()
        # If strict prompt/v2 guardrails selected zero safe cohorts, stop here.
        # This is a valid no-export result, not a pipeline error.
        _selected_for_downstream = (
            locals().get("prompt_selected_cohorts")
            if locals().get("prompt_selected_cohorts") is not None
            else locals().get("selected_cohorts")
        )
        if _selected_for_downstream is None:
            _selected_for_downstream = locals().get("v2_selected_cohorts")

        _prompt_filter_report = locals().get("prompt_filter_report") or {}
        _v2_guided_selection_report = locals().get("v2_guided_selection_report") or {}
        _block_export = bool(
            _prompt_filter_report.get("block_export")
            or _v2_guided_selection_report.get("block_export")
            or _prompt_filter_report.get("filter_mode") in {
                "location_category_gap_no_export",
                "broad_location_no_export",
            }
            or _v2_guided_selection_report.get("filter_mode") == "location_category_gap_no_export"
        )

        _selected_empty = (
            _selected_for_downstream is None
            or getattr(_selected_for_downstream, "empty", False)
            or len(_selected_for_downstream) == 0
        )

        if _block_export or _selected_empty:
            _run_dir = Path(str(locals().get("run_dir")))

            if persist_artifacts:
                _run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            _coverage_warnings = list(locals().get("coverage_warnings") or [])
            for _warning in (_prompt_filter_report.get("coverage_warnings") or []):
                if _warning not in _coverage_warnings:
                    _coverage_warnings.append(_warning)
            for _warning in (_v2_guided_selection_report.get("coverage_warnings") or []):
                if _warning not in _coverage_warnings:
                    _coverage_warnings.append(_warning)

            _no_match_swarm_review = self._build_no_match_swarm_review(
                coverage_warnings=_coverage_warnings,
            )

            _final_summary_path = _run_dir / "final_summary.md"
            _status_message = (
                (
                    "More information is required before "
                    "an audience can be selected. Provide "
                    "a location and business/category."
                )
                if _prompt_filter_report.get(
                    "filter_mode"
                )
                == "needs_clarification"
                else (
                    "No export-ready cohort was created "
                    "because the requested location/"
                    "category/daypart combination has no "
                    "exact safe cohort."
                )
            )

            _final_summary = "\n".join(
                [
                    "# Audience Intelligence Result",
                    "",
                    f"Prompt: {prompt}",
                    f"Run ID: {run_id}",
                    "",
                    "## Status",
                    "",
                    _status_message,
                    "",
                    f"Filter mode: {_prompt_filter_report.get('filter_mode') or _v2_guided_selection_report.get('filter_mode')}",
                    f"Downstream export enabled: False",
                    "",
                    "## Coverage warnings",
                    "",
                    *[f"- {_warning}" for _warning in _coverage_warnings],
                    "",
                ]
            )
            if persist_artifacts:
                _final_summary_path.write_text(_final_summary)

            return {
                "status": "completed",
                "run_id": run_id,
                "prompt": prompt,
                "run_dir": run_reference,
                "source_mode": locals().get("source_mode"),
                "source_rows": locals().get("source_rows"),
                "source_columns": locals().get("source_columns"),
                "privacy_cohorts": int(len(locals().get("privacy_cohorts"))) if locals().get("privacy_cohorts") is not None else 0,
                "prompt_selected_cohorts": 0,
                "prompt_filter_report": _prompt_filter_report,
                "coverage_warnings": _coverage_warnings,
                "final_summary_path": (
                    str(_final_summary_path)
                    if persist_artifacts
                    else final_summary_reference
                ),
                "v2_autonomous": locals().get("v2_result") or {},
                "v2_swarm_review": _no_match_swarm_review,
                "v2_guided_selection_report": _v2_guided_selection_report,
                "safe_export": {
                    "approval_status": "blocked_no_safe_exact_match",
                    "downstream_export_enabled": False,
                    "exported_cohorts": 0,
                    "exported_lookalike_pairs": 0,
                    "outputs": {},
                },
                "privacy_guarantees": {
                    "raw_maids_exported": False,
                    "hashed_identifiers_exported": False,
                    "raw_observations_exported": False,
                    "raw_lat_lng_exported": False,
                    "raw_email_exported": False,
                    "raw_phone_exported": False,
                    "individual_user_data_exported": False,
                },
            }

        synthetic_result = self._call_agent_method(
            agent=synthetic_agent,
            method_names=["generate"],
            kwargs={
                "cohorts": selected_cohorts,
                "df": selected_cohorts,
                "data": selected_cohorts,
                "output_dir": synthetic_dir,
                "run_id": f"{run_id}_synthetic",
                "engine_requested": "dp_aggregate",
                "engine": "dp_aggregate",
                "production_mode": True,
                "allow_fallback": False,
                "synthetic_rows": synthetic_rows,
                "rows": synthetic_rows,
                "epsilon": epsilon,
                "k_min": k_min,
                "persist_artifacts": persist_artifacts,
                "record_privacy_budget": not certification_evaluation,
            },
        )

        print("SYNTHETIC STATUS:", synthetic_result.get("status"))
        print("SYNTHETIC ENGINE:", synthetic_result.get("engine") or synthetic_result.get("engine_used"))
        print()

        embedding_result, cohort_result = self._run_embedding_and_cohort_management(
            selected_cohorts=selected_cohorts,
            embedding_dir=embedding_dir,
            cohort_dir=cohort_dir,
            run_id=run_id,
            min_export_quality=min_export_quality,
            persist_artifacts=persist_artifacts,
        )

        print("EMBEDDING STATUS:", embedding_result["status"])
        print("EMBEDDING BACKEND:", embedding_result.get("embedding_provider"))
        print("VECTOR BACKEND:", embedding_result.get("vector_backend"))
        print("VECTOR COUNT:", embedding_result["vector_count"])
        print("VECTOR DIM:", embedding_result["vector_dimension"])
        print()

        print("COHORT MANAGEMENT STATUS:", cohort_result["status"])
        print("MANAGED COHORTS:", cohort_result["managed_cohorts"])
        print("CLUSTERS:", cohort_result["cluster_count"])
        print("EXPORT READY:", cohort_result["export_ready_cohorts"])
        print()

        cohort_records = cohort_result.get("records") or {}
        top_cohorts_for_export = pd.DataFrame(
            cohort_records.get("top_cohorts") or []
        )
        lookalikes_for_export = pd.DataFrame(
            cohort_records.get("lookalikes") or []
        )

        # STAGE 2 — POST-MANAGEMENT FINAL POLICY
        if not top_cohorts_for_export.empty:
            quality_intent = (prompt_filter_report or {}).get("quality_intent") or "balanced"
            requested_locations = [
                str(loc).strip().lower() for loc in (
                    (prompt_filter_report or {}).get("locations_detected")
                    or (prompt_filter_report or {}).get("locations")
                    or []
                ) if str(loc).strip()
            ]

            from app.services.audience_quality_policy_service import AudienceQualityPolicyService
            quality_policy_svc = AudienceQualityPolicyService()
            top_cohorts_for_export, final_quality_report = quality_policy_svc.apply_policy(
                candidates=top_cohorts_for_export,
                quality_intent=quality_intent,
                requested_locations=requested_locations,
            )

            prompt_filter_report["quality_policy_report"] = final_quality_report
            prompt_filter_report["quality_intent"] = quality_intent

            if persist_artifacts:
                final_quality_path = cohort_dir / "quality_qualified_cohorts.csv"
                top_cohorts_for_export.to_csv(final_quality_path, index=False)

            # Filter lookalikes to match retained cohorts
            if not lookalikes_for_export.empty and not top_cohorts_for_export.empty:
                if "cohort_id" in lookalikes_for_export.columns and "cohort_id" in top_cohorts_for_export.columns:
                    valid_cohort_ids = set(top_cohorts_for_export["cohort_id"])
                    lookalikes_for_export = lookalikes_for_export[
                        lookalikes_for_export["cohort_id"].isin(valid_cohort_ids)
                    ].copy()
                elif "cluster_id" in lookalikes_for_export.columns and "cluster_id" in top_cohorts_for_export.columns:
                    valid_cluster_ids = set(top_cohorts_for_export["cluster_id"])
                    lookalikes_for_export = lookalikes_for_export[
                        lookalikes_for_export["cluster_id"].isin(valid_cluster_ids)
                    ].copy()

            # recalculate fulfillment based on final quality
            missing_requested_locations = []
            matched_requested_locations = []
            if requested_locations and "location_name" in top_cohorts_for_export.columns:
                for requested_location in requested_locations:
                    has_location = any(
                        location_matches_request(loc, requested_location)
                        for loc in top_cohorts_for_export["location_name"].dropna().tolist()
                    )
                    if not has_location:
                        missing_requested_locations.append(str(requested_location))
                    else:
                        matched_requested_locations.append(str(requested_location))

            if (
                prompt_filter_report.get(
                    "eligible_for_audience_selection"
                )
                is False
                or prompt_filter_report.get(
                    "filter_mode"
                )
                == "needs_clarification"
            ):
                fulfillment_status = (
                    "needs_clarification"
                )
            elif not requested_locations:
                fulfillment_status = (
                    "needs_clarification"
                )
            elif not matched_requested_locations:
                fulfillment_status = "blocked"
            elif missing_requested_locations:
                fulfillment_status = "partial"
            else:
                fulfillment_status = "complete"

            prompt_filter_report["fulfillment_status"] = fulfillment_status
            prompt_filter_report["missing_requested_locations"] = missing_requested_locations
            prompt_filter_report["matched_requested_locations"] = matched_requested_locations

            if final_quality_report.get("quality_policy_status") == "unmet":
                prompt_filter_report["block_export"] = True
                prompt_filter_report["downstream_export_enabled"] = False
                prompt_filter_report["filter_mode"] = "location_category_gap_no_export"
                prompt_filter_report["reason"] = "blocked_requested_quality_unmet"
                prompt_filter_report["coverage_status"] = prompt_filter_report.get("coverage_status", "complete")

        if top_cohorts_for_export.empty:
            if prompt_filter_report.get("reason") == "blocked_requested_quality_unmet":
                _coverage_warnings = list(locals().get("coverage_warnings") or [])
                existing_warnings_str = "\n".join(_coverage_warnings)
                for loc in prompt_filter_report.get("missing_requested_locations", []):
                    if loc not in existing_warnings_str:
                         _coverage_warnings.append(f"{loc} was requested, but no export-ready cohort for that location passed the final quality and safety filters.")

                _quality_unmet_swarm_review = {
                    "status": "completed",
                    "overall_review_status": "blocked",
                    "signals": [],
                    "recommendations": [
                        "Wait for stronger/fresher cohorts.",
                        "Use balanced quality when broader coverage is acceptable.",
                        "Review the quality requirement."
                    ],
                    "approval_required": True,
                    "downstream_export_enabled": False,
                }

                _run_dir = Path(str(locals().get("run_dir")))
                if persist_artifacts:
                     _run_dir.mkdir(parents=True, exist_ok=True)
                     _final_summary_path = _run_dir / "final_summary.md"
                     _final_summary_path.write_text("Blocked: requested quality unmet")

                final_summary_dict = {
                    "status": "completed",
                    "approval_status": "blocked_requested_quality_unmet",
                    "downstream_export_enabled": False,
                    "coverage_status": prompt_filter_report.get("coverage_status"),
                    "quality_policy_status": "unmet",
                    "fulfillment_status": "blocked",
                    "quality_candidates_before": final_quality_report.get("quality_candidates_before", 0),
                    "quality_candidates_after": final_quality_report.get("quality_candidates_after", 0),
                    "quality_excluded_count": final_quality_report.get("quality_excluded_count", 0),
                    "run_id": run_id,
                    "prompt": prompt,
                    "run_dir": run_reference,
                    "source_mode": locals().get("source_mode"),
                    "source_rows": locals().get("source_rows"),
                    "source_columns": locals().get("source_columns"),
                    "privacy_cohorts": int(len(locals().get("privacy_cohorts"))) if locals().get("privacy_cohorts") is not None else 0,
                    "prompt_selected_cohorts": int(len(selected_cohorts)) if locals().get("selected_cohorts") is not None else 0,
                    "prompt_filter_report": prompt_filter_report,
                    "coverage_warnings": _coverage_warnings,
                    "final_summary_path": (
                        str(_final_summary_path) if persist_artifacts else final_summary_reference
                    ),
                    "v2_autonomous": locals().get("v2_result") or {},
                    "v2_swarm_review": _quality_unmet_swarm_review,
                    "v2_guided_selection_report": locals().get("v2_guided_selection_report") or {},
                    "safe_export": {
                        "approval_status": "blocked_requested_quality_unmet",
                        "downstream_export_enabled": False,
                        "exported_cohorts": 0,
                        "exported_lookalike_pairs": 0,
                        "outputs": {},
                    },
                    "privacy_guarantees": {
                        "raw_maids_exported": False,
                        "hashed_identifiers_exported": False,
                        "raw_observations_exported": False,
                        "raw_lat_lng_exported": False,
                        "raw_email_exported": False,
                        "raw_phone_exported": False,
                        "individual_user_data_exported": False,
                    }
                }
                if persist_artifacts:
                    final_summary_path_str = str(_run_dir / "final_prompt_summary.json")
                    final_summary_dict["final_summary_path"] = final_summary_path_str
                    with open(final_summary_path_str, "w", encoding="utf-8") as f:
                        json.dump(final_summary_dict, f, indent=2, default=str)

                return final_summary_dict
            else:
                raise ValueError(
                    "Cohort management returned no safe top cohorts for export."
                )

        export_agent = SafeExportAgent()
        export_result = export_agent.run(
            top_cohorts=top_cohorts_for_export,
            lookalikes=lookalikes_for_export,
            output_dir=export_dir,
            run_id=f"{run_id}_safe_export",
            audience_namespace="punk_audience",
            approval_required=approval_required,
            min_management_quality=min_export_quality,
            max_export_cohorts=max_export_cohorts,
            persist_artifacts=persist_artifacts,
        )

        export_result = self._apply_freshness_fail_closed(
            export_result=export_result,
            freshness_guardrail=freshness_guardrail,
        )
        export_result = self._apply_v2_failure_fail_closed(
            export_result=export_result,
            v2_result=v2_result,
        )

        print("SAFE EXPORT STATUS:", export_result["status"])
        print("APPROVAL STATUS:", export_result["approval_status"])
        print("DOWNSTREAM ENABLED:", export_result["downstream_export_enabled"])
        print("EXPORTED COHORTS:", export_result["exported_cohorts"])
        print("EXPORTED LOOKALIKE PAIRS:", export_result["exported_lookalike_pairs"])
        print()

        coverage_warnings = self._build_coverage_warnings(
            prompt=prompt,
            prompt_filter_report=prompt_filter_report,
            export_result=export_result,
        )

        for warning in (
            v2_result.get("coverage_warnings")
            if isinstance(v2_result, dict)
            else []
        ) or []:
            if warning not in coverage_warnings:
                coverage_warnings.append(warning)

        if coverage_warnings:
            print("COVERAGE WARNINGS:")
            for warning in coverage_warnings:
                print("-", warning)
            print()

        final_summary = {
            "status": "completed",
            "run_id": run_id,
            "prompt": prompt,
            "source_mode": source_mode,
            "source_rows": int(len(safe_raw_input)),
            "freshness_status": freshness_guardrail.get("freshness_status"),
            "source_freshness": self._safe_dict(freshness_guardrail),
            "approval_status": export_result.get("approval_status"),
            "downstream_export_enabled": bool(export_result.get("downstream_export_enabled", False)),
            "block_export": bool(export_result.get("block_export", False)),
            "privacy_cohorts": int(len(privacy_cohorts)),
            "prompt_selected_cohorts": int(len(selected_cohorts)),
            "prompt_filter_report": prompt_filter_report,
            "coverage_warnings": coverage_warnings,
            "v2_autonomous": self._safe_dict(v2_result),
            "synthetic": self._safe_dict(synthetic_result),
            "embedding": {
                "status": embedding_result["status"],
                "vector_count": embedding_result["vector_count"],
                "vector_dimension": embedding_result["vector_dimension"],
                "outputs": embedding_result["outputs"],
            },
            "cohort_management": self._safe_dict(cohort_result),
            "safe_export": self._safe_dict(export_result),
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "approval_required": bool(approval_required),
                "approval_status": export_result["approval_status"],
            },
            "certification_evaluation": bool(certification_evaluation),
            "run_dir": run_reference,
            "final_summary_path": final_summary_reference,
        }

        if persist_artifacts:
            final_summary_path.write_text(
                json.dumps(
                    final_summary,
                    indent=2,
                    allow_nan=False,
                )
            )

        try:
            review_agent = AutonomousV2SwarmReviewAgent()

            if persist_artifacts:
                v2_swarm_review = review_agent.review_run(run_dir)
            else:
                v2_swarm_review = review_agent.review_v2(
                    v2_result,
                    run_reference=run_reference,
                )
        except Exception as exc:
            v2_swarm_review = {
                "status": "failed",
                "overall_review_status": "blocked",
                "error": str(exc),
                "approval_required": True,
                "downstream_export_enabled": False,
            }

        final_summary["v2_swarm_review"] = self._safe_dict(
            v2_swarm_review
        )

        if persist_artifacts:
            final_summary_path.write_text(
                json.dumps(
                    final_summary,
                    indent=2,
                    allow_nan=False,
                )
            )

        print("V2 SWARM REVIEW:", v2_swarm_review.get("overall_review_status"))
        print("FINAL SUMMARY:", final_summary_reference)
        print("SAFE EXPORT:", export_result["outputs"]["safe_export_manifest"])

        return final_summary

    def _run_embedding_and_cohort_management(
        self,
        *,
        selected_cohorts: pd.DataFrame,
        embedding_dir: Path,
        cohort_dir: Path,
        run_id: str,
        min_export_quality: float,
        persist_artifacts: bool = True,
    ):
        """
        Use the active Postgres vector backend in production-style environments.

        Local development and tests retain the legacy artifact flow until the
        remaining cohort/export artifacts are migrated in Sprint 4B.
        """
        vector_backend = os.getenv("VECTOR_BACKEND", "local").strip().lower()
        postgres_backends = {
            "postgres",
            "postgres_array",
            "pg_array",
            "pgvector",
        }

        if vector_backend in postgres_backends and persist_artifacts:
            embedding_job_id = f"{run_id}_embedding"
            safe_records = (
                selected_cohorts
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

            raw_embedding_result = embed_records(
                job_id=embedding_job_id,
                records=safe_records,
            )

            vector_store = load_vector_store(embedding_job_id)
            vectors = np.asarray(vector_store.get("vectors"), dtype=float)
            metadata = pd.DataFrame(vector_store.get("metadata") or [])

            if vectors.ndim != 2:
                raise ValueError(
                    "Postgres embedding store returned vectors that are not two-dimensional."
                )

            if metadata.empty:
                raise ValueError(
                    "Postgres embedding store returned empty cohort metadata."
                )

            if len(metadata) != vectors.shape[0]:
                raise ValueError(
                    "Postgres embedding metadata row count does not match vector count."
                )

            vector_dimension = int(vectors.shape[1])
            if vector_dimension != 384:
                raise ValueError(
                    f"Production main embedding dimension must be 384, got {vector_dimension}."
                )

            embedding_result = {
                "module": "Embedding & Feature Store",
                "status": raw_embedding_result.get("status", "completed"),
                "run_id": embedding_job_id,
                "embedding_provider": raw_embedding_result.get("embedding_backend"),
                "model_name": raw_embedding_result.get("model_name"),
                "vector_backend": vector_backend,
                "vector_count": int(vectors.shape[0]),
                "vector_dimension": vector_dimension,
                "vectors_normalized": True,
                "metadata_rows": int(len(metadata)),
                "raw_maids_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "outputs": {
                    "cohort_vectors": raw_embedding_result.get("vectors_path"),
                    "cohort_metadata": raw_embedding_result.get("metadata_path"),
                    "embedding_model": raw_embedding_result.get("model_path"),
                },
            }

            cohort_result = CohortManagementAgent().run(
                metadata=metadata,
                vectors=vectors,
                output_dir=cohort_dir,
                run_id=f"{run_id}_cohort_management",
                min_clusters=2,
                max_clusters=8,
                top_n=25,
                lookalike_top_k=3,
                min_export_quality=min_export_quality,
                persist_artifacts=persist_artifacts,
            )

            return embedding_result, cohort_result

        embedding_agent = EmbeddingFeatureStoreAgent()
        if persist_artifacts:
            embedding_result = embedding_agent.build(
                cohorts=selected_cohorts,
                output_dir=embedding_dir,
                embedding_provider="sklearn_tfidf",
                run_id=f"{run_id}_embedding",
                max_features=384,
            )
            cohort_result = CohortManagementAgent().run_from_artifacts(
                metadata_path=embedding_result["outputs"]["cohort_metadata"],
                vectors_path=embedding_result["outputs"]["cohort_vectors"],
                output_dir=cohort_dir,
                run_id=f"{run_id}_cohort_management",
                min_clusters=2,
                max_clusters=8,
                top_n=25,
                lookalike_top_k=3,
                min_export_quality=min_export_quality,
            )
        else:
            embedding_result, metadata, vectors = (
                embedding_agent.build_in_memory(
                    cohorts=selected_cohorts,
                    embedding_provider="sklearn_tfidf",
                    run_id=f"{run_id}_embedding",
                    max_features=384,
                )
            )
            cohort_result = CohortManagementAgent().run(
                metadata=metadata,
                vectors=vectors,
                output_dir=cohort_dir,
                run_id=f"{run_id}_cohort_management",
                min_clusters=2,
                max_clusters=8,
                top_n=25,
                lookalike_top_k=3,
                min_export_quality=min_export_quality,
                persist_artifacts=False,
            )

        return embedding_result, cohort_result

    def _build_freshness_guardrail(self, v2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts V2 data freshness into a production export guardrail.
        Stale source data may still produce a preview, but must not be approvable/exportable.
        """
        freshness = {}
        if isinstance(v2_result, dict):
            freshness = v2_result.get("data_freshness") or {}

        freshness_status = str(
            freshness.get("freshness_status")
            or freshness.get("status")
            or "unknown"
        ).strip().lower()

        reason = (
            freshness.get("reason")
            or freshness.get("diagnosis")
            or freshness.get("message")
            or ""
        )

        block_export = freshness_status in {"stale", "expired", "outdated"} or "stale" in freshness_status

        return {
            "freshness_status": freshness_status,
            "latest_source_timestamp": freshness.get("latest_source_timestamp"),
            "oldest_source_timestamp": freshness.get("oldest_source_timestamp"),
            "source_age_hours": freshness.get("source_age_hours"),
            "freshness_threshold_hours": freshness.get("freshness_threshold_hours"),
            "source_rows_checked": freshness.get("source_rows_checked"),
            "block_export": bool(block_export),
            "approval_status": "blocked_stale_source" if block_export else "pending_approval",
            "downstream_export_enabled": False,
            "block_export_reason": reason or (
                "Source data is stale. Refresh upstream MAID extraction before approval/export."
                if block_export
                else ""
            ),
        }

    def _apply_v2_failure_fail_closed(
        self,
        *,
        export_result: Dict[str, Any],
        v2_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self._safe_dict(export_result)
        v2 = v2_result or {}

        if str(v2.get("status") or "").lower() != "failed":
            return result

        blocked_status = "blocked_v2_failure"
        result["status"] = "blocked"
        result["approval_status"] = blocked_status
        result["downstream_export_enabled"] = False
        result["block_export"] = True
        result["export_blocked"] = True
        result["exported_cohorts"] = 0
        result["exported_lookalike_pairs"] = 0
        result["block_export_reason"] = (
            "Autonomous Audience Intelligence v2 failed. "
            "Approval and downstream delivery are blocked."
        )
        result["v2_failure"] = {
            "status": "failed",
            "error": str(v2.get("error") or ""),
        }

        package = result.get("package") or {}

        for collection_name in ("cohorts", "lookalikes"):
            for item in package.get(collection_name) or []:
                if not isinstance(item, dict):
                    continue

                item["approval_status"] = blocked_status
                item["export_status"] = blocked_status
                item["allowed_destination"] = "none"
                item["downstream_export_enabled"] = False

        payload = package.get("payload") or {}
        if isinstance(payload, dict):
            payload["approval_status"] = blocked_status
            payload["downstream_export_enabled"] = False
            payload["exported_cohorts"] = 0

        approval_request = package.get("approval_request") or {}
        if isinstance(approval_request, dict):
            approval_request["status"] = blocked_status
            approval_request["approval_status"] = blocked_status

        result["package"] = package
        return result

    def _apply_freshness_fail_closed(
        self,
        *,
        export_result: Dict[str, Any],
        freshness_guardrail: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Applies fail-closed export behavior when source freshness is stale.
        """
        result = self._safe_dict(export_result)

        if not freshness_guardrail.get("block_export"):
            result.setdefault("block_export", False)
            result.setdefault("source_freshness", self._safe_dict(freshness_guardrail))
            return result

        result["approval_status"] = "blocked_stale_source"
        result["downstream_export_enabled"] = False
        result["block_export"] = True
        result["export_blocked_until_source_refresh"] = True
        result["block_export_reason"] = freshness_guardrail.get("block_export_reason")
        result["source_freshness"] = self._safe_dict(
            freshness_guardrail
        )

        package = result.get("package") or {}

        for collection_name in ("cohorts", "lookalikes"):
            for item in package.get(collection_name) or []:
                if not isinstance(item, dict):
                    continue

                item["approval_status"] = "blocked_stale_source"
                item["export_status"] = "blocked_stale_source"
                item["allowed_destination"] = "none"
                item["downstream_export_enabled"] = False

        payload = package.get("payload") or {}
        if isinstance(payload, dict):
            payload["approval_status"] = "blocked_stale_source"
            payload["downstream_export_enabled"] = False

        approval_request = package.get("approval_request") or {}
        if isinstance(approval_request, dict):
            approval_request["status"] = "blocked_stale_source"
            approval_request["approval_status"] = (
                "blocked_stale_source"
            )

        result["package"] = package

        outputs = result.get("outputs") or {}
        manifest_path = outputs.get("safe_export_manifest")
        if (
            manifest_path
            and not str(manifest_path).startswith(
                ("postgres://", "memory://")
            )
        ):
            try:
                Path(manifest_path).write_text(
                    json.dumps(result, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            except Exception:
                # Never fail the whole preview because manifest rewrite failed.
                pass

        return result

    def _select_balanced_location_candidates(
        self,
        candidate,
        requested_locations: list[str],
        sort_cols: list[str],
        max_rows: int,
    ):
        import pandas as pd
        if candidate.empty or max_rows <= 0:
            return candidate.iloc[0:0].copy()

        if not requested_locations or "location_name" not in candidate.columns:
            if sort_cols:
                return candidate.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(max_rows).copy()
            return candidate.head(max_rows).copy()

        if sort_cols:
            candidate = candidate.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()

        location_pools = {loc: [] for loc in requested_locations}
        for idx, row in candidate.iterrows():
            loc_val = str(row.get("location_name", ""))
            for req_loc in requested_locations:
                if location_matches_request(loc_val, req_loc):
                    location_pools[req_loc].append((idx, row))
                    break

        selected_indices = set()
        selected_rows = []

        for req_loc in requested_locations:
            if location_pools[req_loc] and len(selected_rows) < max_rows:
                idx, row = location_pools[req_loc].pop(0)
                selected_indices.add(idx)
                selected_rows.append(row)

        if len(selected_rows) < max_rows:
            for idx, row in candidate.iterrows():
                if len(selected_rows) >= max_rows:
                    break
                if idx not in selected_indices:
                    loc_val = str(row.get("location_name", ""))
                    if any(location_matches_request(loc_val, req_loc) for req_loc in requested_locations):
                        selected_indices.add(idx)
                        selected_rows.append(row)

        res = pd.DataFrame(selected_rows)
        if sort_cols and not res.empty:
             res = res.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        return res

    def _select_cohorts_from_v2_ranked(
        self,
        *,
        privacy_cohorts,
        v2_result,
        prompt_filter_report=None,
        max_rows=10,
    ):
        import os
        from pathlib import Path
        import pandas as pd

        prompt_filter_report = prompt_filter_report or {}
        v2_result = v2_result or {}

        ranked_path = v2_result.get("ranked_matches_path")
        ranked_records = v2_result.get("ranked_matches")
        min_score = float(os.getenv("V2_EXPORT_MIN_MATCH_SCORE", "0.55"))
        strong_category_score = float(os.getenv("V2_EXPORT_MIN_CATEGORY_SCORE", "0.70"))

        allowed_match_types = {
            item.strip()
            for item in os.getenv(
                "V2_EXPORT_ALLOWED_MATCH_TYPES",
                "exact_match,adjacent_category",
            ).split(",")
            if item.strip()
        }

        requested_poi_terms = [
            str(term).strip().lower()
            for term in (
                prompt_filter_report.get("poi_terms_detected")
                or prompt_filter_report.get("poi_terms")
                or []
            )
            if str(term).strip()
        ]

        requested_locations = [
            str(term).strip().lower()
            for term in (
                prompt_filter_report.get("locations_detected")
                or prompt_filter_report.get("locations")
                or []
            )
            if str(term).strip()
        ]

        requested_dayparts = [
            str(term).strip().lower()
            for term in (
                prompt_filter_report.get("dayparts_detected")
                or prompt_filter_report.get("dayparts")
                or []
            )
            if str(term).strip()
        ]

        strict_category_requested = bool(requested_poi_terms)
        allowed_export_pois = self._allowed_export_poi_terms_for_request(requested_poi_terms)

        report = {
            "enabled": False,
            "reason": "no_ranked_matches",
            "ranked_path": ranked_path,
            "min_score": min_score,
            "strong_category_score": strong_category_score,
            "allowed_match_types": sorted(allowed_match_types),
            "strict_category_requested": strict_category_requested,
            "requested_poi_terms": requested_poi_terms,
            "requested_locations": requested_locations,
            "requested_dayparts": requested_dayparts,
            "allowed_export_poi_terms": allowed_export_pois,
            "rows": 0,
            "block_export": False,
            "downstream_export_enabled": True,
        }

        # _active_default_quality_report_v3
        _initial_quality_intent = str(
            (prompt_filter_report or {}).get(
                "quality_intent"
            )
            or "balanced"
        ).strip().lower()

        if _initial_quality_intent not in {
            "high",
            "balanced",
            "broad",
        }:
            _initial_quality_intent = "balanced"

        report.setdefault(
            "quality_intent",
            _initial_quality_intent,
        )
        report.setdefault(
            "quality_policy_report",
            {
                "quality_intent": (
                    _initial_quality_intent
                ),
                "quality_policy_status": (
                    "pending_final_quality_evaluation"
                ),
                "quality_candidates_before": 0,
                "quality_candidates_after": 0,
                "quality_excluded_count": 0,
            },
        )

        quality_intent = str(
            prompt_filter_report.get(
                "quality_intent"
            )
            or "balanced"
        ).strip().lower()

        if quality_intent not in {
            "high",
            "balanced",
            "broad",
        }:
            quality_intent = "balanced"

        # Always preserve quality intent, including early
        # clarification/blocking returns.
        report["quality_intent"] = quality_intent

        eligibility_value = (
            prompt_filter_report.get(
                "eligible_for_audience_selection"
            )
        )

        eligibility_blocked = bool(
            eligibility_value is False
            or prompt_filter_report.get(
                "filter_mode"
            )
            == "needs_clarification"
        )

        missing_required_constraints = list(
            prompt_filter_report.get(
                "missing_required_constraints"
            )
            or []
        )

        # Do not infer eligibility from missing low-level
        # selector fields. Real orchestrated flows set the
        # eligibility field explicitly during semantic merge.
        if eligibility_blocked:
            report.update(
                {
                    "enabled": True,
                    "reason": (
                        "audience_request_needs_clarification"
                    ),
                    "filter_mode": (
                        "needs_clarification"
                    ),
                    "fulfillment_status": (
                        "needs_clarification"
                    ),
                    "missing_required_constraints": (
                        missing_required_constraints
                    ),
                    "rows": 0,
                    "block_export": True,
                    "export_blocked": True,
                    "downstream_export_enabled": False,
                    "quality_intent": quality_intent,
                }
            )
            return pd.DataFrame(), report

        if requested_poi_terms and not allowed_export_pois:
            report.update(
                {
                    "enabled": True,
                    "reason": (
                        "requested_category_has_no_safe_mapping"
                    ),
                    "filter_mode": (
                        "location_category_gap_no_export"
                    ),
                    "fulfillment_status": "blocked",
                    "rows": 0,
                    "block_export": True,
                    "export_blocked": True,
                    "downstream_export_enabled": False,
                    "coverage_warnings": [
                        (
                            "The requested category has no "
                            "privacy-safe export mapping."
                        )
                    ],
                }
            )
            return pd.DataFrame(), report

        if isinstance(ranked_records, list):
            ranked = pd.DataFrame(ranked_records)
        else:
            if not ranked_path:
                return pd.DataFrame(), report

            if str(ranked_path).startswith(
                ("postgres://", "memory://")
            ):
                report["reason"] = "ranked_records_missing"
                return pd.DataFrame(), report

            ranked_file = Path(ranked_path)
            if not ranked_file.exists():
                report["reason"] = "ranked_matches_path_missing"
                return pd.DataFrame(), report

            ranked = pd.read_csv(ranked_file)
        if ranked.empty:
            report["reason"] = "empty_ranked_matches"
            return pd.DataFrame(), report

        candidate = ranked.copy()

        if "privacy_status" in candidate.columns:
            candidate = candidate[
                candidate["privacy_status"].fillna("").astype(str).str.lower().isin(["passed", "pass", "safe", ""])
            ].copy()

        if "final_match_score" in candidate.columns:
            candidate["final_match_score"] = pd.to_numeric(candidate["final_match_score"], errors="coerce").fillna(0)
            candidate = candidate[candidate["final_match_score"] >= min_score].copy()

        if "match_type" in candidate.columns:
            candidate = candidate[candidate["match_type"].astype(str).isin(allowed_match_types)].copy()

        if requested_locations and "location_name" in candidate.columns:
            candidate = candidate[
                candidate["location_name"].apply(
                    lambda loc: any(location_matches_request(loc, req) for req in requested_locations)
                )
            ].copy()

        if allowed_export_pois and "primary_poi_type" in candidate.columns:
            candidate = candidate[
                candidate["primary_poi_type"].apply(
                    lambda poi: self._vijay_poi_matches_allowed(poi, allowed_export_pois)
                )
            ].copy()

        if "category_match_score" in candidate.columns:
            candidate["category_match_score"] = pd.to_numeric(candidate["category_match_score"], errors="coerce").fillna(0)
            if strict_category_requested:
                candidate = candidate[candidate["category_match_score"] >= strong_category_score].copy()
            else:
                candidate = candidate[candidate["category_match_score"] >= 0.50].copy()

        if requested_dayparts and "created_day_part" in candidate.columns:
            allowed_dayparts = {self._vijay_norm_text(x) for x in requested_dayparts}
            candidate = candidate[
                candidate["created_day_part"].apply(lambda x: self._vijay_norm_text(x) in allowed_dayparts)
            ].copy()

        if candidate.empty:
            report["enabled"] = True
            report["reason"] = "strict_location_category_requested_but_no_safe_exact_match"
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["downstream_export_enabled"] = False
            report["fulfillment_status"] = "blocked"
            return pd.DataFrame(), report

        # Stage 1 — PRE-MANAGEMENT
        # Do not reject candidates for high quality here. Just resolve intent and report.
        quality_intent = (prompt_filter_report or {}).get("quality_intent") or "balanced"

        from app.services.audience_quality_policy_service import AudienceQualityPolicyService
        quality_policy_svc = AudienceQualityPolicyService()
        candidate = quality_policy_svc.normalize_quality_scores(candidate)

        quality_report = {
            "quality_intent": quality_intent,
            "quality_policy_status": "pending_final_quality_evaluation",
            "quality_candidates_before": len(candidate),
            "quality_candidates_after": len(candidate),
        }

        report["quality_policy_report"] = quality_report
        report["quality_intent"] = quality_intent

        missing_requested_locations = []
        matched_requested_locations = []
        if requested_locations and "location_name" in candidate.columns:
            for requested_location in requested_locations:
                has_location = any(
                    location_matches_request(cohort_location, requested_location)
                    for cohort_location in candidate["location_name"].dropna().tolist()
                )
                if not has_location:
                    missing_requested_locations.append(str(requested_location))
                else:
                    matched_requested_locations.append(str(requested_location))

        if not requested_locations:
            coverage_status = "complete"
        elif not matched_requested_locations:
            coverage_status = "blocked"
        elif missing_requested_locations:
            coverage_status = "partial"
        else:
            coverage_status = "complete"

        report["coverage_status"] = coverage_status
        report["fulfillment_status"] = coverage_status
        report["missing_requested_locations"] = missing_requested_locations
        report["matched_requested_locations"] = matched_requested_locations

        if coverage_status == "blocked" and requested_locations:
            report["enabled"] = True
            report["reason"] = "strict_location_category_requested_but_no_safe_exact_match"
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["downstream_export_enabled"] = False

            existing_warnings_str = "\n".join(report.get("coverage_warnings", []))
            for location in missing_requested_locations:
                if location not in existing_warnings_str:
                    report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                        f"{location} had no exact privacy-safe candidate for the requested category/daypart."
                    ]
            return pd.DataFrame(), report

        if missing_requested_locations:
            existing_warnings_str = "\n".join(report.get("coverage_warnings", []))
            for location in missing_requested_locations:
                if location not in existing_warnings_str:
                    report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                        f"{location} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                    ]

        if quality_intent == "broad":
            sort_cols = [
                col for col in ["total_maid_volume", "_effective_quality_score", "final_match_score"]
                if col in candidate.columns
            ]
        elif quality_intent == "high":
            sort_cols = [
                col for col in ["_effective_quality_score", "final_match_score", "total_maid_volume"]
                if col in candidate.columns
            ]
        else:
            sort_cols = [
                col for col in ["final_match_score", "_effective_quality_score", "total_maid_volume"]
                if col in candidate.columns
            ]

        candidate = self._select_balanced_location_candidates(
            candidate=candidate,
            requested_locations=requested_locations,
            sort_cols=sort_cols,
            max_rows=max_rows,
        ).copy()

        selected = candidate

        if privacy_cohorts is not None and hasattr(privacy_cohorts, "columns"):
            if "cohort_id" in candidate.columns and "cohort_id" in privacy_cohorts.columns:
                ids = candidate["cohort_id"].dropna().tolist()
                selected = privacy_cohorts[privacy_cohorts["cohort_id"].isin(ids)].copy()

                if not selected.empty:
                    order = {value: idx for idx, value in enumerate(ids)}
                    selected["_vijay_order"] = selected["cohort_id"].map(order).fillna(999999)
                    selected = selected.sort_values("_vijay_order").drop(columns=["_vijay_order"]).head(max_rows).copy()

        report["enabled"] = True
        report["reason"] = "v2_ranked_safe_selection"
        report["rows"] = int(len(selected))
        report["filter_mode"] = "location+poi+daypart" if requested_locations and allowed_export_pois and requested_dayparts else "v2_ranked_safe"
        report["block_export"] = False
        report["downstream_export_enabled"] = True

        return selected, report

    def _build_no_match_swarm_review(
        self,
        *,
        coverage_warnings: list[str],
    ) -> Dict[str, Any]:
        warnings = [
            str(item)
            for item in (coverage_warnings or [])
            if str(item).strip()
        ]

        return {
            "status": "completed",
            "overall_review_status": "blocked",
            "review_reason": "no_safe_exact_match",
            "coverage_warning_count": len(warnings),
            "coverage_warnings": warnings,
            "data_gap_count": 0,
            "approval_required": True,
            "downstream_export_enabled": False,
            "signals": [
                {
                    "name": "exact_safe_match",
                    "status": "blocked",
                    "reason": (
                        "No exact privacy-safe cohort matched the "
                        "requested location, category, and daypart."
                    ),
                }
            ],
            "recommendations": [
                (
                    "Clarify or correct the requested location, or wait "
                    "until matching privacy-safe source coverage is available."
                )
            ],
        }

    def _run_privacy_layer(
        self,
        *,
        safe_raw_input: pd.DataFrame,
        output_dir: Path,
        run_id: str,
        k_min: int,
        epsilon: float,
        persist_artifacts: bool,
        record_privacy_budget: bool = True,
    ) -> Dict[str, Any]:
        if persist_artifacts:
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        agent = PrivacyLayerAgent()

        return self._call_agent_method(
            agent=agent,
            method_names=["run", "process", "build", "execute", "transform", "anonymize"],
            kwargs={
                "raw_observations": safe_raw_input,
                "observations": safe_raw_input,
                "input_df": safe_raw_input,
                "df": safe_raw_input,
                "data": safe_raw_input,
                "cohorts": safe_raw_input,
                "output_dir": output_dir,
                "run_id": run_id,
                "identifier_column": "session_id",
                "count_column": "maid_count",
                "timestamp_column": "created_at",
                "location_column": "location_name",
                "poi_column": "primary_poi_type",
                "k_min": k_min,
                "epsilon": epsilon,
                "persist_artifacts": persist_artifacts,
                "record_privacy_budget": record_privacy_budget,
            },
        )

    def _call_agent_method(self, *, agent: Any, method_names: list[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        errors = []

        for method_name in method_names:
            if not hasattr(agent, method_name):
                continue

            method = getattr(agent, method_name)
            signature = inspect.signature(method)

            accepted_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }

            try:
                result = method(**accepted_kwargs)
                if isinstance(result, dict):
                    return result
                return {"status": "completed", "result": result}
            except TypeError as exc:
                errors.append(f"{method_name}: {exc}")

        public_methods = [
            name for name in dir(agent)
            if not name.startswith("_") and callable(getattr(agent, name))
        ]

        raise RuntimeError(
            f"Could not call agent {agent.__class__.__name__}. "
            f"Tried methods={method_names}. Public methods={public_methods}. Errors={errors}"
        )

    def _resolve_privacy_feature_path(self, *, privacy_result: Dict[str, Any], privacy_dir: Path) -> Path:
        candidates = []

        outputs = privacy_result.get("outputs")
        if isinstance(outputs, dict):
            for key in ["clean_feature_table", "feature_table", "privacy_feature", "safe_feature_table"]:
                if key in outputs:
                    candidates.append(Path(outputs[key]))

        for key in ["clean_feature_table", "feature_path", "output_path", "safe_feature_table"]:
            if key in privacy_result:
                candidates.append(Path(privacy_result[key]))

        candidates.extend(
            [
                privacy_dir / "clean_feature_table.csv",
                privacy_dir / "safe_feature_table.csv",
                privacy_dir / "privacy_feature_table.csv",
            ]
        )

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find PrivacyLayerAgent clean feature table in {privacy_dir}. "
            f"Privacy result keys: {list(privacy_result.keys())}"
        )

    def _load_safe_rows_from_postgres(self, *, prompt: str, limit: int) -> pd.DataFrame:
        self._load_dotenv()

        db_url = (
            os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_DATABASE_URL")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DB_URL")
        )

        if not db_url:
            raise RuntimeError(
                "No Postgres DB URL found in env. Expected DATABASE_URL or POSTGRES_URL. "
                "Use --source safe_artifact --safe-cohort-path <clean_feature_table.csv> to run from existing safe data."
            )

        from sqlalchemy import create_engine

        engine = create_engine(db_url)

        query = """
        SELECT
            session_id,
            created_at,
            maid_count,
            pois,
            center,
            lookback_days
        FROM public.maid_extractions
        WHERE maid_count IS NOT NULL
        ORDER BY created_at DESC
        LIMIT %(limit)s
        """

        raw = pd.read_sql_query(query, engine, params={"limit": int(limit)})

        safe = pd.DataFrame(
            {
                "session_id": raw["session_id"].astype(str),
                "created_at": raw["created_at"],
                "maid_count": pd.to_numeric(raw["maid_count"], errors="coerce").fillna(0).astype(int),
                "location_name": raw["center"].apply(lambda value: self._derive_location_name(value, prompt)),
                "primary_poi_type": raw["pois"].apply(lambda value: self._derive_poi_type(value, prompt)),
            }
        )

        safe = safe[safe["maid_count"] > 0].reset_index(drop=True)

        return safe

    def _load_dotenv(self) -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return

        for line in env_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')

            if key and key not in os.environ:
                os.environ[key] = value

    def _derive_location_name(self, value: Any, prompt: str) -> str:
        obj = self._parse_jsonish(value)

        if isinstance(obj, dict):
            for key in [
                "location_name",
                "city",
                "place_name",
                "name",
                "label",
                "area",
                "region",
                "address",
                "formatted_address",
            ]:
                candidate = obj.get(key)
                if self._safe_text(candidate):
                    return self._clean_text(candidate)

        return self._location_hint_from_prompt(prompt) or "selected location"

    def _derive_poi_type(self, value: Any, prompt: str) -> str:
        obj = self._parse_jsonish(value)

        prompt_poi = self._poi_hint_from_prompt(prompt)

        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict):
                candidate = self._poi_from_dict(first)
                if candidate:
                    return candidate

        if isinstance(obj, dict):
            candidate = self._poi_from_dict(obj)
            if candidate:
                return candidate

        return prompt_poi or "point_of_interest"

    def _poi_from_dict(self, obj: dict) -> Optional[str]:
        for key in ["primary_poi_type", "poi_type", "place_type", "category", "type"]:
            value = obj.get(key)
            if self._safe_text(value):
                return self._clean_poi(value)

        types = obj.get("types")
        if isinstance(types, list):
            generic = {"point_of_interest", "establishment"}
            for item in types:
                cleaned = self._clean_poi(item)
                if cleaned and cleaned not in generic:
                    return cleaned
            if types:
                return self._clean_poi(types[0])

        return None

    def _parse_jsonish(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value

        if pd.isna(value):
            return None

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return json.loads(value)
            except Exception:
                return None

        return None

    def _safe_text(self, value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return False
        return True

    def _clean_text(self, value: Any) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9\s,\-]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:120] or "selected location"

    def _clean_poi(self, value: Any) -> str:
        text = str(value).strip().lower()
        text = text.replace(" ", "_").replace("-", "_")
        text = re.sub(r"[^a-z0-9_]+", "", text)
        return text[:80] or "point_of_interest"

    def _location_hint_from_prompt(self, prompt: str) -> Optional[str]:
        lower = prompt.lower()

        known = [
            "san francisco",
            "montreal",
            "montréal",
            "new york",
            "los angeles",
            "toronto",
            "quebec",
            "hyderabad",
            "bangalore",
            "chicago",
            "dhaka",
        ]

        found = [item for item in known if item in lower]
        if found:
            return found[0]

        return None

    def _poi_hint_from_prompt(self, prompt: str) -> Optional[str]:
        lower = prompt.lower().replace("-", " ")

        mapping = {
            "restaurant": "restaurant",
            "quick service food": "restaurant",
            "quick bites": "restaurant",
            "casual dining": "restaurant",
            "fast meals": "restaurant",
            "food": "food",
            "cafe": "cafe",
            "coffee": "cafe",
            "espresso": "cafe",
            "gym": "gym",
            "fitness": "gym",
            "workout": "gym",
            "yoga": "gym",
            "health": "health",
            "barber": "barber_shop",
            "tattoo": "body_art_service",
            "body ink": "body_art_service",
            "piercing": "body_art_service",
            "office": "coworking_space",
            "coworking": "coworking_space",
            "business center": "coworking_space",
            "business centers": "coworking_space",
            "business place": "coworking_space",
            "business places": "coworking_space",
            "professional": "coworking_space",
            "professionals": "coworking_space",
            "store": "store",
            "retail": "store",
            "shopping": "store",
            "clinic": "healthcare",
            "pharmacy": "healthcare",
            "healthcare": "healthcare",
            "students": "school",
            "campus": "school",
            "nightlife": "bar",
            "self care": "salon",
            "self-care": "salon",
            "wellness": "salon",
            "grooming": "salon",
        }

        for key, value in mapping.items():
            if key in lower:
                return value

        return None

    def _extract_location_terms(self, prompt: str, df: pd.DataFrame) -> list[str]:
        lower = prompt.lower()
        found = []

        if "location_name" in df.columns:
            for location in df["location_name"].dropna().astype(str).str.lower().unique():
                location = location.strip()
                if len(location) < 3:
                    continue

                if location in lower:
                    found.append(location)
                    continue

                parts = [p for p in re.split(r"[\s,]+", location) if len(p) >= 4]
                if any(part in lower for part in parts):
                    found.append(location)

        manual = [
            "montreal downtown",
            "montreal qc",
            "san francisco",
            "montreal",
            "new york",
            "los angeles",
            "toronto",
            "quebec",
            "vancouver",
            "chicago",
        ]
        for item in manual:
            if item in lower and item not in found:
                found.append(item)

        return list(dict.fromkeys(found))

    def _extract_poi_terms(self, prompt: str) -> list[str]:
        lower = prompt.lower().replace("-", " ")
        terms = []

        mapping = {
            "restaurant": ["restaurant", "food", "fast_food", "quick_service_food"],
            "food": ["food", "restaurant", "fast_food"],
            "quick service": ["restaurant", "fast_food", "food"],
            "quick bites": ["restaurant", "fast_food", "food"],
            "fast meals": ["restaurant", "fast_food", "food"],
            "casual dining": ["restaurant", "food"],
            "cafe": ["cafe", "coffee"],
            "coffee": ["cafe", "coffee"],
            "espresso": ["cafe", "coffee"],
            "gym": ["gym", "fitness", "health_club"],
            "fitness": ["gym", "fitness", "health_club"],
            "workout": ["gym", "fitness", "health_club"],
            "yoga": ["gym", "fitness", "health_club"],
            "health": ["health"],
            "barber": ["barber_shop", "barber"],
            "tattoo": ["body_art_service", "tattoo"],
            "body ink": ["body_art_service", "tattoo"],
            "piercing": ["body_art_service", "tattoo"],
            "office": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "coworking": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "business": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "professional": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "store": ["store", "retail", "shopping_mall"],
            "retail": ["store", "retail", "shopping_mall"],
            "shopping": ["store", "retail", "shopping_mall"],
            "clinic": ["clinic", "healthcare", "hospital", "pharmacy"],
            "pharmacy": ["pharmacy", "healthcare", "clinic"],
            "healthcare": ["healthcare", "clinic", "hospital", "pharmacy"],
            "student": ["school", "college", "university", "campus"],
            "campus": ["school", "college", "university", "campus"],
            "nightlife": ["bar", "pub", "club", "lounge"],
            "wellness": ["salon", "spa", "beauty"],
            "grooming": ["salon", "spa", "beauty", "barber_shop"],
            "self care": ["salon", "spa", "beauty"],
            "self-care": ["salon", "spa", "beauty"],
        }

        for key, values in mapping.items():
            if key in lower:
                terms.extend(values)

        return list(dict.fromkeys(terms))

    def _build_coverage_warnings(
        self,
        *,
        prompt: str,
        prompt_filter_report: Dict[str, Any],
        export_result: Dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []

        package = export_result.get("package") or {}
        cohort_records = package.get("cohorts") or []

        if isinstance(cohort_records, list) and cohort_records:
            exported = pd.DataFrame(cohort_records)
        else:
            outputs = export_result.get("outputs") or {}
            cohorts_path = outputs.get("safe_export_cohorts")

            if (
                not cohorts_path
                or str(cohorts_path).startswith(
                    ("postgres://", "memory://")
                )
                or not Path(cohorts_path).exists()
            ):
                return [
                    "No safe export cohorts were available "
                    "for coverage verification."
                ]

            try:
                exported = pd.read_csv(cohorts_path)
            except Exception as exc:
                return [
                    "Could not read safe export cohorts for "
                    f"coverage verification: {exc}"
                ]

        prompt_lower = prompt.lower().replace("café", "cafe").replace("cafés", "cafes")
        filter_mode = str(prompt_filter_report.get("filter_mode") or "")

        def clean_values(column: str) -> list[str]:
            if column not in exported.columns:
                return []
            return (
                exported[column]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
                .tolist()
            )

        def token_match(needle: str, haystack: str) -> bool:
            needle = str(needle or "").lower().replace(" ", "_").strip()
            haystack = str(haystack or "").lower().replace(" ", "_").strip()

            if not needle or not haystack:
                return False

            if needle in haystack or haystack in needle:
                return True

            needle_tokens = {t for t in re.split(r"[^a-z0-9]+", needle) if t}
            haystack_tokens = {t for t in re.split(r"[^a-z0-9]+", haystack) if t}

            if not needle_tokens or not haystack_tokens:
                return False

            return bool(needle_tokens & haystack_tokens)

        exported_locations = clean_values("location_name")
        exported_pois = clean_values("primary_poi_type")
        exported_dayparts = clean_values("created_day_part")

        requested_locations = []
        for loc in prompt_filter_report.get("locations_detected", []) or []:
            loc_clean = str(loc).lower().strip()
            if loc_clean and loc_clean in prompt_lower:
                requested_locations.append(loc_clean)
        requested_locations = list(dict.fromkeys(requested_locations))

        requested_poi_terms = [
            str(term).lower().strip()
            for term in (prompt_filter_report.get("poi_terms_detected", []) or [])
            if str(term).strip()
        ]
        requested_poi_terms = list(dict.fromkeys(requested_poi_terms))

        requested_dayparts = [
            str(daypart).lower().strip()
            for daypart in (prompt_filter_report.get("dayparts_detected", []) or [])
            if str(daypart).strip()
        ]
        requested_dayparts = list(dict.fromkeys(requested_dayparts))

        missing_from_v2 = set(prompt_filter_report.get("missing_requested_locations", []))

        for requested in requested_locations:
            covered = any(
                requested in exported_location or exported_location in requested
                for exported_location in exported_locations
            )

            if not covered:
                if requested not in missing_from_v2:
                    warnings.append(
                        f"{requested} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                    )

        if requested_poi_terms:
            exact_poi_covered = any(
                token_match(term, exported_poi)
                for term in requested_poi_terms
                for exported_poi in exported_pois
            )

            if "poi" not in filter_mode:
                warnings.append(
                    "The requested business/category terms were detected, but exact category coverage was not strong enough. "
                    "The exported cohorts are fallback/adjacent candidates and must be reviewed before approval."
                )
            elif not exact_poi_covered:
                warnings.append(
                    "The requested business/category terms were detected, but no export-ready cohort directly matched those terms."
                )

        if requested_dayparts:
            daypart_covered = any(
                requested == exported_daypart
                for requested in requested_dayparts
                for exported_daypart in exported_dayparts
            )

            if not daypart_covered:
                warnings.append(
                    "The requested time/daypart was detected, but no export-ready cohort matched that time window."
                )

        if not requested_locations and prompt_filter_report.get("locations_detected"):
            warnings.append(
                "Location terms were inferred from available cohorts, but no exact location phrase was directly confirmed from the prompt."
            )

        return warnings

    def _build_run_id(self, prompt: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:50]
        suffix = uuid.uuid4().hex[:6]
        return f"prompt_{timestamp}_{slug}_{suffix}"

    def _is_terminal_prompt_safety_report(
        self,
        prompt_filter_report: Dict[str, Any],
    ) -> bool:
        return str(
            (prompt_filter_report or {}).get("filter_mode")
            or ""
        ) in {
            "privacy_identifier_request_blocked",
            "export_action_requires_existing_audience",
            "approval_bypass_attempt_blocked",
        }

    def _build_preflight_prompt_filter_report(
        self,
        *,
        reason_code: str,
        explanation: str,
    ) -> Dict[str, Any]:
        if reason_code == "blocked_privacy_identifier_request":
            filter_mode = "privacy_identifier_request_blocked"
            default_explanation = (
                "Raw identifiers and individual-level user data cannot be "
                "retrieved or exported. Only privacy-safe aggregated "
                "cohorts are allowed."
            )
        elif reason_code == "blocked_approval_bypass_attempt":
            filter_mode = "approval_bypass_attempt_blocked"
            default_explanation = (
                "Safety, freshness, governance, and manual approval "
                "controls cannot be bypassed. No audience data was read, "
                "ranked, activated, or exported."
            )
        else:
            filter_mode = (
                "export_action_requires_existing_audience"
            )
            default_explanation = (
                "An export action requires an explicit existing audience "
                "or run and an authoritative approval record. No audience "
                "data was read, ranked, activated, or exported."
            )
        message = explanation or default_explanation
        return {
            "enabled": True,
            "filter_mode": filter_mode,
            "block_export": True,
            "export_blocked": True,
            "downstream_export_enabled": False,
            "reason": reason_code,
            "locations_detected": [],
            "poi_terms_detected": [],
            "dayparts_detected": [],
            "coverage_warnings": [message],
            "selected_count": 0,
            "preflight_decision": True,
            "source_read_performed": False,
            "model_evaluation_performed": False,
        }

    def _build_terminal_prompt_safety_result(
        self,
        *,
        prompt: str,
        run_id: str,
        run_reference: str,
        final_summary_reference: str,
        final_summary_path: Path,
        persist_artifacts: bool,
        source_mode: str,
        source_rows: int | None,
        source_columns: list[str],
        privacy_cohorts: pd.DataFrame,
        prompt_filter_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        filter_mode = str(
            (prompt_filter_report or {}).get("filter_mode")
            or ""
        )
        report = dict(prompt_filter_report or {})
        report.update(
            {
                "audience_request_detected": False,
                "eligible_for_audience_selection": False,
                "missing_required_constraints": [],
                "fulfillment_status": "blocked",
                "quality_intent": "not_requested",
                "block_export": True,
                "export_blocked": True,
                "downstream_export_enabled": False,
                "selected_count": 0,
            }
        )

        if filter_mode == "privacy_identifier_request_blocked":
            approval_status = "blocked_privacy_identifier_request"
            status_message = (
                "Raw MAIDs, device IDs, and individual-level "
                "user data cannot be provided or exported. Only "
                "privacy-safe aggregated cohorts are allowed."
            )
            review_reason = "privacy_identifier_request_blocked"
        elif filter_mode == "approval_bypass_attempt_blocked":
            approval_status = "blocked_approval_bypass_attempt"
            status_message = (
                "Safety, freshness, governance, and manual approval "
                "controls cannot be bypassed. No audience data was read, "
                "ranked, activated, or exported."
            )
            review_reason = "approval_bypass_attempt_blocked"
        else:
            approval_status = (
                "blocked_export_action_requires_existing_audience"
            )
            status_message = (
                "Export-action-only requests require an existing "
                "selected audience/run and manual approval. No new "
                "audience was generated."
            )
            review_reason = (
                "export_action_requires_existing_audience"
            )

        coverage_warnings = list(
            report.get("coverage_warnings")
            or []
        )
        v2_result = {
            "status": "skipped",
            "pipeline_version": "v2_autonomous_preview",
            "reason": review_reason,
            "ranked_match_count": 0,
            "mutation": {
                "suggestion_count": 0,
                "suggestions": [],
            },
            "approval_status": approval_status,
            "block_export": True,
            "approval_required": True,
            "downstream_export_enabled": False,
        }
        v2_swarm_review = {
            "status": "completed",
            "overall_review_status": "blocked",
            "review_reason": review_reason,
            "coverage_warning_count": len(coverage_warnings),
            "coverage_warnings": coverage_warnings,
            "data_gap_count": 0,
            "approval_required": True,
            "downstream_export_enabled": False,
            "signals": [],
            "recommendations": [],
        }
        safe_export = {
            "approval_status": approval_status,
            "downstream_export_enabled": False,
            "block_export": True,
            "block_export_reason": status_message,
            "exported_cohorts": 0,
            "exported_lookalike_pairs": 0,
            "outputs": {},
        }

        final_summary = {
            "status": "skipped",
            "run_id": run_id,
            "prompt": prompt,
            "run_dir": run_reference,
            "source_mode": source_mode,
            "source_rows": (
                int(source_rows)
                if source_rows is not None
                else None
            ),
            "source_columns": source_columns,
            "approval_status": approval_status,
            "downstream_export_enabled": False,
            "block_export": True,
            "block_export_reason": status_message,
            "policy_reason_code": approval_status,
            "terminal_policy_decision": True,
            "freshness_status": "not_evaluated",
            "pipeline_stages": self._terminal_pipeline_stages(
                filter_mode=filter_mode,
            ),
            "privacy_cohorts": int(len(privacy_cohorts)),
            "prompt_selected_cohorts": 0,
            "prompt_filter_report": report,
            "coverage_warnings": coverage_warnings,
            "final_summary_path": final_summary_reference,
            "v2_autonomous": v2_result,
            "v2_swarm_review": v2_swarm_review,
            "v2_guided_selection_report": {
                "enabled": False,
                "reason": review_reason,
                "rows": 0,
                "block_export": True,
                "downstream_export_enabled": False,
            },
            "safe_export": safe_export,
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "approval_required": True,
                "approval_status": approval_status,
            },
        }

        if persist_artifacts:
            final_summary_path.write_text(
                json.dumps(
                    final_summary,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

        print("TERMINAL SAFETY DECISION:", review_reason)
        print("APPROVAL STATUS:", approval_status)
        print("DOWNSTREAM ENABLED:", False)
        print()

        return final_summary

    def _terminal_pipeline_stages(
        self,
        *,
        filter_mode: str,
    ) -> list[Dict[str, str]]:
        if filter_mode == "privacy_identifier_request_blocked":
            boundary_detail = "Prohibited identifier request blocked"
            governance_detail = "Privacy policy enforced"
        elif filter_mode == "approval_bypass_attempt_blocked":
            boundary_detail = "Policy-bypass attempt blocked"
            governance_detail = "Approval controls enforced"
        else:
            boundary_detail = "Action request blocked before data access"
            governance_detail = "Existing approved audience required"
        return [
            {
                "id": "data_privacy",
                "status": "blocked",
                "detail": boundary_detail,
            },
            {
                "id": "semantic_retrieval",
                "status": "skipped",
                "detail": "Not evaluated",
            },
            {
                "id": "cohort_intelligence",
                "status": "skipped",
                "detail": "Not evaluated",
            },
            {
                "id": "evolution_review",
                "status": "skipped",
                "detail": "Not evaluated",
            },
            {
                "id": "governance",
                "status": "blocked",
                "detail": governance_detail,
            },
        ]

    def _safe_dict(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._safe_dict(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._safe_dict(v) for v in value]

        if isinstance(value, tuple):
            return [self._safe_dict(v) for v in value]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        return value


    # === Vijay local recovery guardrails: START ===
    # Local recovery block for audience-intelligence-agents.
    # Keep these methods inside AudienceIntelligenceOrchestratorAgent.
    # Do not paste shell commands into this Python file.

    def _vijay_norm_text(self, value):
        import re

        if value is None:
            return ""
        value = str(value).lower().replace("&", " and ")
        value = value.replace("_", " ")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _vijay_snake(self, value):
        return self._vijay_norm_text(value).replace(" ", "_")

    def _vijay_dedupe(self, values):
        out = []
        seen = set()
        for value in values or []:
            if value is None:
                continue
            raw = str(value).strip().lower()
            if not raw:
                continue
            if raw not in seen:
                out.append(raw)
                seen.add(raw)
        return out

    def _vijay_has_any(self, text, terms):
        norm = self._vijay_norm_text(text)
        return any(self._vijay_norm_text(term) in norm for term in terms)

    def _vijay_is_coffee_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "coffee",
                "caffeine",
                "espresso",
                "latte",
                "cappuccino",
                "cafe",
                "cafes",
                "café",
                "coffee shop",
                "coffee_shop",
                "snacks after work",
                "caffeine break",
            ],
        )

    def _vijay_is_restaurant_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "restaurant",
                "restaurants",
                "shawarma",
                "food",
                "dining",
                "eatery",
                "lunch",
                "dinner",
                "takeaway",
            ],
        )

    def _vijay_is_retail_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "retail",
                "shopping",
                "shopping mall",
                "mall",
                "fashion",
                "apparel",
                "clothing",
                "department store",
                "shoppers",
            ],
        )

    def _vijay_is_gym_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "gym",
                "gyms",
                "fitness",
                "fitness center",
                "fitness centre",
                "fitness centers",
                "fitness centres",
                "health club",
                "health clubs",
                "workout",
                "workouts",
                "premium gym",
                "yoga studio",
                "fitness studio",
            ],
        )

    def _vijay_is_bakery_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "bakery",
                "bakeries",
                "baked goods",
                "pastry",
                "pastries",
                "bread",
                "dessert shop",
                "desserts",
            ],
        )

    def _is_raw_identifier_request(self, prompt):
        """
        Detect dangerous requests for raw identifiers.

        Safe negative instructions such as "do not export raw MAIDs",
        "block device IDs", or "remove hashes" must NOT be treated as
        a request to receive raw identifiers.
        """
        import re

        normalized = str(prompt or "").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()

        identifier_terms = [
            "raw maid",
            "raw maids",
            "maid id",
            "maid ids",
            "device id",
            "device ids",
            "mobile advertising id",
            "mobile advertising ids",
            "hashed identifier",
            "hashed identifiers",
            "hashes",
            "raw hash",
            "raw hashes",
            "exact lat",
            "exact lon",
            "exact lng",
            "lat/lon",
            "lat lng",
            "latitude longitude",
            "individual user",
            "individual users",
            "user-level data",
            "individual-level data",
        ]

        terms = "|".join(re.escape(term) for term in identifier_terms)

        safe_negative_patterns = [
            rf"\bdo not\b.{{0,80}}({terms})",
            rf"\bdon't\b.{{0,80}}({terms})",
            rf"\bnever\b.{{0,80}}({terms})",
            rf"\bblock\b.{{0,80}}({terms})",
            rf"\bremove\b.{{0,80}}({terms})",
            rf"\bexclude\b.{{0,80}}({terms})",
            rf"\bwithout\b.{{0,80}}({terms})",
            rf"\bno\b.{{0,80}}({terms})",
        ]

        if any(re.search(item, normalized) for item in safe_negative_patterns):
            return False

        dangerous_action_patterns = [
            rf"\b(give|show|list|display|return|export|download|send|provide|share|reveal|extract)\b.{{0,80}}({terms})",
            rf"({terms}).{{0,80}}\b(give|show|list|display|return|export|download|send|provide|share|reveal|extract)\b",
        ]

        if any(re.search(item, normalized) for item in dangerous_action_patterns):
            return True

        return False

    def _is_export_action_only_request(self, prompt, locations=None, poi_terms=None, dayparts=None):
        text = self._vijay_norm_text(prompt)
        export_action = self._vijay_has_any(
            text,
            [
                "export",
                "upload",
                "send to meta",
                "meta immediately",
                "without manual approval",
                "immediately without approval",
                "push to meta",
            ],
        )

        if not export_action:
            return False

        has_targeting_intent = bool(locations or poi_terms or dayparts)
        has_business_intent = self._vijay_has_any(
            text,
            [
                "coffee",
                "cafe",
                "restaurant",
                "shawarma",
                "bakery",
                "gym",
                "fitness",
                "retail",
                "store",
                "coworking",
                "office",
            ],
        )

        return not has_targeting_intent and not has_business_intent

    def _allowed_export_poi_terms_for_request(
        self,
        poi_terms,
    ):
        terms = [
            self._vijay_norm_text(term)
            for term in (poi_terms or [])
            if self._vijay_norm_text(term)
        ]
        joined = " ".join(terms)

        coffee = self._vijay_is_coffee_intent(joined)
        restaurant = self._vijay_is_restaurant_intent(
            joined
        )
        retail = self._vijay_is_retail_intent(joined)
        gym = self._vijay_is_gym_intent(joined)
        bakery = self._vijay_is_bakery_intent(joined)

        # Generic taxonomy values such as ``retail``, ``store``,
        # and ``shop`` may be introduced during broad semantic
        # expansion. They are not enough to prove that the user
        # explicitly requested a retail audience when a stronger
        # food or coffee intent is present.
        concrete_retail = any(
            value in joined
            for value in (
                "shopping mall",
                "shopping_mall",
                "mall",
                "fashion",
                "apparel",
                "clothing",
                "clothing store",
                "clothing_store",
                "department store",
                "department_store",
                "shoe store",
                "shoe_store",
            )
        )

        food_intent = bool(
            coffee
            or restaurant
            or bakery
        )

        allow_retail_expansion = bool(
            retail
            and (
                concrete_retail
                or not food_intent
            )
        )

        allowed = []

        if coffee:
            allowed += [
                "cafe",
                "coffee",
                "coffee_shop",
                "bakery",
            ]

        if bakery:
            allowed += [
                "bakery",
                "cafe",
                "coffee_shop",
                "food",
                "restaurant",
            ]

        if restaurant:
            allowed += [
                "restaurant",
                "shawarma_restaurant",
                "middle_eastern_restaurant",
                "fast_food_restaurant",
                "meal_takeaway",
                "food_court",
                "food",
            ]

        # Do not turn the word "shop" in "coffee shop" into
        # retail intent. Retail must be explicitly represented.
        if allow_retail_expansion:
            allowed += [
                "retail",
                "store",
                "shopping_mall",
                "clothing_store",
                "shoe_store",
                "department_store",
                "fashion",
            ]

        if gym:
            allowed += [
                "gym",
                "fitness",
                "fitness_center",
                "fitness_centre",
                "health_club",
                "sports_club",
                "yoga_studio",
                "fitness_studio",
            ]

        if self._vijay_has_any(
            joined,
            [
                "coworking",
                "office",
                "business center",
                "business centre",
                "workspace",
            ],
        ):
            allowed += [
                "coworking_space",
                "corporate_office",
                "office",
                "business_center",
                "business_centre",
            ]

        if self._vijay_has_any(
            joined,
            [
                "casino",
                "gaming",
                "gambling",
            ],
        ):
            allowed += [
                "casino",
                "gaming_venue",
                "tourist_attraction",
                "entertainment",
            ]

        return self._vijay_dedupe(
            allowed or terms
        )

    def _extract_poi_terms(self, prompt):
        text = self._vijay_norm_text(prompt)

        # Priority rule: caffeine/coffee after office means cafe audience in evening,
        # not office/coworking audience.
        if self._vijay_is_coffee_intent(text):
            return ["cafe", "coffee", "coffee_shop"]

        if self._vijay_is_bakery_intent(text):
            return ["bakery", "cafe", "food"]

        if self._vijay_is_gym_intent(text):
            return ["gym", "fitness", "fitness_center"]

        if self._vijay_has_any(text, ["casino", "casinos", "gaming venues", "gaming venue", "tourist entertainment"]):
            return ["casino", "gaming_venue", "tourist_attraction", "entertainment"]

        if self._vijay_has_any(
            text,
            [
                "coworking",
                "coworking hubs",
                "business centers",
                "business centres",
                "flexible workspace",
                "flexible workspaces",
                "workspace",
                "workspaces",
            ],
        ):
            return ["coworking_space", "corporate_office", "office"]

        if self._vijay_is_restaurant_intent(text):
            if "shawarma" in text:
                return ["shawarma_restaurant", "middle_eastern_restaurant", "restaurant"]
            return ["restaurant", "food", "meal_takeaway"]

        if self._vijay_is_retail_intent(text):
            return ["retail", "shopping_mall", "store", "fashion"]

        return []

    def _extract_daypart_terms_for_prompt(self, prompt):
        text = self._vijay_norm_text(prompt)
        dayparts = []

        if self._vijay_has_any(text, ["morning", "breakfast", "before work"]):
            dayparts.append("morning")
        if self._vijay_has_any(text, ["afternoon", "lunch"]):
            dayparts.append("afternoon")
        if self._vijay_has_any(
            text,
            [
                "evening",
                "after work",
                "after office",
                "post work",
                "post office",
                "dinner",
                "night",
                "late night",
                "office hours",
            ],
        ):
            dayparts.append("evening")

        return self._vijay_dedupe(dayparts)

    def _extract_location_terms(self, prompt, cohorts=None):
        import re

        prompt_norm = self._vijay_norm_text(prompt)
        available = []

        if cohorts is not None and hasattr(cohorts, "columns") and "location_name" in cohorts.columns:
            try:
                available = [
                    str(x).strip().lower()
                    for x in cohorts["location_name"].dropna().unique().tolist()
                    if str(x).strip()
                ]
            except Exception:
                available = []

        # Explicit known specific locations first. This prevents
        # "Westmount Montreal" from being reduced to available broad "montreal".
        specific_known = [
            ("times square new york", "times square, new york"),
            ("times square, new york", "times square, new york"),
            ("westmount montreal", "westmount montreal"),
        ]

        for needle, canonical in specific_known:
            if self._vijay_norm_text(needle) in prompt_norm:
                return [canonical]

        # Special case: prompt says only New York, do not auto-expand to Times Square.
        if "new york" in prompt_norm and "times square" not in prompt_norm:
            for loc in available:
                if self._vijay_norm_text(loc) == "new york":
                    return [loc]
            return ["new york"]

        matched = []
        for loc in available:
            loc_norm = self._vijay_norm_text(loc)
            if not loc_norm:
                continue
            if loc_norm in prompt_norm:
                matched.append(loc)

        if matched:
            matched_sorted = sorted(
                matched,
                key=lambda x: len(self._vijay_norm_text(x).split()),
                reverse=True,
            )
            most_specific = matched_sorted[0]
            most_specific_norm = self._vijay_norm_text(most_specific)

            filtered = []
            for loc in matched_sorted:
                loc_norm = self._vijay_norm_text(loc)
                if loc_norm == most_specific_norm:
                    filtered.append(loc)
                    continue
                if loc_norm in most_specific_norm and loc_norm != most_specific_norm:
                    continue
                filtered.append(loc)

            return self._vijay_dedupe(filtered)

        known_locations = [
            "san francisco",
            "montreal",
            "quebec",
            "canada",
            "usa",
            "united states",
            "india",
            "california",
            "ontario",
        ]

        found = []
        for loc in known_locations:
            if self._vijay_norm_text(loc) in prompt_norm:
                found.append(loc)

        if found:
            return [sorted(found, key=lambda x: len(self._vijay_norm_text(x).split()), reverse=True)[0]]

        m = re.search(r"\bnear\s+([a-zA-Z][a-zA-Z\s,.-]{2,80})", str(prompt))
        if m:
            phrase = m.group(1)
            phrase = re.split(r"[.?!;:]", phrase)[0]
            phrase = re.sub(r"\b(find|people|who|visit|during|hours|for|with)\b.*$", "", phrase, flags=re.I)
            phrase = phrase.strip(" ,.-").lower()
            if phrase:
                return [phrase]

        return []

    def _run_broad_location_guardrail(self, prompt_filter_report):
        broad_locations = {
            "canada",
            "usa",
            "united states",
            "united states of america",
            "india",
            "california",
            "ontario",
            "quebec province",
            "british columbia",
            "texas",
            "new york state",
        }

        locations = [
            self._vijay_norm_text(x)
            for x in (prompt_filter_report or {}).get("locations_detected", [])
            if self._vijay_norm_text(x)
        ]

        blocked_locations = [x for x in locations if x in broad_locations]
        blocked = bool(blocked_locations)

        return {
            "enabled": True,
            "blocked": blocked,
            "block_export": blocked,
            "export_blocked": blocked,
            "should_block": blocked,
            "export_allowed": not blocked,
            "should_export": not blocked,
            "downstream_export_enabled": not blocked,
            "filter_mode": "broad_location_no_export" if blocked else "allowed",
            "reason": "country_or_broad_location_requires_city_area" if blocked else "location_scope_ok",
            "blocked_locations": blocked_locations,
            "locations_detected": locations,
        }

    # Removed _vijay_is_broad_city_request and _vijay_location_matches_request

    def _vijay_poi_matches_allowed(self, cohort_poi, allowed_terms):
        poi = self._vijay_snake(cohort_poi)
        allowed = [self._vijay_snake(x) for x in (allowed_terms or []) if self._vijay_norm_text(x)]

        if not allowed:
            return True

        for term in allowed:
            if poi == term:
                return True
            if term in {"coffee", "coffee_shop"} and poi in {"cafe", "coffee_shop"}:
                return True
            if term == "restaurant" and poi.endswith("_restaurant"):
                return True
            if term == "food" and poi in {"restaurant", "food_court", "meal_takeaway"}:
                return True

        return False

    def _filter_to_requested_export_category(self, cohorts, prompt_filter_report):
        import pandas as pd

        if cohorts is None:
            cohorts = pd.DataFrame()

        report = dict(prompt_filter_report or {})
        selected = cohorts.copy()

        if (
            report.get(
                "eligible_for_audience_selection"
            )
            is False
            or report.get("filter_mode")
            == "needs_clarification"
        ):
            report.update(
                {
                    "filter_mode": (
                        "needs_clarification"
                    ),
                    "fulfillment_status": (
                        "needs_clarification"
                    ),
                    "block_export": True,
                    "export_blocked": True,
                    "downstream_export_enabled": False,
                }
            )
            return selected.iloc[0:0].copy(), report

        locations = report.get("locations_detected") or report.get("locations") or []
        poi_terms = report.get("poi_terms_detected") or report.get("poi_terms") or []
        dayparts = report.get("dayparts_detected") or report.get("dayparts") or []

        allowed_pois = self._allowed_export_poi_terms_for_request(poi_terms)

        report["enabled"] = True
        report["block_export"] = False
        report["export_blocked"] = False

        if selected.empty:
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["export_blocked"] = True
            report["fulfillment_status"] = "blocked"
            report["allowed_export_poi_terms"] = allowed_pois
            return selected, report

        if "privacy_status" in selected.columns:
            selected = selected[
                selected["privacy_status"].fillna("").astype(str).str.lower().isin(["passed", "pass", "safe", ""])
            ].copy()

        if locations and "location_name" in selected.columns:
            selected = selected[
                selected["location_name"].apply(
                    lambda loc: any(location_matches_request(loc, req) for req in locations)
                )
            ].copy()

        location_matched_count = len(selected)

        if allowed_pois and "primary_poi_type" in selected.columns:
            selected = selected[
                selected["primary_poi_type"].apply(
                    lambda poi: self._vijay_poi_matches_allowed(poi, allowed_pois)
                )
            ].copy()

        category_matched_count = len(selected)

        if dayparts and "created_day_part" in selected.columns:
            daypart_norms = {self._vijay_norm_text(x) for x in dayparts}
            selected = selected[
                selected["created_day_part"].apply(lambda x: self._vijay_norm_text(x) in daypart_norms)
            ].copy()

        if selected.empty and locations and allowed_pois:
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["export_blocked"] = True
            report["fulfillment_status"] = "blocked"
            report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                "requested location/category/daypart had no exact safe cohort; export blocked instead of falling back."
            ]
        else:
            parts = []
            if locations:
                parts.append("location")
            if allowed_pois:
                parts.append("poi")
            if dayparts:
                parts.append("daypart")
            report["filter_mode"] = (
                "+".join(parts)
                if parts
                else "needs_clarification"
            )

        report["locations_detected"] = locations
        report["poi_terms_detected"] = poi_terms
        report["dayparts_detected"] = dayparts
        report["allowed_export_poi_terms"] = allowed_pois
        report["location_matched_count"] = int(location_matched_count)
        report["category_matched_count"] = int(category_matched_count)
        missing_requested_locations = []
        matched_requested_locations = []
        if locations and "location_name" in selected.columns:
            for requested_location in locations:
                has_location = any(
                    location_matches_request(cohort_location, requested_location)
                    for cohort_location in selected["location_name"].dropna().tolist()
                )
                if not has_location:
                    missing_requested_locations.append(str(requested_location))
                else:
                    matched_requested_locations.append(str(requested_location))

        if not locations:
            fulfillment_status = "complete"
        elif not matched_requested_locations:
            fulfillment_status = "blocked"
        elif missing_requested_locations:
            fulfillment_status = "partial"
        else:
            fulfillment_status = "complete"

        report["fulfillment_status"] = fulfillment_status
        report["missing_requested_locations"] = missing_requested_locations
        report["matched_requested_locations"] = matched_requested_locations

        if fulfillment_status == "blocked" and locations:
            selected = selected.iloc[0:0].copy()
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["export_blocked"] = True
            report["downstream_export_enabled"] = False
            report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                f"{location} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                for location in missing_requested_locations
            ]
        elif missing_requested_locations:
            report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                f"{location} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                for location in missing_requested_locations
            ]

        if not selected.empty and locations:
            sort_cols = [
                col for col in ["quality_score", "total_maid_volume"]
                if col in selected.columns
            ]
            selected = self._select_balanced_location_candidates(
                candidate=selected,
                requested_locations=locations,
                sort_cols=sort_cols,
                max_rows=len(selected),
            ).copy()

        report["selected_count"] = int(len(selected))

        return selected, report

    def _select_cohorts_for_prompt(self, prompt, cohorts, semantic_intent=None):
        import pandas as pd

        if cohorts is None:
            cohorts = pd.DataFrame()

        if semantic_intent and "locations" in semantic_intent:
            locations = semantic_intent["locations"]
        else:
            locations = self._extract_location_terms(prompt, cohorts)

        if semantic_intent and "categories" in semantic_intent:
            poi_terms = semantic_intent["categories"]
        else:
            poi_terms = self._extract_poi_terms(prompt)

        if semantic_intent and "dayparts" in semantic_intent:
            dayparts = semantic_intent["dayparts"]
        else:
            dayparts = self._extract_daypart_terms_for_prompt(prompt)

        if self._is_raw_identifier_request(prompt):
            empty = cohorts.iloc[0:0].copy()
            return empty, {
                "enabled": True,
                "filter_mode": "privacy_identifier_request_blocked",
                "block_export": True,
                "export_blocked": True,
                "downstream_export_enabled": False,
                "reason": "raw_identifier_request_blocked",
                "locations_detected": locations,
                "poi_terms_detected": poi_terms,
                "dayparts_detected": dayparts,
                "coverage_warnings": [
                    "Raw MAIDs, device IDs, and individual-level user data cannot be exported. Only privacy-safe aggregated cohorts are allowed."
                ],
                "selected_count": 0,
            }

        if self._is_export_action_only_request(prompt, locations=locations, poi_terms=poi_terms, dayparts=dayparts):
            empty = cohorts.iloc[0:0].copy()
            return empty, {
                "enabled": True,
                "filter_mode": "export_action_requires_existing_audience",
                "block_export": True,
                "export_blocked": True,
                "downstream_export_enabled": False,
                "reason": "export_requires_existing_approved_audience",
                "locations_detected": locations,
                "poi_terms_detected": poi_terms,
                "dayparts_detected": dayparts,
                "coverage_warnings": [
                    "Export requests require an existing selected audience/run and manual approval. No new audience was generated from an action-only prompt."
                ],
                "selected_count": 0,
            }
        # _active_empty_targeting_abstention_v3
        if (
            not locations
            and not poi_terms
            and not dayparts
        ):
            empty = cohorts.iloc[0:0].copy()
            return empty, {
                "filter_mode": "needs_clarification",
                "locations_detected": [],
                "locations": [],
                "poi_terms_detected": [],
                "poi_terms": [],
                "requested_categories": [],
                "dayparts_detected": [],
                "dayparts": [],
                "attempts": [],
                "audience_request_detected": False,
                "eligible_for_audience_selection": False,
                "missing_required_constraints": [
                    "location",
                    "category",
                ],
                "fulfillment_status": (
                    "needs_clarification"
                ),
                "block_export": True,
                "export_blocked": True,
                "downstream_export_enabled": False,
            }


        report = {
            "enabled": True,
            "locations_detected": locations,
            "poi_terms_detected": poi_terms,
            "dayparts_detected": dayparts,
        }

        broad_report = self._run_broad_location_guardrail(report)
        if broad_report.get("block_export"):
            empty = cohorts.iloc[0:0].copy() if hasattr(cohorts, "iloc") else pd.DataFrame()
            report.update(broad_report)
            report["filter_mode"] = "broad_location_no_export"
            return empty, report

        selected, report = self._filter_to_requested_export_category(cohorts, report)

        if not selected.empty:
            sort_cols = [
                col for col in ["final_match_score", "quality_score", "total_maid_volume"]
                if col in selected.columns
            ]
            if sort_cols:
                selected = selected.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

        return selected, report

    def _strict_category_export_guardrail(self, prompt_filter_report, selected_cohorts, v2_result=None):
        import pandas as pd

        report = dict(prompt_filter_report or {})
        selected = selected_cohorts if selected_cohorts is not None else pd.DataFrame()
        v2_result = v2_result or {}

        locations = report.get("locations_detected") or report.get("locations") or []
        poi_terms = report.get("poi_terms_detected") or report.get("poi_terms") or []
        dayparts = report.get("dayparts_detected") or report.get("dayparts") or []
        allowed_pois = self._allowed_export_poi_terms_for_request(poi_terms)

        reasons = []

        if report.get("filter_mode") in {"location_category_gap_no_export", "broad_location_no_export"}:
            reasons.append(report.get("filter_mode"))

        if selected is None or getattr(selected, "empty", True):
            if locations and allowed_pois:
                reasons.append("empty_selected_cohorts_for_requested_location_category")

        if selected is not None and not getattr(selected, "empty", True):
            if locations and "location_name" in selected.columns:
                bad_locations = [
                    str(x)
                    for x in selected["location_name"].dropna().tolist()
                    if not any(location_matches_request(x, req) for req in locations)
                ]
                if bad_locations:
                    reasons.append("selected_location_mismatch")

            if allowed_pois and "primary_poi_type" in selected.columns:
                bad_pois = [
                    str(x)
                    for x in selected["primary_poi_type"].dropna().tolist()
                    if not self._vijay_poi_matches_allowed(x, allowed_pois)
                ]
                if bad_pois:
                    reasons.append("selected_category_mismatch")

            if dayparts and "created_day_part" in selected.columns:
                allowed_dayparts = {self._vijay_norm_text(x) for x in dayparts}
                bad_dayparts = [
                    str(x)
                    for x in selected["created_day_part"].dropna().tolist()
                    if self._vijay_norm_text(x) not in allowed_dayparts
                ]
                if bad_dayparts:
                    reasons.append("selected_daypart_mismatch")

        reasons = self._vijay_dedupe(reasons)
        blocked = bool(reasons)

        if not blocked:
            reason = "safe_to_export"
        elif "location_category_gap_no_export" in reasons or "empty_selected_cohorts_for_requested_location_category" in reasons:
            reason = "strict_location_category_requested_but_no_safe_exact_match"
        elif "selected_location_mismatch" in reasons:
            reason = "strict_location_category_requested_but_selected_location_mismatch"
        elif "selected_category_mismatch" in reasons:
            reason = "strict_location_category_requested_but_selected_category_mismatch"
        elif "selected_daypart_mismatch" in reasons:
            reason = "strict_location_category_requested_but_selected_daypart_mismatch"
        else:
            reason = reasons[0]

        return {
            "enabled": True,
            "blocked": blocked,
            "block_export": blocked,
            "export_blocked": blocked,
            "should_block": blocked,
            "export_allowed": not blocked,
            "should_export": not blocked,
            "reason": reason,
            "reasons": reasons,
            "filter_mode": "blocked" if blocked else "allowed",
            "locations_detected": locations,
            "poi_terms_detected": poi_terms,
            "dayparts_detected": dayparts,
            "allowed_export_poi_terms": allowed_pois,
        }

    def _build_hybrid_retrieval_intent(self, prompt, prompt_filter_report=None, privacy_cohorts=None):
        prompt_filter_report = dict(prompt_filter_report or {})

        locations = (
            prompt_filter_report.get("locations_detected")
            or prompt_filter_report.get("locations")
            or self._extract_location_terms(prompt, privacy_cohorts)
        )
        poi_terms = (
            prompt_filter_report.get("poi_terms_detected")
            or prompt_filter_report.get("poi_terms")
            or self._extract_poi_terms(prompt)
        )
        dayparts = (
            prompt_filter_report.get("dayparts_detected")
            or prompt_filter_report.get("dayparts")
            or self._extract_daypart_terms_for_prompt(prompt)
        )

        business_intent = prompt_filter_report.get("business_intent") or "unknown_business_intent"

        prompt_norm = self._vijay_norm_text(prompt)
        available_pois = []
        if privacy_cohorts is not None and hasattr(privacy_cohorts, "columns") and "primary_poi_type" in privacy_cohorts.columns:
            available_pois = [
                str(x).strip().lower()
                for x in privacy_cohorts["primary_poi_type"].dropna().unique().tolist()
                if str(x).strip()
            ]

        if business_intent == "unknown_business_intent":
            if self._vijay_has_any(prompt_norm, ["casino", "casinos", "gaming"]):
                business_intent = "casino"
                if "casino" not in poi_terms:
                    poi_terms = ["casino"] + list(poi_terms or [])
            elif self._vijay_is_coffee_intent(prompt_norm):
                business_intent = "coffee_cafe"
            elif self._vijay_is_restaurant_intent(prompt_norm):
                business_intent = "restaurant_food"
            elif self._vijay_is_retail_intent(prompt_norm):
                business_intent = "retail_shopping"

        allowed = self._allowed_export_poi_terms_for_request(poi_terms)
        matched_available = [
            poi for poi in available_pois
            if self._vijay_poi_matches_allowed(poi, allowed)
        ]

        return {
            "business_intent": business_intent,
            "locations": self._vijay_dedupe(locations),
            "locations_detected": self._vijay_dedupe(locations),
            "poi_terms": self._vijay_dedupe(poi_terms),
            "poi_terms_detected": self._vijay_dedupe(poi_terms),
            "allowed_export_poi_terms": allowed,
            "matched_available_poi_types": self._vijay_dedupe(matched_available or allowed),
            "requested_categories": self._vijay_dedupe(allowed),
            "dayparts": self._vijay_dedupe(dayparts),
            "dayparts_detected": self._vijay_dedupe(dayparts),
            "retrieval_mode": "hybrid_safe_derived",
            "confidence_score": 0.85 if business_intent != "unknown_business_intent" else 0.55,
        }

    def _merge_v2_intent_into_prompt_filter_report(
        self,
        prompt_filter_report,
        v2_result,
    ):
        merged = dict(prompt_filter_report or {})

        # _active_semantic_snapshot_v3
        _pre_semantic_merge = dict(merged)
        v2_result = v2_result or {}

        intent = (
            v2_result.get("prompt_intent")
            or v2_result.get("intent")
            or {}
        )

        resolver_mode = str(
            intent.get("resolver_mode") or ""
        )

        semantic_authoritative = bool(
            intent.get("local_semantic_used")
            or intent.get("llm_used")
            or resolver_mode.startswith(
                "local_semantic_"
            )
            or resolver_mode == "llm_rag_primary"
        )

        def clean_list(*values):
            output = []

            for value in values:
                if isinstance(value, str):
                    value = [value]

                if not isinstance(
                    value,
                    (list, tuple, set),
                ):
                    continue

                output.extend(
                    item
                    for item in value
                    if str(item or "").strip()
                )

            return self._vijay_dedupe(output)

        v2_locations = clean_list(
            intent.get("locations"),
            intent.get("locations_detected"),
        )

        v2_categories = clean_list(
            intent.get("requested_categories"),
            intent.get("canonical_categories"),
        )

        v2_selection_pois = clean_list(
            v2_categories,
            intent.get(
                "matched_available_poi_types"
            ),
        )

        v2_support_terms = clean_list(
            intent.get("poi_terms_detected"),
            intent.get("poi_terms"),
        )

        v2_dayparts = clean_list(
            intent.get("dayparts"),
            intent.get("dayparts_detected"),
        )

        canonical_locations = [
            self._vijay_snake(value)
            for value in v2_locations
        ]
        canonical_pois = [
            self._vijay_snake(value)
            for value in v2_selection_pois
        ]
        canonical_support_terms = [
            self._vijay_snake(value)
            for value in v2_support_terms
        ]
        canonical_dayparts = [
            self._vijay_norm_text(value)
            for value in v2_dayparts
        ]

        merged["semantic_support_terms"] = (
            self._vijay_dedupe(
                canonical_support_terms
            )
        )

        if semantic_authoritative:
            if canonical_locations:
                merged["locations_detected"] = (
                    self._vijay_dedupe(
                        canonical_locations
                    )
                )
                merged["locations"] = (
                    self._vijay_dedupe(
                        canonical_locations
                    )
                )

            # Semantic intent replaces the partial first-match
            # parser rather than merely filling empty fields.
            merged["poi_terms_detected"] = (
                self._vijay_dedupe(
                    canonical_pois
                )
            )
            merged["poi_terms"] = (
                self._vijay_dedupe(
                    canonical_pois
                )
            )

            # An empty semantic daypart is meaningful: the user
            # did not request one. Do not retain or invent one.
            merged["dayparts_detected"] = (
                self._vijay_dedupe(
                    canonical_dayparts
                )
            )
            merged["dayparts"] = (
                self._vijay_dedupe(
                    canonical_dayparts
                )
            )

            merged["requested_categories"] = (
                self._vijay_dedupe(
                    [
                        self._vijay_snake(value)
                        for value in v2_categories
                    ]
                )
            )

            if intent.get("business_intent"):
                merged["business_intent"] = (
                    intent.get("business_intent")
                )

        else:
            if (
                not merged.get("locations_detected")
                and canonical_locations
            ):
                merged["locations_detected"] = (
                    self._vijay_dedupe(
                        canonical_locations
                    )
                )
                merged["locations"] = (
                    self._vijay_dedupe(
                        canonical_locations
                    )
                )

            if (
                not merged.get("poi_terms_detected")
                and canonical_pois
            ):
                merged["poi_terms_detected"] = (
                    self._vijay_dedupe(
                        canonical_pois
                    )
                )
                merged["poi_terms"] = (
                    self._vijay_dedupe(
                        canonical_pois
                    )
                )

            if (
                not merged.get("dayparts_detected")
                and canonical_dayparts
            ):
                merged["dayparts_detected"] = (
                    self._vijay_dedupe(
                        canonical_dayparts
                    )
                )
                merged["dayparts"] = (
                    self._vijay_dedupe(
                        canonical_dayparts
                    )
                )

        # _active_restore_omitted_fields_v3
        _location_fields_present = any(
            key in intent
            for key in (
                "locations",
                "locations_detected",
            )
        )
        _category_fields_present = any(
            key in intent
            for key in (
                "requested_categories",
                "canonical_categories",
                "matched_available_poi_types",
            )
        )
        _daypart_fields_present = any(
            key in intent
            for key in (
                "dayparts",
                "dayparts_detected",
            )
        )

        if semantic_authoritative:
            if not _location_fields_present:
                _saved_locations = clean_list(
                    _pre_semantic_merge.get(
                        "locations_detected"
                    ),
                    _pre_semantic_merge.get(
                        "locations"
                    ),
                )
                if _saved_locations:
                    _saved_locations = (
                        self._vijay_dedupe(
                            [
                                self._vijay_snake(value)
                                for value
                                in _saved_locations
                            ]
                        )
                    )
                    merged[
                        "locations_detected"
                    ] = _saved_locations
                    merged[
                        "locations"
                    ] = _saved_locations

            if not _category_fields_present:
                _saved_categories = clean_list(
                    _pre_semantic_merge.get(
                        "requested_categories"
                    ),
                    _pre_semantic_merge.get(
                        "poi_terms_detected"
                    ),
                    _pre_semantic_merge.get(
                        "poi_terms"
                    ),
                )
                if _saved_categories:
                    _saved_categories = (
                        self._vijay_dedupe(
                            [
                                self._vijay_snake(value)
                                for value
                                in _saved_categories
                            ]
                        )
                    )
                    merged[
                        "poi_terms_detected"
                    ] = _saved_categories
                    merged[
                        "poi_terms"
                    ] = _saved_categories
                    merged[
                        "requested_categories"
                    ] = _saved_categories

            if not _daypart_fields_present:
                _saved_dayparts = clean_list(
                    _pre_semantic_merge.get(
                        "dayparts_detected"
                    ),
                    _pre_semantic_merge.get(
                        "dayparts"
                    ),
                )
                if _saved_dayparts:
                    _saved_dayparts = (
                        self._vijay_dedupe(
                            [
                                self._vijay_norm_text(
                                    value
                                )
                                for value
                                in _saved_dayparts
                            ]
                        )
                    )
                    merged[
                        "dayparts_detected"
                    ] = _saved_dayparts
                    merged[
                        "dayparts"
                    ] = _saved_dayparts

        parts = []

        if merged.get("locations_detected"):
            parts.append("location")
        if merged.get("poi_terms_detected"):
            parts.append("poi")
        if merged.get("dayparts_detected"):
            parts.append("daypart")

        business_intent = str(
            merged.get("business_intent")
            or intent.get("business_intent")
            or "unknown_business_intent"
        ).strip().lower()

        has_location = bool(
            merged.get("locations_detected")
        )
        has_category = bool(
            merged.get("requested_categories")
            or merged.get("poi_terms_detected")
        )

        audience_request_detected = bool(
            has_category
            or business_intent
            not in {
                "",
                "unknown_business_intent",
                "general_audience",
                "none",
                "null",
            }
        )

        missing_required_constraints = []

        if not has_location:
            missing_required_constraints.append(
                "location"
            )

        if not has_category:
            missing_required_constraints.append(
                "category"
            )

        eligible_for_selection = bool(
            audience_request_detected
            and has_location
            and has_category
        )

        merged["audience_request_detected"] = (
            audience_request_detected
        )
        merged[
            "eligible_for_audience_selection"
        ] = eligible_for_selection
        merged[
            "missing_required_constraints"
        ] = missing_required_constraints

        if eligible_for_selection:
            merged["filter_mode"] = "+".join(parts)
        else:
            merged["filter_mode"] = (
                "needs_clarification"
            )
            merged["block_export"] = True
            merged["export_blocked"] = True
            merged[
                "downstream_export_enabled"
            ] = False
            merged["fulfillment_status"] = (
                "needs_clarification"
            )

        merged["v2_intent_merged"] = bool(parts)
        merged[
            "semantic_intent_authoritative"
        ] = semantic_authoritative
        merged["v2_resolver_mode"] = resolver_mode
        merged["v2_confidence_score"] = (
            intent.get("confidence_score")
        )
        merged["v2_llm_used"] = (
            intent.get("llm_used")
        )
        merged["local_semantic_used"] = (
            intent.get("local_semantic_used")
        )
        merged["data_gap_likely"] = bool(
            intent.get("data_gap_likely")
            or not merged.get(
                "eligible_for_audience_selection",
                False,
            )
        )
        merged["missing_location_coverage"] = (
            intent.get(
                "missing_location_coverage"
            )
            or []
        )

        merged["ai_intent_merge"] = {
            "llm_used": intent.get("llm_used"),
            "local_semantic_used": (
                intent.get(
                    "local_semantic_used"
                )
            ),
            "resolver_mode": resolver_mode,
            "confidence_score": (
                intent.get("confidence_score")
            ),
            "data_gap_likely": (
                intent.get("data_gap_likely")
            ),
            "semantic_authoritative": (
                semantic_authoritative
            ),
        }

        v2_quality = str(intent.get("quality_intent") or "").lower().strip()
        existing_quality = str(merged.get("quality_intent") or "").lower().strip()

        if v2_quality not in {"high", "balanced", "broad"}:
            v2_quality = None

        if existing_quality not in {"high", "balanced", "broad"}:
            existing_quality = None

        merged["quality_intent"] = v2_quality or existing_quality or "balanced"

        return merged

    # === Vijay local recovery guardrails: END ===
