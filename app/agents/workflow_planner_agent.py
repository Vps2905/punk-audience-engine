from __future__ import annotations

from typing import Any, Dict, List


class WorkflowPlannerAgent:
    """
    Creates a dynamic workflow plan based on the schema profile.

    This is the main change from fixed pipeline to adaptive pipeline.
    """

    def create_plan(self, schema_profile: Dict[str, Any]) -> Dict[str, Any]:
        row_count = schema_profile.get("row_count", 0)
        pii_columns = schema_profile.get("pii_columns", [])
        cohort_columns = schema_profile.get("cohort_candidate_columns", [])
        embedding_columns = schema_profile.get("embedding_candidate_columns", [])
        quality_warnings = schema_profile.get("quality_warnings", [])

        recommended_k_min = self._recommend_k_min(row_count)
        workflow_steps = self._build_steps(
            pii_columns=pii_columns,
            cohort_columns=cohort_columns,
            embedding_columns=embedding_columns,
            row_count=row_count,
        )

        return {
            "agent": "workflow_planner_agent",
            "status": "planned",
            "workflow_mode": "adaptive",
            "recommended_k_min": recommended_k_min,
            "recommended_epsilon": 1.0,
            "pii_columns_to_hash_or_remove": pii_columns,
            "cohort_columns": cohort_columns,
            "embedding_columns": embedding_columns,
            "workflow_steps": workflow_steps,
            "warnings": quality_warnings,
            "next_action": self._next_action(row_count, cohort_columns),
        }

    def _recommend_k_min(self, row_count: int) -> int:
        if row_count >= 1000:
            return 1000
        if row_count >= 100:
            return 100
        return max(1, row_count)

    def _build_steps(
        self,
        pii_columns: List[str],
        cohort_columns: List[str],
        embedding_columns: List[str],
        row_count: int,
    ) -> List[Dict[str, Any]]:
        steps = []

        steps.append({
            "step": 1,
            "agent": "data_source_agent",
            "action": "load_data",
            "reason": "Load data from CSV now or Postgres later.",
        })

        steps.append({
            "step": 2,
            "agent": "schema_profiler_agent",
            "action": "profile_schema",
            "reason": "Understand actual columns before deciding workflow.",
        })

        steps.append({
            "step": 3,
            "agent": "privacy_agent",
            "action": "hash_or_remove_sensitive_fields",
            "fields": pii_columns,
            "reason": "Sensitive fields must not enter cohort/export layers.",
        })

        steps.append({
            "step": 4,
            "agent": "aggregation_agent",
            "action": "aggregate_by_safe_traits",
            "fields": cohort_columns,
            "reason": "Create cohort-level groups instead of user-level exports.",
        })

        steps.append({
            "step": 5,
            "agent": "privacy_agent",
            "action": "apply_k_anonymity_and_dp_noise",
            "reason": "Block small groups and add basic DP noise.",
        })

        steps.append({
            "step": 6,
            "agent": "synthetic_agent",
            "action": "generate_synthetic_variants",
            "preferred_engine": "SDV_DPGCSynthesizer_when_available",
            "fallback": "aggregated_sampler",
            "reason": "Create safe synthetic seeds instead of raw user rows.",
        })

        if embedding_columns:
            steps.append({
                "step": 7,
                "agent": "embedding_agent",
                "action": "embed_safe_signals",
                "fields": embedding_columns,
                "reason": "Convert safe traits into vectors for search and lookalikes.",
            })

        steps.append({
            "step": 8,
            "agent": "cohort_agent",
            "action": "cluster_score_and_create_cohorts",
            "reason": "Create high-quality cohorts dynamically from data signals.",
        })

        steps.append({
            "step": 9,
            "agent": "export_agent",
            "action": "create_safe_export_package",
            "reason": "Export only aggregated traits and synthetic seed profiles.",
        })

        steps.append({
            "step": 10,
            "agent": "swarm_monitor_agent",
            "action": "monitor_quality_and_trigger_replanning",
            "reason": "Background evolution when new data arrives or quality changes.",
        })

        return steps

    def _next_action(self, row_count: int, cohort_columns: List[str]) -> str:
        if row_count == 0:
            return "Connect a valid data source."
        if not cohort_columns:
            return "Need human review because no usable cohort columns were detected."
        return "Run adaptive privacy and cohort pipeline."
