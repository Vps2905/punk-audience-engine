from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.agents.schema_profiler_agent import SchemaProfilerAgent
from app.agents.workflow_planner_agent import WorkflowPlannerAgent


class SwarmCoordinator:
    """
    Coordinates agent-like workflow.

    Current v1.1:
    - profiles data
    - creates adaptive workflow plan

    Next:
    - connect privacy service
    - connect synthetic service
    - connect embedding service
    - connect cohort service
    - connect export service
    - add Postgres source agent
    """

    def __init__(self) -> None:
        self.schema_profiler = SchemaProfilerAgent()
        self.workflow_planner = WorkflowPlannerAgent()

    def plan_from_csv(self, csv_path: str | Path) -> Dict[str, Any]:
        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)

        schema_profile = self.schema_profiler.profile_dataframe(
            df=df,
            source_name=path.name,
        )

        workflow_plan = self.workflow_planner.create_plan(schema_profile)

        return {
            "agent": "swarm_coordinator",
            "status": "ready_for_adaptive_execution",
            "source": {
                "type": "csv",
                "path": str(path),
            },
            "schema_profile": schema_profile,
            "workflow_plan": workflow_plan,
            "swarm_architecture": self._swarm_architecture(),
        }

    def _swarm_architecture(self) -> Dict[str, Any]:
        return {
            "mode": "multi_agent_swarm",
            "agents": [
                "data_source_agent",
                "schema_profiler_agent",
                "privacy_agent",
                "aggregation_agent",
                "synthetic_agent",
                "embedding_agent",
                "cohort_agent",
                "lookalike_agent",
                "export_agent",
                "swarm_monitor_agent",
            ],
            "coordination_strategy": "supervisor_plans_then_agents_execute",
            "background_processes": [
                "monitor_new_data",
                "monitor_cohort_quality",
                "trigger_reprofile_when_schema_changes",
                "trigger_recluster_when_distribution_changes",
            ],
        }
