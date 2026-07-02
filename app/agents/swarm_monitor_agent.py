from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.agents.postgres_source_agent import PostgresSourceAgent


class SwarmMonitorAgent:
    """
    Background-style monitor for real Postgres swarm pipeline.

    It checks whether the source table changed and recommends:
    - no action
    - rerun swarm pipeline
    - reprofile schema and replan workflow

    This is v1 monitoring, not advanced mutation logic.
    """

    def __init__(self) -> None:
        self.source_agent = PostgresSourceAgent()
        self.state_dir = Path("data/swarm_monitor")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def monitor_table(
        self,
        schema_name: str = "public",
        table_name: str = "maid_extractions",
    ) -> Dict[str, Any]:
        engine = self.source_agent._engine()

        row_count = self._row_count(engine, schema_name, table_name)
        latest_created_at = self._latest_created_at(engine, schema_name, table_name)
        schema_profile = self._schema_profile(engine, schema_name, table_name)

        schema_hash = self._hash_json(schema_profile)

        state_path = self.state_dir / f"{schema_name}_{table_name}_state.json"

        previous_state = None
        if state_path.exists():
            previous_state = json.loads(state_path.read_text())

        current_state = {
            "schema_name": schema_name,
            "table_name": table_name,
            "row_count": row_count,
            "latest_created_at": latest_created_at,
            "schema_hash": schema_hash,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        changes = self._detect_changes(previous_state, current_state)

        recommendation = self._recommend_action(changes)

        report = {
            "agent": "swarm_monitor_agent",
            "status": "checked",
            "source": {
                "schema": schema_name,
                "table": table_name,
            },
            "current_state": current_state,
            "previous_state": previous_state,
            "changes": changes,
            "recommendation": recommendation,
            "monitoring_scope": [
                "row_count",
                "latest_created_at",
                "schema_hash",
            ],
        }

        state_path.write_text(json.dumps(current_state, indent=2, default=str))

        report_path = self.state_dir / f"{schema_name}_{table_name}_latest_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

        report["outputs"] = {
            "state_path": str(state_path),
            "report_path": str(report_path),
        }

        return report

    def _row_count(self, engine, schema_name: str, table_name: str) -> int:
        query = f'SELECT COUNT(*) AS row_count FROM "{schema_name}"."{table_name}";'
        df = pd.read_sql_query(query, engine)
        return int(df.iloc[0]["row_count"])

    def _latest_created_at(self, engine, schema_name: str, table_name: str) -> str | None:
        query = f'SELECT MAX(created_at) AS latest_created_at FROM "{schema_name}"."{table_name}";'
        df = pd.read_sql_query(query, engine)
        value = df.iloc[0]["latest_created_at"]

        if pd.isna(value):
            return None

        return str(value)

    def _schema_profile(self, engine, schema_name: str, table_name: str) -> list[dict[str, Any]]:
        query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %(schema_name)s
        AND table_name = %(table_name)s
        ORDER BY ordinal_position;
        """

        df = pd.read_sql_query(
            query,
            engine,
            params={
                "schema_name": schema_name,
                "table_name": table_name,
            },
        )

        return df.to_dict(orient="records")

    def _hash_json(self, value: Any) -> str:
        raw = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _detect_changes(
        self,
        previous_state: Dict[str, Any] | None,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if previous_state is None:
            return {
                "first_run": True,
                "row_count_changed": True,
                "latest_created_at_changed": True,
                "schema_changed": True,
            }

        return {
            "first_run": False,
            "row_count_changed": previous_state.get("row_count") != current_state.get("row_count"),
            "latest_created_at_changed": previous_state.get("latest_created_at") != current_state.get("latest_created_at"),
            "schema_changed": previous_state.get("schema_hash") != current_state.get("schema_hash"),
        }

    def _recommend_action(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        if changes.get("schema_changed"):
            return {
                "action": "reprofile_schema_and_replan_workflow",
                "reason": "Schema changed or this is first monitor run.",
                "should_run_swarm": True,
            }

        if changes.get("row_count_changed") or changes.get("latest_created_at_changed"):
            return {
                "action": "rerun_swarm_pipeline",
                "reason": "New or changed source data detected.",
                "should_run_swarm": True,
            }

        return {
            "action": "no_action",
            "reason": "No source table changes detected.",
            "should_run_swarm": False,
        }
