from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AutonomousV2SwarmReviewAgent:
    """
    Reviews v2 autonomous audience intelligence output.

    This does not export anything.
    It only reviews:
    - freshness
    - embedding strength
    - ranking coverage
    - mutation suggestions
    - approval safety
    """

    def review_run(self, run_dir: str | Path) -> dict[str, Any]:
        run_path = Path(run_dir)

        final_summary_path = run_path / "final_prompt_summary.json"
        v2_summary_path = run_path / "06_v2_autonomous_preview" / "v2_preview_summary.json"
        output_path = run_path / "06_v2_autonomous_preview" / "v2_swarm_review.json"

        final_summary = self._read_json(final_summary_path)
        v2 = final_summary.get("v2_autonomous") or self._read_json(v2_summary_path)

        if not v2:
            result = {
                "status": "missing_v2_output",
                "run_dir": str(run_path),
                "overall_review_status": "blocked",
                "signals": [],
                "recommendations": [
                    "Run Audience Intelligence v2 before swarm review."
                ],
                "approval_required": True,
                "downstream_export_enabled": False,
            }
            self._write_json(output_path, result)
            return result

        freshness = v2.get("data_freshness") or {}
        embed = v2.get("embedding_manifest") or {}
        mutation = v2.get("mutation") or {}
        warnings = v2.get("coverage_warnings") or []
        suggestions = mutation.get("mutation_suggestions") or []

        data_gap_count = sum(
            1 for item in suggestions
            if str(item.get("mutation_type")) == "data_gap"
        )

        signals = [
            self._freshness_signal(freshness),
            self._embedding_signal(embed),
            self._coverage_signal(warnings),
            self._mutation_signal(mutation, data_gap_count),
            self._safety_signal(v2),
        ]

        overall = self._overall_status(signals)

        result = {
            "status": "completed",
            "run_dir": str(run_path),
            "overall_review_status": overall,
            "v2_status": v2.get("status"),
            "freshness_status": freshness.get("freshness_status"),
            "latest_source_timestamp": freshness.get("latest_source_timestamp"),
            "source_rows_checked": freshness.get("source_rows_checked"),
            "vector_count": embed.get("vector_count"),
            "vector_dimension": embed.get("vector_dimension"),
            "ranked_match_count": v2.get("ranked_match_count"),
            "coverage_warning_count": len(warnings),
            "coverage_warnings": warnings,
            "mutation_suggestion_count": mutation.get("suggestion_count"),
            "data_gap_count": data_gap_count,
            "approval_required": v2.get("approval_required", True),
            "downstream_export_enabled": v2.get("downstream_export_enabled", False),
            "signals": signals,
            "recommendations": self._recommendations(signals),
        }

        self._write_json(output_path, result)
        return result

    def _freshness_signal(self, freshness: dict[str, Any]) -> dict[str, Any]:
        status = str(freshness.get("freshness_status", "unknown")).lower()

        if status == "fresh":
            severity = "ok"
            message = "Fresh source data is available."
        elif status == "unchanged":
            severity = "watch"
            message = "Source data is unchanged compared with previous run."
        elif status == "stale":
            severity = "warning"
            message = "Source data is stale. Refresh Echo/Postgres data before approval."
        else:
            severity = "watch"
            message = "Freshness could not be fully verified."

        return {
            "name": "data_freshness",
            "severity": severity,
            "value": status,
            "message": message,
        }

    def _embedding_signal(self, embed: dict[str, Any]) -> dict[str, Any]:
        vector_count = int(embed.get("vector_count") or 0)
        vector_dimension = int(embed.get("vector_dimension") or 0)

        if vector_count > 0 and vector_dimension >= 384:
            severity = "ok"
            message = "All-safe-cohort embedding strength is acceptable."
        elif vector_count > 0:
            severity = "watch"
            message = "Embeddings exist, but vector dimension is below recommended 384."
        else:
            severity = "warning"
            message = "No v2 embeddings were created."

        return {
            "name": "embedding_strength",
            "severity": severity,
            "vector_count": vector_count,
            "vector_dimension": vector_dimension,
            "message": message,
        }

    def _coverage_signal(self, warnings: list[str]) -> dict[str, Any]:
        if warnings:
            return {
                "name": "coverage",
                "severity": "review",
                "warning_count": len(warnings),
                "message": "Coverage gaps detected. Human review is required before approval.",
            }

        return {
            "name": "coverage",
            "severity": "ok",
            "warning_count": 0,
            "message": "No coverage gaps detected.",
        }

    def _mutation_signal(self, mutation: dict[str, Any], data_gap_count: int) -> dict[str, Any]:
        suggestion_count = int(mutation.get("suggestion_count") or 0)

        if data_gap_count > 0:
            severity = "review"
            message = "Data-gap mutation suggestions exist and need review."
        elif suggestion_count > 0:
            severity = "watch"
            message = "Fallback/mutation suggestions exist and should be reviewed."
        else:
            severity = "ok"
            message = "No mutation suggestions required."

        return {
            "name": "mutation_review",
            "severity": severity,
            "suggestion_count": suggestion_count,
            "data_gap_count": data_gap_count,
            "message": message,
        }

    def _safety_signal(self, v2: dict[str, Any]) -> dict[str, Any]:
        approval_required = bool(v2.get("approval_required", True))
        downstream_export_enabled = bool(v2.get("downstream_export_enabled", False))

        if approval_required and not downstream_export_enabled:
            severity = "ok"
            message = "Output is approval-gated and downstream export is disabled."
        else:
            severity = "blocked"
            message = "Safety issue: approval gating or downstream export setting is unsafe."

        return {
            "name": "export_safety",
            "severity": severity,
            "approval_required": approval_required,
            "downstream_export_enabled": downstream_export_enabled,
            "message": message,
        }

    def _overall_status(self, signals: list[dict[str, Any]]) -> str:
        severities = {signal.get("severity") for signal in signals}

        if "blocked" in severities or "warning" in severities:
            return "blocked"
        if "review" in severities:
            return "needs_human_review"
        if "watch" in severities:
            return "watch"
        return "ready_for_human_approval"

    def _recommendations(self, signals: list[dict[str, Any]]) -> list[str]:
        recs = []

        for signal in signals:
            name = signal.get("name")
            severity = signal.get("severity")

            if name == "data_freshness" and severity in {"warning", "watch"}:
                recs.append("Refresh or verify Echo/Postgres source freshness before approval.")

            if name == "embedding_strength" and severity != "ok":
                recs.append("Rebuild v2 all-safe-cohort embeddings with EMBEDDING_MAX_FEATURES=384.")

            if name == "coverage" and severity == "review":
                recs.append("Review coverage gaps and decide whether to wait for fresh Echo data or approve fallback cohorts.")

            if name == "mutation_review" and severity in {"review", "watch"}:
                recs.append("Review mutation suggestions before exporting any audience.")

            if name == "export_safety" and severity == "blocked":
                recs.append("Block export until approval_required=True and downstream_export_enabled=False.")

        if not recs:
            recs.append("Run is ready for human approval review.")

        return recs

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        try:
            return json.loads(path.read_text(errors="ignore"))
        except Exception:
            return {}

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
