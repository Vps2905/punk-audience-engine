from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AudienceSwarmMonitorAgent:
    """
    Basic Swarm / Background Monitoring Agent.

    Responsibilities:
    - Monitor audience jobs.
    - Detect failed jobs.
    - Detect pending approvals.
    - Detect coverage warnings.
    - Detect stale export approvals.
    - Produce safe operational health report.

    This agent does not read or expose raw MAIDs, raw observations, lat/lng,
    emails, phones, hashed identifiers, or individual-level rows.
    """

    def __init__(
        self,
        jobs_dir: str | Path = "data/audience_jobs",
        prompt_runs_dir: str | Path = "data/prompt_runs",
        report_dir: str | Path = "data/swarm_reports",
    ):
        self.jobs_dir = Path(jobs_dir)
        self.prompt_runs_dir = Path(prompt_runs_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run_once(
        self,
        stale_pending_hours: int = 24,
        max_failed_jobs_warning: int = 1,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)

        jobs = self._load_jobs()
        run_manifests = self._load_run_manifests()

        job_summary = self._summarize_jobs(jobs)
        approval_summary = self._summarize_approvals(jobs, run_manifests, now, stale_pending_hours)
        warning_summary = self._summarize_warnings(jobs)
        recommendations = self._build_recommendations(
            job_summary=job_summary,
            approval_summary=approval_summary,
            warning_summary=warning_summary,
            max_failed_jobs_warning=max_failed_jobs_warning,
        )

        health_status = self._health_status(
            job_summary=job_summary,
            approval_summary=approval_summary,
            warning_summary=warning_summary,
            max_failed_jobs_warning=max_failed_jobs_warning,
        )

        report = {
            "module": "Audience Swarm Monitor",
            "status": "completed",
            "health_status": health_status,
            "generated_at": now.isoformat(),
            "jobs_dir": str(self.jobs_dir),
            "prompt_runs_dir": str(self.prompt_runs_dir),
            "job_summary": job_summary,
            "approval_summary": approval_summary,
            "warning_summary": warning_summary,
            "recommendations": recommendations,
            "privacy_guarantees": {
                "raw_maids_exposed": False,
                "hashed_identifiers_exposed": False,
                "raw_observations_exposed": False,
                "raw_lat_lng_exposed": False,
                "email_phone_exposed": False,
                "individual_user_data_exposed": False,
            },
        }

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"audience_swarm_health_{timestamp}.json"
        latest_path = self.report_dir / "latest_audience_swarm_health.json"

        report["report_path"] = str(report_path)
        report["latest_report_path"] = str(latest_path)

        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        latest_path.write_text(json.dumps(report, indent=2, allow_nan=False))

        return report

    def _load_jobs(self) -> List[Dict[str, Any]]:
        if not self.jobs_dir.exists():
            return []

        jobs = []

        for path in sorted(self.jobs_dir.glob("job_*.json")):
            try:
                data = json.loads(path.read_text())
                data["_job_file"] = str(path)
                jobs.append(data)
            except Exception:
                jobs.append(
                    {
                        "job_id": path.stem,
                        "status": "unreadable",
                        "error": f"Could not parse job file: {path}",
                        "_job_file": str(path),
                    }
                )

        return jobs

    def _load_run_manifests(self) -> List[Dict[str, Any]]:
        if not self.prompt_runs_dir.exists():
            return []

        manifests = []

        for run_dir in sorted(self.prompt_runs_dir.glob("prompt_*")):
            safe_export_manifest = run_dir / "05_safe_export" / "safe_export_manifest.json"
            final_summary = run_dir / "final_prompt_summary.json"

            manifest_data: Dict[str, Any] = {
                "run_dir": str(run_dir),
                "safe_export_manifest_exists": safe_export_manifest.exists(),
                "final_summary_exists": final_summary.exists(),
            }

            if safe_export_manifest.exists():
                try:
                    manifest_data["safe_export_manifest"] = json.loads(safe_export_manifest.read_text())
                except Exception as exc:
                    manifest_data["safe_export_manifest_error"] = str(exc)

            if final_summary.exists():
                try:
                    final_data = json.loads(final_summary.read_text())
                    manifest_data["run_id"] = final_data.get("run_id")
                    manifest_data["prompt"] = final_data.get("prompt")
                    manifest_data["coverage_warnings"] = final_data.get("coverage_warnings", [])
                    manifest_data["source_mode"] = final_data.get("source_mode")
                    manifest_data["prompt_selected_cohorts"] = final_data.get("prompt_selected_cohorts")
                except Exception as exc:
                    manifest_data["final_summary_error"] = str(exc)

            manifests.append(manifest_data)

        return manifests

    def _summarize_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}

        failed_jobs = []
        running_jobs = []
        latest_completed = None

        for job in jobs:
            status = str(job.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1

            if status == "failed":
                failed_jobs.append(
                    {
                        "job_id": job.get("job_id"),
                        "error": job.get("error"),
                        "updated_at": job.get("updated_at"),
                    }
                )

            if status in {"queued", "running"}:
                running_jobs.append(
                    {
                        "job_id": job.get("job_id"),
                        "status": status,
                        "progress": job.get("progress"),
                        "updated_at": job.get("updated_at"),
                    }
                )

            if status == "completed":
                latest_completed = job.get("job_id")

        return {
            "total_jobs": len(jobs),
            "status_counts": counts,
            "failed_jobs": failed_jobs[-10:],
            "active_jobs": running_jobs[-10:],
            "latest_completed_job_id": latest_completed,
        }

    def _summarize_approvals(
        self,
        jobs: List[Dict[str, Any]],
        run_manifests: List[Dict[str, Any]],
        now: datetime,
        stale_pending_hours: int,
    ) -> Dict[str, Any]:
        pending = []
        approved = []
        not_required = []
        unknown = []

        for item in run_manifests:
            manifest = item.get("safe_export_manifest") or {}
            approval_status = manifest.get("approval_status")

            row = {
                "run_dir": item.get("run_dir"),
                "run_id": item.get("run_id"),
                "prompt": item.get("prompt"),
                "approval_status": approval_status,
                "downstream_export_enabled": manifest.get("downstream_export_enabled"),
                "exported_cohorts": manifest.get("exported_cohorts"),
                "exported_lookalike_pairs": manifest.get("exported_lookalike_pairs"),
            }

            if approval_status == "pending_approval":
                pending.append(row)
            elif approval_status == "approved":
                approved.append(row)
            elif approval_status == "not_required":
                not_required.append(row)
            else:
                unknown.append(row)

        stale_pending = []
        for item in pending:
            run_dir = item.get("run_dir")
            if not run_dir:
                continue

            path = Path(run_dir)
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                age_hours = (now - mtime).total_seconds() / 3600
            except Exception:
                age_hours = 0

            if age_hours >= stale_pending_hours:
                stale_item = dict(item)
                stale_item["age_hours"] = round(age_hours, 2)
                stale_pending.append(stale_item)

        return {
            "pending_approval_count": len(pending),
            "approved_count": len(approved),
            "not_required_count": len(not_required),
            "unknown_approval_count": len(unknown),
            "stale_pending_count": len(stale_pending),
            "pending_approvals": pending[-20:],
            "approved_exports": approved[-20:],
            "stale_pending_approvals": stale_pending[-20:],
        }

    def _summarize_warnings(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        coverage_warning_jobs = []

        for job in jobs:
            result = job.get("result") or {}
            warnings = result.get("coverage_warnings") or []

            if warnings:
                coverage_warning_jobs.append(
                    {
                        "job_id": job.get("job_id"),
                        "run_id": result.get("run_id"),
                        "prompt": result.get("prompt"),
                        "coverage_warnings": warnings,
                    }
                )

        return {
            "coverage_warning_job_count": len(coverage_warning_jobs),
            "coverage_warning_jobs": coverage_warning_jobs[-20:],
        }

    def _build_recommendations(
        self,
        *,
        job_summary: Dict[str, Any],
        approval_summary: Dict[str, Any],
        warning_summary: Dict[str, Any],
        max_failed_jobs_warning: int,
    ) -> List[str]:
        recommendations = []

        failed_count = len(job_summary.get("failed_jobs", []))
        pending_count = approval_summary.get("pending_approval_count", 0)
        stale_count = approval_summary.get("stale_pending_count", 0)
        coverage_count = warning_summary.get("coverage_warning_job_count", 0)

        if failed_count >= max_failed_jobs_warning:
            recommendations.append("Investigate failed Audience Intelligence jobs before enabling more exports.")

        if pending_count > 0:
            recommendations.append("Review pending safe export approvals.")

        if stale_count > 0:
            recommendations.append("Some exports are stale pending approvals; review or reject them.")

        if coverage_count > 0:
            recommendations.append("Review coverage warnings. Some requested locations or segments did not become export-ready.")

        if not recommendations:
            recommendations.append("Audience Intelligence system is healthy. Continue monitoring.")

        return recommendations

    def _health_status(
        self,
        *,
        job_summary: Dict[str, Any],
        approval_summary: Dict[str, Any],
        warning_summary: Dict[str, Any],
        max_failed_jobs_warning: int,
    ) -> str:
        failed_count = len(job_summary.get("failed_jobs", []))
        stale_count = approval_summary.get("stale_pending_count", 0)
        coverage_count = warning_summary.get("coverage_warning_job_count", 0)

        if failed_count >= max_failed_jobs_warning:
            return "critical"

        if stale_count > 0 or coverage_count > 0:
            return "warning"

        return "healthy"
