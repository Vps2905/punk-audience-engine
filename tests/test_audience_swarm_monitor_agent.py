from __future__ import annotations

import json
from pathlib import Path

from app.agents.audience_swarm_monitor_agent import AudienceSwarmMonitorAgent


def test_swarm_monitor_generates_health_report(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    runs_dir = tmp_path / "prompt_runs"
    report_dir = tmp_path / "reports"

    jobs_dir.mkdir()
    run_dir = runs_dir / "prompt_test_run"
    export_dir = run_dir / "05_safe_export"
    export_dir.mkdir(parents=True)

    job = {
        "job_id": "job_test_123",
        "status": "completed",
        "created_at": "2026-07-03T00:00:00+00:00",
        "updated_at": "2026-07-03T00:01:00+00:00",
        "payload": {"prompt": "restaurant evening audience"},
        "progress": {"stage": "completed", "message": "done"},
        "result": {
            "run_id": "prompt_test_run",
            "prompt": "restaurant evening audience",
            "coverage_warnings": ["san francisco was requested but not export ready"],
        },
        "error": None,
    }

    (jobs_dir / "job_test_123.json").write_text(json.dumps(job))

    final_summary = {
        "run_id": "prompt_test_run",
        "prompt": "restaurant evening audience",
        "coverage_warnings": ["san francisco was requested but not export ready"],
        "source_mode": "postgres_safe_derived",
        "prompt_selected_cohorts": 6,
    }

    manifest = {
        "approval_status": "pending_approval",
        "downstream_export_enabled": False,
        "exported_cohorts": 5,
        "exported_lookalike_pairs": 13,
    }

    (run_dir / "final_prompt_summary.json").write_text(json.dumps(final_summary))
    (export_dir / "safe_export_manifest.json").write_text(json.dumps(manifest))

    agent = AudienceSwarmMonitorAgent(
        jobs_dir=jobs_dir,
        prompt_runs_dir=runs_dir,
        report_dir=report_dir,
    )

    report = agent.run_once(stale_pending_hours=999999)

    assert report["status"] == "completed"
    assert report["health_status"] == "warning"
    assert report["job_summary"]["total_jobs"] == 1
    assert report["approval_summary"]["pending_approval_count"] == 1
    assert report["warning_summary"]["coverage_warning_job_count"] == 1
    assert report["privacy_guarantees"]["raw_maids_exposed"] is False

    assert Path(report["report_path"]).exists()
    assert Path(report["latest_report_path"]).exists()
