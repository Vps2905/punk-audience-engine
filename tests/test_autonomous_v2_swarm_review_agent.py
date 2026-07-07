import json

from app.agents.autonomous_v2_swarm_review_agent import AutonomousV2SwarmReviewAgent


def test_v2_swarm_review_detects_needs_human_review(tmp_path):
    run_dir = tmp_path / "prompt_test"
    run_dir.mkdir()

    final_summary = {
        "v2_autonomous": {
            "status": "completed",
            "data_freshness": {
                "freshness_status": "fresh",
                "latest_source_timestamp": "2026-07-06T18:46:16+00:00",
                "source_rows_checked": 216,
            },
            "embedding_manifest": {
                "vector_count": 95,
                "vector_dimension": 384,
            },
            "ranked_match_count": 95,
            "coverage_warnings": [
                "san francisco was requested, but no exact safe cohort matched the requested category/daypart."
            ],
            "mutation": {
                "suggestion_count": 2,
                "mutation_suggestions": [
                    {"mutation_type": "data_gap"},
                    {"mutation_type": "broader_daypart"},
                ],
            },
            "approval_required": True,
            "downstream_export_enabled": False,
        }
    }

    (run_dir / "final_prompt_summary.json").write_text(json.dumps(final_summary))

    result = AutonomousV2SwarmReviewAgent().review_run(run_dir)

    assert result["status"] == "completed"
    assert result["overall_review_status"] == "needs_human_review"
    assert result["vector_dimension"] == 384
    assert result["data_gap_count"] == 1
    assert result["approval_required"] is True
    assert result["downstream_export_enabled"] is False
    assert (run_dir / "06_v2_autonomous_preview" / "v2_swarm_review.json").exists()


def test_v2_swarm_review_blocks_unsafe_export(tmp_path):
    run_dir = tmp_path / "prompt_test"
    run_dir.mkdir()

    final_summary = {
        "v2_autonomous": {
            "status": "completed",
            "data_freshness": {"freshness_status": "fresh"},
            "embedding_manifest": {"vector_count": 10, "vector_dimension": 384},
            "ranked_match_count": 10,
            "coverage_warnings": [],
            "mutation": {"suggestion_count": 0, "mutation_suggestions": []},
            "approval_required": False,
            "downstream_export_enabled": True,
        }
    }

    (run_dir / "final_prompt_summary.json").write_text(json.dumps(final_summary))

    result = AutonomousV2SwarmReviewAgent().review_run(run_dir)

    assert result["overall_review_status"] == "blocked"


def test_v2_swarm_review_handles_missing_v2(tmp_path):
    run_dir = tmp_path / "prompt_test"
    run_dir.mkdir()
    (run_dir / "final_prompt_summary.json").write_text("{}")

    result = AutonomousV2SwarmReviewAgent().review_run(run_dir)

    assert result["status"] == "missing_v2_output"
    assert result["overall_review_status"] == "blocked"
