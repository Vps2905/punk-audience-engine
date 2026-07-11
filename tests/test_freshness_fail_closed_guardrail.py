import json

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_stale_freshness_builds_fail_closed_guardrail():
    agent = AudienceIntelligenceOrchestratorAgent()

    guardrail = agent._build_freshness_guardrail(
        {
            "data_freshness": {
                "freshness_status": "stale",
                "latest_source_timestamp": "2026-07-08T08:40:40+00:00",
                "source_age_hours": 82.75,
                "freshness_threshold_hours": 48.0,
                "reason": "Latest source timestamp is older than 48 hours.",
            }
        }
    )

    assert guardrail["freshness_status"] == "stale"
    assert guardrail["block_export"] is True
    assert guardrail["approval_status"] == "blocked_stale_source"
    assert guardrail["downstream_export_enabled"] is False
    assert "48 hours" in guardrail["block_export_reason"]


def test_stale_freshness_overrides_safe_export_manifest(tmp_path):
    agent = AudienceIntelligenceOrchestratorAgent()

    manifest_path = tmp_path / "safe_export_manifest.json"
    manifest_path.write_text(json.dumps({"approval_status": "pending_approval"}))

    export_result = {
        "status": "completed",
        "approval_status": "pending_approval",
        "downstream_export_enabled": False,
        "exported_cohorts": 2,
        "outputs": {
            "safe_export_manifest": str(manifest_path),
        },
    }

    guardrail = {
        "freshness_status": "stale",
        "block_export": True,
        "approval_status": "blocked_stale_source",
        "downstream_export_enabled": False,
        "block_export_reason": "Latest source timestamp is older than 48 hours.",
    }

    result = agent._apply_freshness_fail_closed(
        export_result=export_result,
        freshness_guardrail=guardrail,
    )

    assert result["approval_status"] == "blocked_stale_source"
    assert result["downstream_export_enabled"] is False
    assert result["block_export"] is True
    assert result["export_blocked_until_source_refresh"] is True

    rewritten = json.loads(manifest_path.read_text())
    assert rewritten["approval_status"] == "blocked_stale_source"
    assert rewritten["block_export"] is True


def test_fresh_freshness_does_not_block_export():
    agent = AudienceIntelligenceOrchestratorAgent()

    guardrail = agent._build_freshness_guardrail(
        {
            "data_freshness": {
                "freshness_status": "fresh",
            }
        }
    )

    result = agent._apply_freshness_fail_closed(
        export_result={
            "status": "completed",
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "outputs": {},
        },
        freshness_guardrail=guardrail,
    )

    assert result["approval_status"] == "pending_approval"
    assert result["block_export"] is False
