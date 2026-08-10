from pathlib import Path


def test_prompt_ui_combines_all_coverage_warning_sources():
    source = Path(
        "app/static/audience_workspace.js"
    ).read_text(encoding="utf-8")

    assert "return Array.from(new Set([" in source
    assert "...(data.coverage_warnings || [])" in source
    assert "...(review.coverage_warnings || [])" in source
    assert "...(v2.coverage_warnings || [])" in source
    assert (
        "renderWarnings(uniqueWarnings(data))" in source
    )


def test_workspace_does_not_expose_server_credentials_or_data_paths():
    html = Path(
        "app/static/audience_workspace.html"
    ).read_text(encoding="utf-8")
    script = Path(
        "app/static/audience_workspace.js"
    ).read_text(encoding="utf-8")

    assert "Audience API Key" not in html
    assert "Tenant Signature" not in html
    assert "safe_cohort_path" not in html
    assert "X-Audience-API-Key" not in script
    assert "X-Audience-Tenant-Id" not in script
    assert 'JSON.stringify({ prompt })' in script
    assert 'source: "postgres"' not in script
    assert ".innerHTML" not in script
    assert "data.pipeline_stages" in script
    assert "data.terminal_policy_decision" in script
    assert 'terminalPolicy ? "Not evaluated"' in script
