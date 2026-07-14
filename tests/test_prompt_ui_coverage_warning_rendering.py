from pathlib import Path


def test_prompt_ui_combines_all_coverage_warning_sources():
    source = Path(
        "app/api/audience_intelligence_prompt.py"
    ).read_text(encoding="utf-8")

    assert (
        "const coverageWarnings = Array.from(new Set(["
        in source
    )
    assert "...(data.coverage_warnings || [])" in source
    assert "...(review.coverage_warnings || [])" in source
    assert "...(v2.coverage_warnings || [])" in source
    assert (
        '...coverageWarnings.map(w => "- " + w)'
        in source
    )
    assert (
        '...(v2.coverage_warnings || []).map(w => "- " + w)'
        not in source
    )
