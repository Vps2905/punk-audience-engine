from __future__ import annotations

import json

import pytest

from app.services.production_module5_repeated_shadow_evidence_service import (
    ProductionModule5RepeatedShadowEvidenceService,
    discover_historical_prompt_results,
)


TENANT = "tenant_a"


def pending_result(index: int) -> dict:
    return {
        "status": "completed",
        "run_id": f"run-{index}",
        "prompt": f"Generalized audience objective number {index}.",
        "source_rows": 220,
        "freshness_status": "fresh",
        "approval_status": "pending_approval",
        "approval_required": True,
        "downstream_export_enabled": False,
        "prompt_selected_cohorts": 3,
        "embedding": {"vector_count": 90, "vector_dimension": 384},
        "safe_export": {
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
            "exported_cohorts": 1,
        },
    }


def stale_result(index: int) -> dict:
    value = pending_result(index)
    value["freshness_status"] = "stale"
    value["approval_status"] = "blocked_stale_source"
    value["safe_export"] = {
        **value["safe_export"],
        "approval_status": "blocked_stale_source",
        "block_export": True,
    }
    return value


def privacy_result(index: int) -> dict:
    return {
        "status": "skipped",
        "run_id": f"run-{index}",
        "prompt": f"Privacy safety objective number {index}.",
        "source_rows": None,
        "freshness_status": "not_evaluated",
        "approval_status": "blocked_privacy_identifier_request",
        "approval_required": True,
        "downstream_export_enabled": False,
        "prompt_selected_cohorts": 0,
        "prompt_filter_report": {
            "filter_mode": "privacy_identifier_request_blocked",
        },
    }


def write_result(root, index, value):
    run = root / f"run-{index:03d}"
    run.mkdir(parents=True)
    path = run / "final_prompt_summary.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def passing_results(tmp_path):
    paths = []
    for index in range(15):
        paths.append(write_result(tmp_path, index, pending_result(index)))
    for index in range(15, 20):
        paths.append(write_result(tmp_path, index, stale_result(index)))
    for index in range(20, 25):
        paths.append(write_result(tmp_path, index, privacy_result(index)))
    return paths


def test_real_historical_results_generate_ready_minimized_evidence(tmp_path):
    service = ProductionModule5RepeatedShadowEvidenceService(environment={})
    paths = passing_results(tmp_path)

    inventory = service.inventory(
        tenant_id=TENANT,
        result_paths=paths,
    )
    report = service.run(
        tenant_id=TENANT,
        result_paths=paths,
    )

    assert inventory == {
        "historical_result_count": 25,
        "unique_objective_count": 25,
        "terminal_case_count": 10,
        "nonterminal_review_case_count": 15,
        "unsafe_result_count": 0,
        "prompt_content_stored": False,
        "source_paths_stored": False,
    }
    assert report["status"] == "engineering_preview_ready"
    assert report["summary"]["evaluated_run_count"] == 25
    assert report["review"]["eligible_for_staging_review"] is True
    assert report["review"]["live_cutover_authorized"] is False
    serialized = json.dumps(report)
    assert "Generalized audience objective" not in serialized
    assert "Privacy safety objective" not in serialized
    assert str(tmp_path) not in serialized


def test_discovery_accepts_only_canonical_final_summaries(tmp_path):
    expected = write_result(tmp_path, 1, pending_result(1))
    ignored = tmp_path / "run-002" / "other.json"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("{}", encoding="utf-8")

    assert discover_historical_prompt_results([tmp_path]) == (
        expected.absolute(),
    )


def test_unsafe_historical_result_is_rejected_before_certification(tmp_path):
    value = pending_result(1)
    value["downstream_export_enabled"] = True
    path = write_result(tmp_path, 1, value)

    with pytest.raises(ValueError, match="unsafe effect signal"):
        ProductionModule5RepeatedShadowEvidenceService(environment={}).run(
            tenant_id=TENANT,
            result_paths=[path],
        )


def test_release_effect_flag_blocks_before_historical_file_read(tmp_path):
    missing = tmp_path / "missing.json"
    service = ProductionModule5RepeatedShadowEvidenceService(
        environment={"MODULE5_PRODUCTION_ROUTING_ENABLED": "true"}
    )

    with pytest.raises(RuntimeError, match="release-effect"):
        service.run(
            tenant_id=TENANT,
            result_paths=[missing],
        )
    with pytest.raises(RuntimeError, match="release-effect"):
        service.inventory(
            tenant_id=TENANT,
            result_paths=[missing],
        )


def test_missing_prompt_and_cross_tenant_results_fail_closed(tmp_path):
    missing_prompt = pending_result(1)
    missing_prompt.pop("prompt")
    missing_path = write_result(tmp_path, 1, missing_prompt)

    with pytest.raises(ValueError, match="original prompt"):
        ProductionModule5RepeatedShadowEvidenceService(environment={}).run(
            tenant_id=TENANT,
            result_paths=[missing_path],
        )

    other_root = tmp_path / "other"
    cross_tenant = pending_result(2)
    cross_tenant["tenant_id"] = "tenant_other"
    cross_path = write_result(other_root, 2, cross_tenant)
    with pytest.raises(ValueError, match="tenant lineage"):
        ProductionModule5RepeatedShadowEvidenceService(environment={}).run(
            tenant_id=TENANT,
            result_paths=[cross_path],
        )
