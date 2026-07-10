import json
import os
from copy import deepcopy

import pytest
from sqlalchemy import create_engine, text

from app.services.audience_run_history_service import AudienceRunHistoryService


def _postgres_test_db_url() -> str:
    return os.getenv("AUDIENCE_TEST_DATABASE_URL", "")


pytestmark = pytest.mark.skipif(
    not _postgres_test_db_url().startswith(("postgresql://", "postgres://")),
    reason=(
        "Postgres integration test skipped. "
        "Set AUDIENCE_TEST_DATABASE_URL to a disposable Postgres test database to run it."
    ),
)


def _clean_review_ready_summary(run_id: str, budget_scope: str, epsilon: float, max_budget: float):
    return {
        "run_id": run_id,
        "status": "completed",
        "pipeline": "test_privacy_budget_integration",
        "source_mode": "postgres_safe_derived",
        "source_rows_checked": 1500,
        "freshness": "fresh",
        "freshness_status": "fresh",
        "latest_source_timestamp": "2026-07-10T10:00:00Z",
        "approval_required": True,
        "downstream_export_enabled": False,
        "prompt_filter_report": {
            "locations_detected": ["montreal"],
            "poi_terms_detected": ["cafe"],
            "dayparts_detected": ["evening"],
        },
        "v2_swarm_review": {
            "overall_review_status": "review_ready",
            "approval_required": True,
            "data_freshness": {
                "status": "fresh",
                "stale_data_warning": False,
            },
        },
        "safe_export": {
            "export_blocked": False,
            "downstream_export_enabled": False,
            "budget_scope": budget_scope,
            "epsilon": epsilon,
            "delta": 1e-5,
            "sensitivity": 1.0,
            "max_budget": max_budget,
            "mechanism": "gaussian",
            "raw_maids_exported": False,
            "hashed_identifiers_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
        },
        "privacy_guarantees": {
            "raw_maids_exported": False,
            "hashed_identifiers_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
            "only_aggregated_or_synthetic_outputs": True,
        },
        "safe_export_cohorts": [
            {
                "cohort_id": f"{run_id}_cohort_1",
                "name": "Montreal cafe evening cohort",
                "size": 1500,
                "quality_score": 0.86,
                "approval_status": "pending_approval",
            }
        ],
    }


@pytest.fixture()
def postgres_history_service(monkeypatch):
    db_url = _postgres_test_db_url()

    # Disposable integration DB cleanup.
    # This keeps privacy-budget integration tests repeatable.
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    monkeypatch.setenv("AUDIENCE_HISTORY_DATABASE_URL", db_url)
    monkeypatch.setenv("AUDIENCE_PRIVACY_BUDGET_DATABASE_URL", db_url)
    return AudienceRunHistoryService()


def test_approval_spends_privacy_budget_and_records_audit(postgres_history_service):
    service = postgres_history_service

    summary = _clean_review_ready_summary(
        run_id="test_run_budget_allowed",
        budget_scope="test_customer_allowed_integration",
        epsilon=1.0,
        max_budget=5.0,
    )

    persisted = service.persist_run(final_summary=summary, selected_cohorts=summary.get("safe_export_cohorts"))
    run_id = persisted.get("run_id") or summary["run_id"]

    approval = service.approve_run(
        run_id=run_id,
        actor="privacy_test_reviewer",
        note="integration approval should spend budget",
        downstream_export_enabled=True,
    )

    assert approval["status"] == "approved"
    assert approval["approval_status"] == "approved"
    assert approval["downstream_export_enabled"] is True

    audit = service.get_audit(run_id)
    audit_text = json.dumps(audit, default=str)

    assert "privacy_budget_spent" in audit_text
    assert "run_approved" in audit_text


def test_approval_blocks_when_privacy_budget_exceeded(postgres_history_service):
    service = postgres_history_service

    first_summary = _clean_review_ready_summary(
        run_id="test_run_budget_first",
        budget_scope="test_customer_exhausted_integration",
        epsilon=1.0,
        max_budget=1.5,
    )

    second_summary = deepcopy(first_summary)
    second_summary["run_id"] = "test_run_budget_second"
    second_summary["safe_export_cohorts"][0]["cohort_id"] = "test_run_budget_second_cohort_1"

    first_persisted = service.persist_run(final_summary=first_summary, selected_cohorts=first_summary.get("safe_export_cohorts"))
    first_run_id = first_persisted.get("run_id") or first_summary["run_id"]

    first_approval = service.approve_run(
        run_id=first_run_id,
        actor="privacy_test_reviewer",
        note="first approval should spend budget",
        downstream_export_enabled=True,
    )

    assert first_approval["status"] == "approved"
    assert first_approval["downstream_export_enabled"] is True

    second_persisted = service.persist_run(final_summary=second_summary, selected_cohorts=second_summary.get("safe_export_cohorts"))
    second_run_id = second_persisted.get("run_id") or second_summary["run_id"]

    second_approval = service.approve_run(
        run_id=second_run_id,
        actor="privacy_test_reviewer",
        note="second approval should be blocked by budget",
        downstream_export_enabled=True,
    )

    assert second_approval["status"] == "blocked"
    assert second_approval["approval_status"] == "blocked_privacy_budget"
    assert second_approval["downstream_export_enabled"] is False
    assert "Privacy budget exceeded." in second_approval["blockers"]

    audit = service.get_audit(second_run_id)
    audit_text = json.dumps(audit, default=str)

    assert "privacy_budget_blocked" in audit_text



def test_approval_without_exported_cohorts_does_not_spend_privacy_budget(postgres_history_service):
    service = postgres_history_service

    summary = _clean_review_ready_summary(
        run_id="test_no_cohort_budget_guard",
        budget_scope="test_no_cohort_budget_guard_scope",
        epsilon=1.0,
        max_budget=5.0,
    )
    summary["safe_export_cohorts"] = []

    persisted = service.persist_run(final_summary=summary)
    run_id = persisted.get("run_id") or summary["run_id"]

    approval = service.approve_run(
        run_id=run_id,
        actor="privacy_test_reviewer",
        note="should block before spending budget",
        downstream_export_enabled=True,
    )

    assert approval["status"] == "blocked"
    assert approval["reason"] == "no_exported_cohorts_to_approve"

    audit = service.get_audit(run_id)
    audit_text = json.dumps(audit, default=str)

    assert "approval_blocked" in audit_text
    assert "privacy_budget_spent" not in audit_text
