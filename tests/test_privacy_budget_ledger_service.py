from pathlib import Path

from app.services.privacy_budget_ledger_service import (
    PrivacyBudgetLedgerService,
    PrivacyBudgetRequest,
)


def test_privacy_budget_ledger_allows_budget_when_available(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'budget.db'}"
    service = PrivacyBudgetLedgerService(database_url=db_url)

    result = service.check_and_record(
        PrivacyBudgetRequest(
            run_id="run_1",
            cohort_id="cohort_1",
            budget_scope="customer_1",
            epsilon=1.0,
            max_budget=5.0,
            actor="test",
        )
    )

    assert result["enabled"] is True
    assert result["status"] == "allowed"
    assert result["spent"] is True
    assert result["budget_before"] == 0.0
    assert result["budget_after"] == 1.0
    assert result["remaining_budget"] == 4.0


def test_privacy_budget_ledger_blocks_when_budget_exceeded(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'budget.db'}"
    service = PrivacyBudgetLedgerService(database_url=db_url)

    first = service.check_and_record(
        PrivacyBudgetRequest(
            run_id="run_1",
            cohort_id="cohort_1",
            budget_scope="customer_1",
            epsilon=4.5,
            max_budget=5.0,
            actor="test",
        )
    )

    second = service.check_and_record(
        PrivacyBudgetRequest(
            run_id="run_2",
            cohort_id="cohort_2",
            budget_scope="customer_1",
            epsilon=1.0,
            max_budget=5.0,
            actor="test",
        )
    )

    assert first["status"] == "allowed"
    assert second["status"] == "blocked"
    assert second["spent"] is False
    assert second["budget_before"] == 4.5
    assert second["budget_after"] == 4.5
    assert second["remaining_budget"] == 0.5


def test_privacy_budget_status_reports_remaining_budget(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'budget.db'}"
    service = PrivacyBudgetLedgerService(database_url=db_url)

    service.check_and_record(
        PrivacyBudgetRequest(
            run_id="run_1",
            cohort_id="cohort_1",
            budget_scope="customer_1",
            epsilon=2.0,
            max_budget=5.0,
            actor="test",
        )
    )

    status = service.get_budget_status("customer_1", max_budget=5.0)

    assert status["status"] == "ok"
    assert status["used_budget"] == 2.0
    assert status["remaining_budget"] == 3.0
    assert status["budget_exhausted"] is False
