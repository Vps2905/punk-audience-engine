from app.core.approval_workflow import ApprovalWorkflow
from app.core.audit_logger import AuditLogger
from app.core.privacy_budget_ledger import PrivacyBudgetLedger


def test_core_production_services_do_not_write_local_files(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )
    monkeypatch.setenv(
        "PRODUCTION_MODE",
        "true",
    )
    monkeypatch.setenv(
        "ALLOW_LOCAL_FILE_STORAGE",
        "false",
    )

    audit_path = tmp_path / "audit" / "audit.jsonl"
    ledger_path = tmp_path / "ledger" / "ledger.jsonl"
    approval_dir = tmp_path / "approvals"
    output_dir = tmp_path / "run"

    audit = AuditLogger(audit_path).log(
        "test_event",
        {"run_id": "run_1"},
    )
    ledger = PrivacyBudgetLedger(
        ledger_path,
        max_epsilon_per_run=10,
    ).record_spend(
        run_id="run_1",
        module="privacy",
        epsilon=1,
        engine="test",
    )
    approval = ApprovalWorkflow(
        approval_dir
    ).create_request(
        run_id="run_1",
        module="synthetic",
        output_dir=output_dir,
        artifacts=[],
        summary={},
        persist_artifacts=False,
    )

    assert audit["storage_backend"] == "run_history_jsonb"
    assert ledger["storage_backend"] == "run_history_jsonb"
    assert approval["storage_backend"] == "run_history_jsonb"

    assert not audit_path.exists()
    assert not ledger_path.exists()
    assert not approval_dir.exists()
    assert not output_dir.exists()
