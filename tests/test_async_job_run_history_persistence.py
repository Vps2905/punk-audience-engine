from app.api import audience_intelligence_jobs as jobs


class FakeJobStore:
    backend = "postgres"

    def __init__(self):
        self.record = {
            "tenant_id": "tenant-a",
            "job_id": "job_async_1",
            "status": "queued",
            "payload": {
                "prompt": "privacy-safe coffee audience",
                "source": "postgres",
            },
            "result": None,
            "error": None,
        }
        self.updates = []

    def get(self, job_id, *, tenant_id):
        assert job_id == "job_async_1"
        assert tenant_id == "tenant-a"
        return self.record

    def update_status(
        self,
        job_id,
        *,
        tenant_id,
        status,
        stage,
        message,
        result=None,
        error=None,
    ):
        assert tenant_id == "tenant-a"
        self.record["status"] = status

        if result is not None:
            self.record["result"] = result

        if error is not None:
            self.record["error"] = error

        self.updates.append(
            {
                "status": status,
                "stage": stage,
                "message": message,
                "result": result,
                "error": error,
            }
        )
        return self.record


class FakeAgent:
    def run(self, **kwargs):
        return {
            "status": "completed",
            "run_id": "run_async_1",
            "prompt": kwargs["prompt"],
            "source_mode": "postgres_safe_derived",
            "source_rows": 20,
            "freshness_status": "stale",
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
            "block_export": True,
            "safe_export": {
                "approval_status": "blocked_stale_source",
                "downstream_export_enabled": False,
                "block_export": True,
                "package": {
                    "cohorts": [
                        {
                            "export_cohort_id": "cohort_1",
                            "audience_name": "Safe cohort",
                        }
                    ]
                },
            },
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "individual_user_data_exported": False,
            },
            "run_dir": "unused-production-run-dir",
        }


class PersistedHistory:
    def __init__(self):
        self.final_summary = None

    def persist_run(self, *, tenant_id, final_summary):
        assert tenant_id == "tenant-a"
        self.final_summary = final_summary
        return {
            "enabled": True,
            "status": "persisted",
            "run_id": final_summary["run_id"],
        }


class FailedHistory:
    def persist_run(self, *, tenant_id, final_summary):
        assert tenant_id == "tenant-a"
        return {
            "enabled": True,
            "status": "failed",
            "run_id": final_summary["run_id"],
            "error": "database unavailable",
        }


def _patch_common(monkeypatch, store, history):
    monkeypatch.setattr(jobs, "job_store", store)
    monkeypatch.setattr(
        jobs,
        "AudienceIntelligenceOrchestratorAgent",
        lambda: FakeAgent(),
    )
    monkeypatch.setattr(
        jobs,
        "AudienceRunHistoryService",
        lambda: history,
    )
    monkeypatch.setattr(
        jobs,
        "local_file_storage_allowed",
        lambda: False,
    )
    monkeypatch.setattr(
        jobs,
        "_build_business_summary",
        lambda result: "Safe business summary",
    )


def test_async_job_persists_run_history_before_completion(
    monkeypatch,
):
    store = FakeJobStore()
    history = PersistedHistory()
    _patch_common(monkeypatch, store, history)

    jobs._run_job_background("job_async_1", "tenant-a")

    assert history.final_summary is not None
    assert history.final_summary["run_id"] == "run_async_1"
    assert history.final_summary["tenant_id"] == "tenant-a"
    assert (
        history.final_summary["business_summary"]
        == "Safe business summary"
    )

    assert store.updates[-1]["status"] == "completed"

    result = store.updates[-1]["result"]
    assert result["run_history"]["status"] == "persisted"
    assert (
        result["business_summary_path"]
        == "postgres://audience_jobs.result"
        "?job_id=job_async_1&field=business_summary"
    )
    assert result["approval_status"] == "blocked_stale_source"
    assert result["downstream_export_enabled"] is False


def test_async_postgres_job_fails_when_run_history_fails(
    monkeypatch,
):
    store = FakeJobStore()
    history = FailedHistory()
    _patch_common(monkeypatch, store, history)

    jobs._run_job_background("job_async_1", "tenant-a")

    assert store.updates[-1]["status"] == "failed"
    assert store.updates[-1]["error"] == (
        "audience_job_failed:"
        "run_history_persistence:"
        "AudienceRunHistoryPersistenceError"
    )
