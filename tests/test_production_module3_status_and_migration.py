import json
from pathlib import Path

from app.services.production_module3_status_service import (
    ProductionModule3StatusService,
)


def test_module3_status_is_disabled_by_default_and_does_not_leak_paths():
    status = ProductionModule3StatusService(
        environment={
            "MODULE3_COHORT_EVIDENCE_PATH": "/secret/module3/report.json",
        }
    ).status()
    assert status["status"] == "module3_1_2_evidence_pending"
    assert status["feature_flags"]["cohort_generation_enabled"] is False
    assert status["feature_flags"]["lookalike_generation_enabled"] is False
    assert status["feature_flags"]["production_routing_enabled"] is False
    assert "/secret" not in str(status)
    assert status["safety"]["secret_values_returned"] is False


def test_module3_status_accepts_safe_engineering_evidence(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "status": "engineering_preview_ready",
                "safety": {
                    "raw_identifiers_returned": False,
                    "activation_or_export_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    status = ProductionModule3StatusService(
        environment={"MODULE3_COHORT_EVIDENCE_PATH": str(path)}
    ).status()
    assert status["status"] == "module3_1_2_engineering_evidence_ready"
    assert status["engineering_evidence_ready"] is True


def test_release_affecting_flags_fail_closed():
    status = ProductionModule3StatusService(
        environment={"MODULE3_LOOKALIKE_GENERATION_ENABLED": "true"}
    ).status()
    assert status["status"] == (
        "unsafe_configuration_release_affecting_feature_blocked"
    )


def test_module3_migration_enforces_k_rls_immutability_and_no_activation():
    sql = Path("migrations/0012_module3_governed_cohort_candidates.sql").read_text(
        encoding="utf-8"
    )
    assert "cohort_size >= 1000" in sql
    assert "raw_identifiers_stored = FALSE" in sql
    assert "overlap_or_unique_reach_claimed = FALSE" in sql
    assert "lookalike_generation_performed = FALSE" in sql
    assert "eligible_for_activation = FALSE" in sql
    assert "eligible_for_export = FALSE" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "prevent_module3_candidate_batch_mutation" in sql
    assert "prevent_module3_candidate_identity_change" in sql
    assert "BEFORE UPDATE OR DELETE ON audience_cohort_candidates" in sql


def test_module3_safe_defaults_and_router_registration():
    env_text = Path(".env.example").read_text(encoding="utf-8")
    for value in (
        "MODULE3_COHORT_GENERATION_ENABLED=false",
        "MODULE3_COHORT_PERSISTENCE_ENABLED=false",
        "MODULE3_LOOKALIKE_GENERATION_ENABLED=false",
        "MODULE3_PRODUCTION_ROUTING_ENABLED=false",
    ):
        assert value in env_text

    main_text = Path("app/main.py").read_text(encoding="utf-8")
    assert "audience_intelligence_module3_status_router" in main_text
    assert "app.include_router(audience_intelligence_module3_status_router)" in main_text


def test_module3_evaluation_script_loads_repository_dotenv(monkeypatch):
    from scripts import evaluate_module3_cohort_candidates as evaluation_script

    calls = []

    def fake_load_dotenv(path, *, override):
        calls.append((Path(path), override))
        return True

    monkeypatch.setattr(evaluation_script, "load_dotenv", fake_load_dotenv)
    evaluation_script.load_environment()

    assert calls == [(evaluation_script.ROOT / ".env", False)]
