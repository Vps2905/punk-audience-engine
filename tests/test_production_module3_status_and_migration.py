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
    assert (
        "_include_audience_router("
        "audience_intelligence_module3_status_router)"
        in main_text
    )


def test_module3_evaluation_script_loads_repository_dotenv(monkeypatch):
    from scripts import evaluate_module3_cohort_candidates as evaluation_script

    calls = []

    def fake_load_dotenv(path, *, override):
        calls.append((Path(path), override))
        return True

    monkeypatch.setattr(evaluation_script, "load_dotenv", fake_load_dotenv)
    evaluation_script.load_environment()

    assert calls == [(evaluation_script.ROOT / ".env", False)]


def test_module3_status_accepts_safe_overlap_evidence(tmp_path):
    candidate_path = tmp_path / "candidate-report.json"
    candidate_path.write_text(
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
    overlap_path = tmp_path / "overlap-report.json"
    overlap_path.write_text(
        json.dumps(
            {
                "status": "engineering_preview_ready",
                "safety": {
                    "raw_identifiers_returned": False,
                    "membership_intersection_read": False,
                    "overlap_rate_computed": False,
                    "unique_reach_claimed": False,
                    "cohort_sizes_summed": False,
                    "candidate_lifecycle_mutated": False,
                    "activation_or_export_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    status = ProductionModule3StatusService(
        environment={
            "MODULE3_COHORT_EVIDENCE_PATH": str(candidate_path),
            "MODULE3_OVERLAP_EVIDENCE_PATH": str(overlap_path),
        }
    ).status()

    assert status["status"] == "module3_3_engineering_evidence_ready"
    assert status["engineering_evidence_ready"] is True
    assert status["module3_3_engineering_evidence_ready"] is True
    assert status["components"]["module_3_3_overlap_and_deduplication"] is True


def test_overlap_flag_without_evidence_fails_closed():
    status = ProductionModule3StatusService(
        environment={"MODULE3_OVERLAP_DEDUPLICATION_ENABLED": "true"}
    ).status()
    assert status["status"] == "unsafe_configuration_overlap_evidence_missing"


def test_module3_milestone_booleans_require_upstream_evidence(tmp_path):
    lookalike_path = tmp_path / "lookalike-report.json"
    lookalike_path.write_text(
        json.dumps(
            {
                "status": "engineering_preview_ready",
                "safety": {
                    "raw_identifiers_returned": False,
                    "audience_membership_read": False,
                    "membership_similarity_computed": False,
                    "audience_membership_generated": False,
                    "overlap_rate_computed": False,
                    "unique_reach_claimed": False,
                    "cohort_sizes_summed": False,
                    "candidate_lifecycle_mutated": False,
                    "activation_or_export_performed": False,
                    "downstream_export_enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )

    status = ProductionModule3StatusService(
        environment={
            "MODULE3_LOOKALIKE_EVIDENCE_PATH": str(lookalike_path),
        }
    ).status()

    assert status["status"] == "module3_1_2_evidence_pending"
    assert status["module3_3_engineering_evidence_ready"] is False
    assert status["module3_4_engineering_evidence_ready"] is False


def test_module3_overlap_migration_enforces_review_only_evidence():
    sql = Path("migrations/0013_module3_overlap_deduplication.sql").read_text(
        encoding="utf-8"
    )
    for value in (
        "membership_intersection_read = FALSE",
        "overlap_rate_computed = FALSE",
        "unique_reach_claimed = FALSE",
        "cohort_sizes_summed = FALSE",
        "candidate_lifecycle_mutated = FALSE",
        "lookalike_generation_performed = FALSE",
        "activation_or_export_performed = FALSE",
        "overlap_estimate_available = FALSE",
        "eligible_for_activation = FALSE",
        "eligible_for_export = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "prevent_module3_overlap_evidence_mutation",
        "validate_module3_overlap_member_candidate",
        "validate_module3_duplicate_suppression_candidate",
        "REVOKE ALL ON audience_cohort_overlap_analysis_runs FROM PUBLIC",
    ):
        assert value in sql


def test_module3_overlap_safe_defaults_are_disabled():
    env_text = Path(".env.example").read_text(encoding="utf-8")
    assert "MODULE3_OVERLAP_EVIDENCE_PATH=" in env_text
    assert "MODULE3_OVERLAP_DEDUPLICATION_ENABLED=false" in env_text
