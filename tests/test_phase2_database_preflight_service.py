from __future__ import annotations

from copy import deepcopy

from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)


def _database_report(
    *,
    role: str,
    identity: str,
    legacy: bool = False,
    phase2: bool = False,
) -> dict:
    return {
        "role": role,
        "configured": True,
        "connection_ok": True,
        "read_only_transaction_verified": True,
        "postgresql": True,
        "server_version_num": 160000,
        "pgvector_available": True,
        "pgvector_installed": True,
        "pgvector_version": "0.8.0",
        "current_role_superuser": False,
        "database_create_privilege": False,
        "public_schema_create_privilege": True,
        "legacy_vector_models_table_present": legacy,
        "legacy_vectors_table_present": legacy,
        "legacy_snapshot_present": legacy,
        "legacy_vector_count": 86 if legacy else 0,
        "legacy_vector_dimension": 384 if legacy else None,
        "latest_source_timestamp": (
            "2026-07-08T08:40:40+00:00" if legacy else None
        ),
        "source_events_table_present": legacy,
        "feature_sets_table_present": phase2,
        "feature_vectors_table_present": phase2,
        "feature_embedding_type": "vector(384)" if phase2 else None,
        "feature_sets_rls_forced": phase2,
        "feature_vectors_rls_forced": phase2,
        "feature_sets_tenant_policy_present": phase2,
        "feature_vectors_tenant_policy_present": phase2,
        "feature_set_lookup_index_present": phase2,
        "feature_hard_filter_index_present": phase2,
        "feature_lexical_index_present": phase2,
        "feature_hnsw_index_present": phase2,
        "phase2_migration_recorded": phase2,
        "error_code": None,
        "_database_identity": identity,
    }


class StubPreflightService(Phase2DatabasePreflightService):
    def __init__(self, reports: dict[str, dict], environment: dict[str, str]):
        super().__init__(environment=environment)
        self._reports = reports

    def _inspect_configured_database(
        self,
        database_url: str,
        *,
        role: str,
    ) -> dict:
        if not database_url:
            return {
                "role": role,
                "configured": False,
                "connection_ok": False,
                "error_code": "database_not_configured",
            }
        return deepcopy(self._reports[role])


def test_feature_target_never_falls_back_to_historical_source():
    reports = {
        "historical_source": _database_report(
            role="historical_source",
            identity="source",
            legacy=True,
        ),
    }
    service = StubPreflightService(
        reports,
        environment={"ECHO_DATABASE_URL": "postgresql://source"},
    )

    result = service.run(punk_owned_target_confirmed=True)

    assert result["status"] == "blocked"
    assert (
        "explicit_feature_database_not_configured"
        in result["blockers"]
    )
    assert result["feature_target"]["configured"] is False
    assert result["migration_attempted"] is False
    assert result["indexing_attempted"] is False


def test_ready_status_requires_explicit_target_and_ownership_confirmation():
    reports = {
        "historical_source": _database_report(
            role="historical_source",
            identity="source",
            legacy=True,
        ),
        "feature_target": _database_report(
            role="feature_target",
            identity="target",
        ),
    }
    environment = {
        "ECHO_DATABASE_URL": "postgresql://source",
        "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://target",
    }
    service = StubPreflightService(reports, environment)

    unconfirmed = service.run()
    confirmed = service.run(punk_owned_target_confirmed=True)

    assert unconfirmed["status"] == "blocked"
    assert (
        "punk_owned_feature_database_not_confirmed"
        in unconfirmed["blockers"]
    )
    assert confirmed["status"] == "ready_for_approved_migration"
    assert confirmed["blockers"] == []
    assert confirmed["same_database_as_source"] is False


def test_existing_phase2_schema_is_verified_without_migration():
    reports = {
        "historical_source": _database_report(
            role="historical_source",
            identity="source",
            legacy=True,
        ),
        "feature_target": _database_report(
            role="feature_target",
            identity="target",
            phase2=True,
        ),
    }
    service = StubPreflightService(
        reports,
        environment={
            "ECHO_DATABASE_URL": "postgresql://source",
            "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://target",
        },
    )

    result = service.run(punk_owned_target_confirmed=True)

    assert result["status"] == "phase2_schema_ready"
    assert result["phase2_schema_ready"] is True
    assert result["next_action"] == (
        "index_historical_snapshot_after_operator_approval"
    )
    assert result["migration_attempted"] is False


def test_recorded_but_incomplete_phase2_schema_is_blocked():
    target = _database_report(
        role="feature_target",
        identity="target",
        phase2=True,
    )
    target["feature_hnsw_index_present"] = False
    service = StubPreflightService(
        {
            "historical_source": _database_report(
                role="historical_source",
                identity="source",
                legacy=True,
            ),
            "feature_target": target,
        },
        environment={
            "ECHO_DATABASE_URL": "postgresql://source",
            "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://target",
        },
    )

    result = service.run(punk_owned_target_confirmed=True)

    assert result["status"] == "blocked"
    assert "phase2_recorded_schema_incomplete" in result["blockers"]
    assert result["phase2_schema_ready"] is False
    assert result["next_action"] == "repair_phase2_schema_before_use"


def test_same_database_is_reported_as_warning_not_hidden():
    reports = {
        "historical_source": _database_report(
            role="historical_source",
            identity="same-db",
            legacy=True,
        ),
        "feature_target": _database_report(
            role="feature_target",
            identity="same-db",
        ),
    }
    service = StubPreflightService(
        reports,
        environment={
            "ECHO_DATABASE_URL": "postgresql://source-role",
            "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://feature-role",
        },
    )

    result = service.run(punk_owned_target_confirmed=True)

    assert result["status"] == "ready_for_approved_migration"
    assert result["same_database_as_source"] is True
    assert result["warnings"] == [
        "source_and_feature_target_are_the_same_database"
    ]


def test_missing_pgvector_install_is_fail_closed_for_non_admin_role():
    source = _database_report(
        role="historical_source",
        identity="source",
        legacy=True,
    )
    target = _database_report(
        role="feature_target",
        identity="target",
    )
    target["pgvector_installed"] = False
    target["pgvector_version"] = None
    target["current_role_superuser"] = False
    service = StubPreflightService(
        {
            "historical_source": source,
            "feature_target": target,
        },
        environment={
            "ECHO_DATABASE_URL": "postgresql://source",
            "AUDIENCE_FEATURE_DATABASE_URL": "postgresql://target",
        },
    )

    result = service.run(punk_owned_target_confirmed=True)

    assert result["status"] == "blocked"
    assert (
        "pgvector_requires_database_admin_installation"
        in result["blockers"]
    )
    assert result["next_action"] == (
        "ask_database_admin_to_install_pgvector"
    )


def test_public_report_never_contains_database_identity_or_url():
    reports = {
        "historical_source": _database_report(
            role="historical_source",
            identity="secret-source-identity",
            legacy=True,
        ),
        "feature_target": _database_report(
            role="feature_target",
            identity="secret-target-identity",
        ),
    }
    service = StubPreflightService(
        reports,
        environment={
            "ECHO_DATABASE_URL": "configured-source-value",
            "AUDIENCE_FEATURE_DATABASE_URL": "configured-target-value",
        },
    )

    result = service.run(punk_owned_target_confirmed=True)
    rendered = str(result)

    assert "_database_identity" not in rendered
    assert "configured-source-value" not in rendered
    assert "configured-target-value" not in rendered
    assert result["credentials_exposed"] is False
