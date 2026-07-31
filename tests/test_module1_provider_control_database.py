from pathlib import Path

import pytest

from app.api.audience_intelligence_provider_ingestion import _database_url
from app.services.module1_provider_database_preflight_service import (
    Module1ProviderDatabasePreflightService,
)
from app.services.module1_provider_runtime_role_service import (
    Module1ProviderRuntimeRoleService,
)
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from scripts.register_provider_contract import _database_url as script_db_url


ROOT = Path(__file__).resolve().parents[1]


def test_provider_services_never_fall_back_to_historical_database(monkeypatch):
    monkeypatch.delenv("PROVIDER_INGESTION_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "ECHO_DATABASE_URL",
        "postgresql://source.example/historical",
    )
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://source.example/historical",
    )

    assert _database_url() is None
    assert ProviderIngestionStateService()._db_url() is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="Punk-owned"):
        ProviderContractRegistryService()._engine()  # noqa: SLF001
    with pytest.raises(RuntimeError, match="Punk-owned"):
        script_db_url()


def test_preflight_blocks_historical_database_as_provider_target():
    service = Module1ProviderDatabasePreflightService(environment={})
    same = "postgresql://user:secret@db.example:5432/historical"
    report = service.run(
        target_database_url=same,
        historical_database_url=(
            "postgresql://different:password@db.example:5432/historical"
        ),
        punk_owned_target_confirmed=True,
    )

    assert report["status"] == "blocked"
    assert report["same_database_as_historical_source"] is True
    assert "provider_target_matches_historical_source" in report["blockers"]
    assert report["read_only"] is True


def test_control_plane_migration_uses_role_authenticated_tenant_rls():
    sql = (
        ROOT / "migrations" / "0010_provider_control_plane_rls.sql"
    ).read_text(encoding="utf-8").lower()

    assert "provider_runtime_tenants" in sql
    assert "security definer" in sql
    assert "session_user" in sql
    assert "force row level security" in sql
    assert "current_setting('app.tenant_id'" not in sql
    for policy in Module1ProviderDatabasePreflightService.REQUIRED_POLICIES:
        assert policy in sql


def test_local_control_database_compose_is_isolated_and_pinned():
    compose = (
        ROOT / "docker-compose.module1-control.yml"
    ).read_text(encoding="utf-8")

    assert "postgres:18.4" in compose
    assert "127.0.0.1:${MODULE1_CONTROL_PORT:-55433}:5432" in compose
    assert "MODULE1_CONTROL_PASSWORD:?" in compose
    assert "punk-module1-control-data" in compose
    assert "punk-module1-control-data:/var/lib/postgresql" in compose
    assert "/var/lib/postgresql/data" not in compose


def test_runtime_role_requires_strong_secret_before_connecting():
    service = Module1ProviderRuntimeRoleService(
        environment={
            "MODULE1_PROVIDER_RUNTIME_USER": "punk_provider_runtime",
            "MODULE1_PROVIDER_RUNTIME_PASSWORD": "short",
            "PROVIDER_INGESTION_MIGRATION_DATABASE_URL": (
                "postgresql://admin:secret@127.0.0.1/provider"
            ),
        }
    )

    with pytest.raises(ValueError, match="at least 32"):
        service.provision(
            tenant_id="punk_internal",
            punk_owned_target_confirmed=True,
        )


def test_generic_migration_runner_excludes_provider_control_migrations():
    source = (ROOT / "scripts" / "apply_database_migrations.py").read_text(
        encoding="utf-8"
    )

    for migration in (
        "0003_provider_ingestion_gateway.sql",
        "0006_provider_distributed_scale_dispatch.sql",
        "0007_provider_distributed_privacy_releases.sql",
        "0009_provider_privacy_windows_and_rights.sql",
        "0010_provider_control_plane_rls.sql",
    ):
        assert migration in source


def test_provider_postgres_runtime_services_skip_schema_bootstrap():
    service_paths = (
        "app/services/provider_contract_registry_service.py",
        "app/services/provider_ingestion_state_service.py",
        "app/services/provider_distributed_privacy_budget_service.py",
        "app/services/provider_privacy_window_service.py",
        "app/services/provider_data_rights_service.py",
        "app/services/provider_scale_acceptance_service.py",
    )

    for service_path in service_paths:
        source = (ROOT / service_path).read_text(encoding="utf-8")
        assert 'if dialect == "postgresql":\n            return' in source or (
            'if dialect_name == "postgresql":\n            return' in source
        )

def test_module1_migrations_avoid_reserved_window_alias():
    import re

    migration_names = (
        "0009_provider_privacy_windows_and_rights.sql",
        "0010_provider_control_plane_rls.sql",
    )

    for migration_name in migration_names:
        sql = (
            ROOT / "migrations" / migration_name
        ).read_text(encoding="utf-8")

        unsafe_from_alias = re.search(
            r"\bFROM\s+provider_privacy_windows\s+window\b",
            sql,
            flags=re.IGNORECASE,
        )
        unsafe_column_alias = re.search(
            r"(?<![A-Za-z0-9_])window\.",
            sql,
            flags=re.IGNORECASE,
        )

        assert unsafe_from_alias is None
        assert unsafe_column_alias is None
        assert (
            "FROM provider_privacy_windows AS privacy_window"
            in sql
        )

