from pathlib import Path

import pytest

from app.models.provider_ingestion_contracts import ProviderDatasetContract
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)


def _contract(**overrides):
    values = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "dataset_a",
        "schema_version": "v1",
        "data_format": "csv",
        "allowed_bucket": "provider-landing",
        "allowed_prefix": "delivery/",
        "entity_id_column": "signal_id",
        "timestamp_column": "event_time",
        "cohort_columns": ("region", "affinity", "time_window"),
    }
    values.update(overrides)
    return ProviderDatasetContract(**values)


def test_registry_round_trips_active_non_secret_contract(tmp_path: Path):
    service = ProviderContractRegistryService(
        database_url=f"sqlite:///{tmp_path / 'registry.db'}"
    )
    contract = _contract()

    registered = service.register(contract, actor="test-operator")
    loaded = service.get_active(
        tenant_id=contract.tenant_id,
        provider_id=contract.provider_id,
        dataset_id=contract.dataset_id,
        schema_version=contract.schema_version,
    )

    assert registered["created"] is True
    assert loaded == contract
    assert "credentials" not in registered["contract"]["contract"]


def test_registry_contract_version_is_immutable(tmp_path: Path):
    service = ProviderContractRegistryService(
        database_url=f"sqlite:///{tmp_path / 'registry.db'}"
    )
    service.register(_contract(), actor="test-operator")

    with pytest.raises(ValueError, match="different immutable"):
        service.register(
            _contract(min_cohort_size=2000),
            actor="test-operator",
        )


def test_registry_deactivation_fails_closed_for_new_ingestion(tmp_path: Path):
    service = ProviderContractRegistryService(
        database_url=f"sqlite:///{tmp_path / 'registry.db'}"
    )
    contract = _contract()
    service.register(contract, actor="test-operator")
    service.deactivate(contract.contract_key, actor="test-operator")

    with pytest.raises(KeyError, match="No active"):
        service.get_active(
            tenant_id=contract.tenant_id,
            provider_id=contract.provider_id,
            dataset_id=contract.dataset_id,
            schema_version=contract.schema_version,
        )
