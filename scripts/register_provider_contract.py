from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.models.provider_ingestion_contracts import ProviderDatasetContract
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)


def _database_url() -> str:
    value = (
        os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        or ""
    ).strip()
    if not value:
        raise RuntimeError(
            "The explicit Punk-owned PROVIDER_INGESTION_DATABASE_URL is "
            "not configured."
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register or deactivate a non-secret provider contract."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--contract", type=Path)
    action.add_argument("--deactivate-key")
    parser.add_argument("--actor", required=True)
    args = parser.parse_args()

    registry = ProviderContractRegistryService(
        database_url=_database_url()
    )
    if args.contract is not None:
        payload = json.loads(args.contract.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Provider contract file must contain a JSON object.")
        result = registry.register(
            ProviderDatasetContract(**payload),
            actor=args.actor,
        )
        safe_result = {
            "created": result["created"],
            "contract_key": result["contract"]["contract_key"],
            "contract_checksum": result["contract"]["contract_checksum"],
            "active": result["contract"]["active"],
        }
    else:
        result = registry.deactivate(
            str(args.deactivate_key),
            actor=args.actor,
        )
        safe_result = {
            "contract_key": result["contract_key"],
            "active": result["active"],
        }
    print(json.dumps(safe_result, sort_keys=True))


if __name__ == "__main__":
    main()
