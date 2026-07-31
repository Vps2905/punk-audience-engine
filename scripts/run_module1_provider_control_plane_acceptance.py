from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from app.services.module1_provider_control_plane_acceptance_service import (
    Module1ProviderControlPlaneAcceptanceService,
)


def main() -> None:
    load_dotenv(".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Exercise Module 1 distributed privacy and publication controls "
            "with identifier-free mock object references."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument(
        "--confirm-offline-control-plane-only",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_offline_control_plane_only:
        raise SystemExit(
            "STOP: pass --confirm-offline-control-plane-only; this check does "
            "not certify AWS infrastructure or perform activation."
        )
    database_url = os.getenv("PROVIDER_INGESTION_DATABASE_URL")
    if not database_url:
        raise SystemExit("STOP: PROVIDER_INGESTION_DATABASE_URL is missing")
    report = Module1ProviderControlPlaneAcceptanceService(
        database_url=database_url
    ).run(tenant_id=args.tenant_id)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
