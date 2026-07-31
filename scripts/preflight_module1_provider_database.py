from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.module1_provider_database_preflight_service import (
    Module1ProviderDatabasePreflightService,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only preflight of the Module 1 provider database."
    )
    parser.add_argument("--confirm-punk-owned-target", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    report = Module1ProviderDatabasePreflightService().run(
        punk_owned_target_confirmed=args.confirm_punk_owned_target
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
