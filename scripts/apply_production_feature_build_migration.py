from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.production_feature_build_migration_service import (
    ProductionFeatureBuildMigrationService,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply only the controlled production feature-build registry "
            "migration to the Punk-owned feature database."
        )
    )
    parser.add_argument(
        "--confirm-punk-owned-target",
        action="store_true",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    result = ProductionFeatureBuildMigrationService().apply(
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
