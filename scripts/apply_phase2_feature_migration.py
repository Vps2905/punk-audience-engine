from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.phase2_feature_migration_service import (
    Phase2FeatureMigrationService,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply only migration 0004 to an explicitly configured, "
            "Punk-owned Phase 2 feature database."
        )
    )
    parser.add_argument(
        "--confirm-punk-owned-target",
        action="store_true",
        help=(
            "Confirm that AUDIENCE_FEATURE_MIGRATION_DATABASE_URL is the "
            "approved Punk-owned migration target."
        ),
    )
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    args = _parser().parse_args()
    result = Phase2FeatureMigrationService().apply(
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
