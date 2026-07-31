from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only preflight for the historical source and Punk-owned "
            "Phase 2 pgvector feature database."
        )
    )
    parser.add_argument(
        "--confirm-punk-owned-target",
        action="store_true",
        help=(
            "Assert that AUDIENCE_FEATURE_DATABASE_URL points to a "
            "Punk-owned operational database. This flag performs no writes."
        ),
    )
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    args = _parser().parse_args()
    report = Phase2DatabasePreflightService().run(
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
