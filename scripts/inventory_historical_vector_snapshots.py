from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.historical_vector_snapshot_inventory_service import (
    HistoricalVectorSnapshotInventoryService,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List aggregate validation information for existing "
            "privacy-safe historical vector snapshots."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum model snapshots to inspect (1-1000).",
    )
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    report = HistoricalVectorSnapshotInventoryService().inventory(
        limit=_parser().parse_args().limit
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
