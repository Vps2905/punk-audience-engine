from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from app.core.atomic_evidence_writer import write_new_private_json
from app.services.production_module5_repeated_shadow_evidence_service import (
    ProductionModule5RepeatedShadowEvidenceService,
    discover_historical_prompt_results,
)


def _environment(path: Path | None) -> dict[str, str]:
    values = dict(dotenv_values(path)) if path is not None else {}
    values.update(os.environ)
    return {
        str(key): str(value)
        for key, value in values.items()
        if value is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate real Module 5.8 repeated-shadow evidence from persisted "
            "privacy-safe historical prompt summaries."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument(
        "--historical-run-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing */final_prompt_summary.json files.",
    )
    parser.add_argument(
        "--legacy-result",
        type=Path,
        action="append",
        default=[],
        help="An explicit final_prompt_summary.json file; repeat as needed.",
    )
    parser.add_argument("--maximum-cases", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional policy input; environment values are not stored.",
    )
    args = parser.parse_args()

    if args.output.expanduser().absolute() == (
        args.inventory_output.expanduser().absolute()
    ):
        raise ValueError("Report and inventory outputs must be different files.")

    discovered = discover_historical_prompt_results(args.historical_run_root)
    paths = tuple(discovered) + tuple(args.legacy_result)
    environment = _environment(args.env_file)
    service = ProductionModule5RepeatedShadowEvidenceService(
        environment=environment
    )
    inventory = service.inventory(
        tenant_id=args.tenant_id,
        result_paths=paths,
        maximum_cases=args.maximum_cases,
    )
    write_new_private_json(args.inventory_output, inventory)
    report = service.run(
        tenant_id=args.tenant_id,
        result_paths=paths,
        maximum_cases=args.maximum_cases,
    )
    write_new_private_json(args.output, report)
    summary = {
        "status": report["status"],
        "inventory": inventory,
        "summary": report["summary"],
        "certification_gates": report["certification_gates"],
        "bounded_autonomy_certification_fingerprint": report[
            "bounded_autonomy_certification_fingerprint"
        ],
        "eligible_for_staging_review": report["review"][
            "eligible_for_staging_review"
        ],
        "production_ready": False,
        "live_cutover_authorized": False,
        "output": str(args.output),
        "inventory_output": str(args.inventory_output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["status"] == "engineering_preview_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
