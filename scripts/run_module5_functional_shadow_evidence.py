from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import dotenv_values

from app.core.atomic_evidence_writer import write_new_private_json
from app.services.production_module5_evidence_orchestration_service import (
    ProductionModule5FunctionalEvidenceOrchestrator,
)


def _read_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _environment(path: Path | None) -> dict[str, str]:
    import os

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
            "Generate one real, read-only Module 5.9 functional-shadow report "
            "with validated Module 5.8 lineage."
        )
    )
    parser.add_argument("--source-certification-report", type=Path, required=True)
    parser.add_argument("--functional-input", type=Path, required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--legacy-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional local input only; secret values are never written to evidence.",
    )
    args = parser.parse_args()

    report = ProductionModule5FunctionalEvidenceOrchestrator(
        environment=_environment(args.env_file)
    ).run(
        source_certification_report=_read_mapping(
            args.source_certification_report
        ),
        functional_input=_read_mapping(args.functional_input),
        taxonomy_payload=_read_mapping(args.taxonomy_file),
        legacy_result=_read_mapping(args.legacy_result),
    )
    write_new_private_json(args.output, report)
    ready = report.get("status") == "engineering_preview_ready"
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "eligible_for_staging_review": report.get("review", {}).get(
                    "eligible_for_staging_review"
                ),
                "functional_shadow_report_fingerprint": report.get(
                    "functional_shadow_report_fingerprint"
                ),
                "production_ready": False,
                "live_cutover_authorized": False,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
