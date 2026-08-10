from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import dotenv_values

from app.core.atomic_evidence_writer import write_new_private_json
from app.services.production_module5_evidence_orchestration_service import (
    ProductionModule5AgentSecurityEvidenceOrchestrator,
)


def _read_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


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
            "Assess the actual security environment and generate Module 5.10 "
            "evidence only when every prerequisite passes."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--assessment-id", required=True)
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--functional-shadow-report", type=Path, required=True)
    parser.add_argument("--posture-output", type=Path, required=True)
    parser.add_argument("--certification-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--evaluation-epoch-seconds", type=int)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional local input only; secret values are never written to evidence.",
    )
    args = parser.parse_args()

    result = ProductionModule5AgentSecurityEvidenceOrchestrator(
        environment=_environment(args.env_file)
    ).run(
        tenant_id=args.tenant_id,
        assessment_id=args.assessment_id,
        certification_id=args.certification_id,
        functional_shadow_report=_read_mapping(args.functional_shadow_report),
        evaluation_epoch_seconds=(
            args.evaluation_epoch_seconds or int(time.time())
        ),
    )
    posture = result.pop("security_posture_report")
    certification = result.pop("agent_security_certification_report")
    summary = {
        **result,
        "posture_output": str(args.posture_output),
        "certification_output": (
            str(args.certification_output) if certification is not None else None
        ),
    }
    write_new_private_json(args.posture_output, posture)
    if certification is not None:
        write_new_private_json(args.certification_output, certification)
        summary["agent_security_certification_fingerprint"] = certification[
            "agent_security_certification_fingerprint"
        ]
    write_new_private_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if certification is not None and result["status"] == (
        "engineering_preview_ready"
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
