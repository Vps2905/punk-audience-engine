from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from app.models.provider_scale_acceptance_contracts import (
    ProviderScaleAcceptanceEvidence,
)
from app.services.provider_scale_acceptance_service import (
    ProviderScaleAcceptanceService,
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scale evidence file must contain one JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate measured provider-ingestion scale evidence. This command "
            "does not run activation or export."
        )
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tenant-id")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--confirm-measured-preproduction-evidence",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_measured_preproduction_evidence:
        raise SystemExit(
            "STOP: confirm that the evidence came from a measured staging or "
            "preproduction run"
        )
    evidence = ProviderScaleAcceptanceEvidence.from_dict(
        _read_json(args.evidence)
    )
    service = ProviderScaleAcceptanceService()
    if args.record:
        if not args.tenant_id:
            raise SystemExit("STOP: --tenant-id is required with --record")
        report = service.record(
            tenant_id=args.tenant_id,
            evidence=evidence,
        )
    else:
        report = service.evaluate(evidence)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
