from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.models.production_preproduction_deployment_contracts import (
    PreproductionDeploymentCertificationRequest,
)
from app.services.production_preproduction_deployment_certification_service import (
    ProductionPreproductionDeploymentCertificationService,
)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} must be a readable JSON file.") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must contain a JSON object.")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sanitized, measured preproduction deployment evidence. "
            "This command does not call AWS, read audience data, or mutate "
            "cloud resources."
        )
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    request = PreproductionDeploymentCertificationRequest(
        **_json_object(args.request, label="request")
    )
    observations = _json_object(args.observations, label="observations")
    report = ProductionPreproductionDeploymentCertificationService().certify(
        request=request,
        observations=observations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
