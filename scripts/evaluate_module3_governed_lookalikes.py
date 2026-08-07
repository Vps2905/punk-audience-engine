from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.production_module3_lookalike_contracts import (
    Module3LookalikeRequest,
)
from app.services.production_module3_lookalike_service import (
    ProductionModule3LookalikeService,
)


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Module 3.3 evidence must contain a JSON object.")
    return payload


def build_request(
    overlap_report: dict[str, Any],
) -> Module3LookalikeRequest:
    source_request = overlap_report.get("request")
    if not isinstance(source_request, dict):
        raise ValueError("Module 3.3 evidence is missing request metadata.")

    return Module3LookalikeRequest(
        tenant_id=str(source_request.get("tenant_id") or ""),
        overlap_report_fingerprint=str(
            overlap_report.get("report_fingerprint") or ""
        ),
        execution_mode=str(source_request.get("execution_mode") or ""),
        purpose=str(source_request.get("purpose") or ""),
    )


def evaluate(
    *,
    input_path: Path,
) -> dict[str, Any]:
    overlap_report = read_json(input_path)
    request = build_request(overlap_report)

    return ProductionModule3LookalikeService().generate(
        request=request,
        overlap_report=overlap_report,
    ).to_record()


def main() -> int:
    load_environment()

    parser = argparse.ArgumentParser(
        description=(
            "Generate no-write Module 3.4 aggregate-only "
            "lookalike review evidence."
        )
    )
    parser.add_argument(
        "--input",
        default=os.getenv("MODULE3_OVERLAP_EVIDENCE_PATH", ""),
        help="Module 3.3 overlap evidence JSON path.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MODULE3_LOOKALIKE_EVIDENCE_PATH", ""),
        help="Optional Module 3.4 output JSON path.",
    )
    args = parser.parse_args()

    input_value = str(args.input or "").strip()
    if not input_value:
        raise SystemExit(
            "MODULE3_OVERLAP_EVIDENCE_PATH or --input is required."
        )

    result = evaluate(input_path=Path(input_value))
    rendered = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )

    output_value = str(args.output or "").strip()
    if output_value:
        output_path = Path(output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
