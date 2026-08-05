#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.services.production_module2_certification_service import (
    ProductionModule2CertificationService,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a privacy-safe native-language Module 2 review plan."
    )
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--language-pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-no-auto-approval", action="store_true")
    args = parser.parse_args()
    if not args.confirm_no_auto_approval:
        raise SystemExit("--confirm-no-auto-approval is required.")
    plan = ProductionModule2CertificationService().plan_native_language_review(
        taxonomy_payload=_load(args.taxonomy),
        language_pack_payload=_load(args.language_pack),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "native_language_review_plan_created",
                "output": str(args.output),
                "output_sha256": hashlib.sha256(
                    args.output.read_bytes()
                ).hexdigest(),
                "review_status": "pending",
                "automatic_approval_performed": False,
                "production_routing_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
