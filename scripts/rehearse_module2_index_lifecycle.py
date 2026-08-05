#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.models.production_module2_completion_contracts import (
    Module2IndexManifest,
    Module2IndexValidationEvidence,
)
from app.services.production_module2_index_lifecycle_service import (
    JsonModule2IndexLifecycleRepository,
    ProductionModule2IndexLifecycleService,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rehearse governed index lifecycle with an offline ledger."
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--action",
        required=True,
        choices=("register", "building", "built", "validate", "shadow"),
    )
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--certification-report", type=Path)
    parser.add_argument("--actor", default="module2_operator")
    parser.add_argument("--artifact-checksum", default="0" * 64)
    parser.add_argument("--confirm-no-production-routing", action="store_true")
    args = parser.parse_args()
    if not args.confirm_no_production_routing:
        raise SystemExit("--confirm-no-production-routing is required.")
    manifest = Module2IndexManifest(**_load(args.manifest))
    service = ProductionModule2IndexLifecycleService(
        repository=JsonModule2IndexLifecycleRepository(args.ledger)
    )
    if args.action == "register":
        result = service.register_candidate(manifest)
    elif args.action == "building":
        result = service.mark_building(manifest=manifest, worker_id=args.actor)
    elif args.action == "built":
        result = service.mark_built(
            manifest=manifest,
            observed_document_count=manifest.expected_document_count,
            artifact_checksum=args.artifact_checksum,
        )
    elif args.action == "validate":
        if not args.validation:
            raise SystemExit("--validation is required for validate.")
        result = service.record_validation(
            manifest=manifest,
            validation=Module2IndexValidationEvidence(**_load(args.validation)),
        )
    else:
        if not args.certification_report:
            raise SystemExit("--certification-report is required for shadow.")
        result = service.promote_to_shadow(
            manifest=manifest,
            approved_by=args.actor,
            certification_report=_load(args.certification_report),
        )
    result["production_routing_enabled"] = False
    result["activation_or_export_performed"] = False
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
