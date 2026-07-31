from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.phase2_feature_role_provisioning_service import (
    Phase2FeatureRoleProvisioningService,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Provision and verify separate least-privilege Phase 2 reader "
            "and writer roles."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--expected-feature-count", required=True, type=int)
    parser.add_argument(
        "--confirm-punk-owned-target",
        action="store_true",
    )
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    args = _parser().parse_args()
    result = Phase2FeatureRoleProvisioningService().provision(
        tenant_id=args.tenant_id,
        expected_feature_count=args.expected_feature_count,
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
