from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.phase2_proposal_runtime_role_service import (
    Phase2ProposalRuntimeRoleService,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Provision and verify the least-privilege Punk AI proposal "
            "runtime role."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--feature-set-version", required=True, type=int)
    parser.add_argument(
        "--confirm-punk-owned-target",
        action="store_true",
    )
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    args = _parser().parse_args()
    result = Phase2ProposalRuntimeRoleService().provision(
        tenant_id=args.tenant_id,
        feature_set_id=args.feature_set_id,
        feature_set_version=args.feature_set_version,
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
