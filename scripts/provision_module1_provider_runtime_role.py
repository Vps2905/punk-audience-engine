from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.services.module1_provider_runtime_role_service import (
    Module1ProviderRuntimeRoleService,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision a tenant-pinned Module 1 runtime role."
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--confirm-punk-owned-target", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    result = Module1ProviderRuntimeRoleService().provision(
        tenant_id=args.tenant_id,
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
