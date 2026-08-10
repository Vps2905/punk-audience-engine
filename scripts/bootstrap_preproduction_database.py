from __future__ import annotations

import json

from app.services.preproduction_database_role_provisioning_service import (
    PreproductionDatabaseRoleProvisioningService,
)
from scripts.apply_database_migrations import apply_migrations


def main() -> None:
    apply_migrations()
    result = PreproductionDatabaseRoleProvisioningService().provision()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
