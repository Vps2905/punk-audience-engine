from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from app.models.provider_historical_replay_contracts import (
    HistoricalReplayConfig,
)
from app.services.provider_historical_postgres_replay_service import (
    ProviderHistoricalPostgresReplayService,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only, non-activatable privacy replay against the "
            "legacy Postgres snapshot."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--provider-id", default="historical_echo")
    parser.add_argument("--dataset-id", default="maid_extractions")
    parser.add_argument("--min-cohort-size", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--statement-timeout-ms", type=int, default=900_000)
    parser.add_argument(
        "--data-use-mode",
        choices=("historical_preview", "offline_evaluation"),
        default="offline_evaluation",
    )
    parser.add_argument(
        "--privacy-policy-version",
        default="historical-replay-policy-v2",
    )
    parser.add_argument(
        "--confirm-restricted-identifier-processing",
        action="store_true",
        help=(
            "Confirm that raw identifiers may be processed only inside the "
            "read-only source transaction and may never be returned."
        ),
    )
    args = parser.parse_args()

    load_dotenv(".env", override=False)
    result = ProviderHistoricalPostgresReplayService().replay(
        HistoricalReplayConfig(
            tenant_id=args.tenant_id,
            provider_id=args.provider_id,
            dataset_id=args.dataset_id,
            min_cohort_size=args.min_cohort_size,
            epsilon=args.epsilon,
            delta=args.delta,
            statement_timeout_ms=args.statement_timeout_ms,
            data_use_mode=args.data_use_mode,
            privacy_policy_version=args.privacy_policy_version,
        ),
        confirm_restricted_identifier_processing=(
            args.confirm_restricted_identifier_processing
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
