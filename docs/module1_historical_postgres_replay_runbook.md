# Module 1 historical Postgres replay

This adapter validates the privacy transformation against the authorized
historical `public.maid_extractions` snapshot without modifying the source.

It expands MAIDs only inside a read-only PostgreSQL transaction, deduplicates
the latest extraction per session, bounds each identifier to one contribution
per location/category across the snapshot, enforces k >= 1000, blocks sensitive
POI cohorts, and returns only Gaussian-DP counts. Identifier values, raw
observations, and coordinates never cross the database boundary.

Generic provider taxonomy labels such as `establishment`,
`point_of_interest`, `premise`, and `business` are excluded when selecting the
primary POI category. If no specific category remains, the batch fails closed
instead of becoming a misleading generic audience.

The historical table does not contain trustworthy event-time/daypart linkage.
The adapter therefore reports daypart as `unknown`; it does not substitute the
extraction timestamp as behavioural time.

This path is for offline evaluation only. It does not reserve durable privacy
budget and does not exercise S3 object validation, SQS/DLQ, distributed entity
partitions, or canonical activation. Its output is always stale and blocked
from activation/export.

The DP random stream is derived with HMAC from an immutable source fingerprint,
the privacy-policy version, and `AUDIENCE_DP_SEED_SECRET`. Replaying the same
release therefore produces the same private counts and cannot be averaged to
reduce the intended noise. Changing policy version produces a new fingerprint.

Run:

```bash
PYTHONPATH=. python scripts/run_historical_provider_privacy_replay.py \
  --tenant-id punk_internal \
  --min-cohort-size 1000 \
  --data-use-mode offline_evaluation \
  --confirm-restricted-identifier-processing
```

The default statement timeout is 15 minutes. Increase it only after confirming
that the historical source database can safely support the JSON expansion.
