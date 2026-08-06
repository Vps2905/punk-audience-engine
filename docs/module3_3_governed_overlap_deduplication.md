# Module 3.3: Governed Overlap Risk and Exact Duplicate Suppression

Module 3.3 consumes the privacy-safe candidate report produced by Module 3.1-3.2.
It removes repeated copies of the same immutable candidate record and produces
review-only potential-overlap groups from aggregate candidate metadata.

## Safety boundary

Module 3.3 does not receive or infer audience membership. Therefore it does not:

- compute pairwise or percentage overlap,
- claim unique or deduplicated reach,
- sum or merge cohort sizes,
- suppress candidates merely because their aggregate metadata is similar,
- mutate candidate lifecycle state,
- generate lookalikes,
- route to Punk AI production flows,
- activate or export an audience.

Exact duplicate suppression is allowed only when repeated input records have the
same candidate fingerprint and identical canonical payload. A shared candidate
identity with a different fingerprint, or a shared fingerprint with a different
payload, fails closed.

## Review evidence

Two potential-overlap group types are emitted:

- `potential_constraint_overlap`: different retained candidates have the same
  normalized location, POI category, daypart, and lookback bucket.
- `potential_temporal_overlap`: different retained candidates have the same
  normalized location, POI category, and daypart but different lookback buckets.

These groups are review evidence only. `overlap_estimate_available` and
`unique_reach_claimed` are always false.

## Determinism and lineage

- Input candidates are validated against the source batch fingerprint and tenant.
- Candidate identity conflicts fail closed.
- Exact duplicate suppression and overlap-group IDs are content addressed.
- Input ordering does not change the report or report fingerprint.
- All candidates remain approval gated and ineligible for activation/export.

## Persistence schema

Migration `0013_module3_overlap_deduplication.sql` adds immutable, tenant-scoped
control-plane tables for:

- analysis runs,
- potential-overlap groups,
- overlap-group members,
- exact duplicate suppression summaries.

The schema enforces RLS, public-access revocation, immutable evidence, and false
checks for membership-intersection reads, overlap-rate computation, unique-reach
claims, cohort-size summation, lookalikes, activation, and export.

## Evaluation

The no-write evaluator reads a Module 3.1-3.2 JSON candidate report:

```bash
PYTHONPATH=. python scripts/evaluate_module3_overlap_deduplication.py \
  --candidate-report /path/to/module3-candidates.json \
  --output /path/to/module3-overlap-evidence.json \
  --confirm-engineering-preview-only
```

The evaluator refuses to overwrite evidence or use the same path for input and
output. Evidence should stay outside the repository.

## Remaining Module 3 work

- 3.4 governed lookalike generation
- 3.5 lifecycle approval and monitoring
- 3.6 Punk AI shadow integration
- fresh-data and production-scale validation
