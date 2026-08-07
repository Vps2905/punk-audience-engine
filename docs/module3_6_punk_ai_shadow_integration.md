# Module 3.6 Punk AI shadow integration

Module 3.6 closes the engineering boundary between the existing Punk AI
audience-proposal contract and governed Module 3 cohort intelligence. It
observes proposal results; it does not create proposals or become a routing
path.

## Shadow comparison

The observation service accepts:

- a privacy-safe Punk AI proposal response already produced by the existing
  proposal service;
- its tenant and proposal identity;
- validated Module 3.3 overlap evidence; and
- the derived, validated Module 3.5 lifecycle evidence.

Each proposed `feature_id` is matched to the immutable Module 3 candidate
lineage and lifecycle recommendation. The result is one of:

- `aligned_historical_preview`;
- `aligned_manual_shadow_review`;
- `blocked_lifecycle`;
- `unmatched_candidate`; or
- `ambiguous_candidate`.

Production observations align only when lifecycle evidence recommends
`shadow_review_pending`. This still means manual review is pending. It never
means that shadow or production traffic was routed.

## Fail-closed boundaries

The service rejects tenant, proposal identity, execution-mode, overlap, and
lifecycle fingerprint mismatches. It also rejects proposals that enable
downstream export, omit manual approval, expose raw identifiers, or contain
invalid candidate structures.

All generated evidence asserts that:

- the proposal was not created or modified by Module 3.6;
- candidate lifecycle was not mutated;
- automatic approval was not performed;
- shadow and production routing stayed disabled;
- activation and export were not performed; and
- audience membership was not read.

## Persistence

Migration `0019_module3_punk_ai_shadow_observations.sql` stores immutable
tenant-scoped observation runs and per-candidate comparisons. It verifies
candidate fingerprints, uses append-only triggers, forces PostgreSQL row-level
security through `app.tenant_id`, and revokes public access.

The observation run references the existing immutable Punk AI proposal ledger,
Module 3.3 overlap evidence, and Module 3.5 lifecycle evaluation. It does not
add an alternative proposal table or control plane.

## Configuration

All capabilities remain disabled by default:

- `MODULE3_SHADOW_EVIDENCE_PATH=` supplies reviewed evidence to readiness.
- `MODULE3_SHADOW_OBSERVATION_ENABLED=false` gates observation processing.
- `MODULE3_PRODUCTION_ROUTING_ENABLED=false` remains a release-affecting hard
  block and is not enabled by Module 3.6.

The Module 3 status service declares 3.6 engineering evidence ready only when
the complete 3.1-3.6 chain is valid, at least one proposal candidate was
observed, every candidate aligned, and all safety assertions remain closed.

## Validation

```bash
REQUIRE_AUDIENCE_API_KEY=true PYTHONPATH=. python -m pytest -q \
  tests/test_production_module3_shadow_service.py \
  tests/test_production_module3_lifecycle_service.py \
  tests/test_production_module3_lookalike_service.py \
  tests/test_production_module3_overlap_deduplication_service.py \
  tests/test_production_module3_cohort_candidate_service.py \
  tests/test_production_module3_status_and_migration.py \
  tests/test_punk_ai_audience_proposal_api.py \
  tests/test_audience_feature_proposal_service.py \
  tests/test_migration_version_uniqueness.py
```

Historical evidence completes the engineering contract only. Fresh-data
staging validation and an explicit production release review remain separate
certification gates.
