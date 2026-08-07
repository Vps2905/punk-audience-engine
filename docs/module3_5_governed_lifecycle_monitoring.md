# Module 3.5 governed lifecycle monitoring

Module 3.5 produces immutable, tenant-scoped review evidence for cohort
candidate lifecycle decisions. It does not change a candidate's lifecycle,
approve shadow traffic, activate an audience, or export data.

## Inputs

The lifecycle service accepts validated Module 3.3 overlap evidence and its
derived Module 3.4 lookalike evidence. Tenant, purpose, report fingerprints,
and derivation lineage must agree. Evidence containing raw identifiers or
audience membership is rejected.

An optional prior Module 3.5 report can be supplied for per-candidate quality
drift monitoring. It must belong to the same tenant and pass the standalone
report validator.

## Deterministic recommendations

Recommendations are evaluated in fail-closed order:

1. Policy-blocked sensitive POI candidates remain blocked.
2. Potential overlap groups require manual overlap review.
3. Sensitive POI review candidates require manual policy review.
4. Non-production evidence remains historical preview only.
5. Production candidates without permitted rights are blocked.
6. Stale production sources are paused.
7. Candidates below the quality floor, or beyond the allowed quality drop,
   are paused.
8. Remaining candidates may enter `shadow_review_pending`, but still require
   manual approval and monitoring.

Every output explicitly keeps lifecycle mutation, automatic approval, shadow
routing, activation, and export disabled.

## Persistence and tenant isolation

Migration `0018_module3_governed_lifecycle_evidence.sql` stores fingerprinted
evaluations and recommendations. PostgreSQL constraints enforce the safety
flags, candidate fingerprints are checked against immutable Module 3.2
candidates, update/delete triggers make evidence append-only, and forced row
level security scopes every row to `app.tenant_id`.

The migration stores review recommendations only. It is not a candidate state
machine or a release control plane.

## Rollout flags

All flags are disabled by default:

- `MODULE3_LIFECYCLE_EVALUATION_ENABLED=false` gates evidence evaluation.
- `MODULE3_LIFECYCLE_MUTATION_ENABLED=false` must remain false for Module 3.5.
- `MODULE3_PRODUCTION_ROUTING_ENABLED=false` must remain false until the later
  Module 3.6 shadow-integration gate is separately implemented and approved.

Evidence paths used by readiness checks are:

- `MODULE3_COHORT_EVIDENCE_PATH`
- `MODULE3_OVERLAP_EVIDENCE_PATH`
- `MODULE3_LOOKALIKE_EVIDENCE_PATH`
- `MODULE3_LIFECYCLE_EVIDENCE_PATH`

The status service reports Module 3.5 ready only when the complete Module
3.1-3.5 evidence chain is present and every safety assertion remains closed.
Enabling lifecycle evaluation without valid evidence fails closed. Enabling
lifecycle mutation, lookalike generation, or production routing is reported as
an unsafe release-affecting configuration.

## Validation

Run the focused contract suite before broad regression testing:

```bash
REQUIRE_AUDIENCE_API_KEY=true PYTHONPATH=. python -m pytest -q \
  tests/test_production_module3_lifecycle_service.py \
  tests/test_production_module3_lookalike_service.py \
  tests/test_production_module3_overlap_deduplication_service.py \
  tests/test_production_module3_cohort_candidate_service.py \
  tests/test_production_module3_status_and_migration.py \
  tests/test_migration_version_uniqueness.py
```

Do not enable any release-affecting flag from a historical-data test result.
Fresh production-data certification and explicit operator approval remain
separate gates.
