# Module 5 operator evidence chain

## Purpose

The Module 5.12 historical scale runner consumes real Module 5.9 functional
shadow evidence and real Module 5.10 agent-security evidence. Unit-test
fixtures, copied fingerprints, and hand-authored JSON are not certification
evidence. These operator commands create and validate the missing artifacts
without enabling production routing, activation, export, or automatic
approval.

The chain is deliberately fail closed:

1. Module 5.8 must be a validated, ready repeated-shadow certification.
2. Module 5.9 runs the real rollback-only feature reader, governed taxonomy,
   approved-model registry, pgvector retrieval, cohort services, evolution
   review, and delivery review.
3. Module 5.10 assesses the actual process environment. A failed control writes
   a posture report and blocker summary, but no security certification.
4. Module 5.12 accepts only lineage-matched, ready Module 5.9 and 5.10 reports.
5. Even a passing chain remains staging-review evidence. It never claims
   production readiness or live-cutover authorization.

All evidence files are created atomically with mode `0600`. Existing files are
never overwritten; use a new run-specific directory for each evaluation.

## Generate Module 5.8 evidence

Module 5.8 reads canonical `final_prompt_summary.json` artifacts from prior real
historical executions. Prompt text is required only ephemerally to reconstruct
the autonomy goal; it is represented only by SHA-256 in certification evidence.
Source paths, prompts, result payloads, and identifiers are not stored.

The default policy requires at least 25 evaluated results, 10 unique goals,
five terminal safety results, five non-terminal review results, zero duplicate
comparisons, zero contract failures, exact decision agreement, and p95 shadow
latency below 250 ms. Insufficient or stale-only history creates a valid
blocked report; it must not be relabelled as ready.

```bash
RUN_DIR="$(mktemp -d /tmp/punk-module5-evidence.XXXXXX)"

PYTHONPATH=. python scripts/run_module5_repeated_shadow_evidence.py \
  --tenant-id tenant_slug \
  --historical-run-root data/prompt_runs \
  --env-file .env \
  --inventory-output "$RUN_DIR/module5-8-inventory.json" \
  --output "$RUN_DIR/module5-8.json"
```

Explicit exported summaries can also be added by repeating
`--legacy-result /safe/path/final_prompt_summary.json`. Every production-route,
agent-effect, scale-cutover, security-release, activation, and export flag must
remain disabled.

## Module 5.9 inputs

### Historical semantic feature prerequisites

Legacy `sklearn_hashing_vectorizer` snapshots are privacy-safe historical
sources, but they are not valid substitutes for the governed semantic models.
Module 5.9 requires two separately built feature sets using the exact pinned
primary E5 and complementary multilingual MiniLM revisions. The source rows
may be existing historical cohort traits only when every row is aggregated,
retrieval-eligible, non-activatable, rights-permitted, and at least k=1000.

The runtime reader needs `SELECT` on the immutable
`audience_embedding_models` registry so it can enforce approval at query time.
It receives no access to feature-build jobs and no registry write privilege.
Re-run the existing role provisioner with the current total feature-vector
count after applying the privilege repair.

Before re-embedding, both pinned model revisions must have passed the existing
production embedding benchmark and must be explicitly registered by an
operator. This command never registers or approves a model automatically.

Build one model-specific feature set at a time:

```bash
PYTHONPATH=. python scripts/build_module5_historical_semantic_features.py \
  --tenant-id tenant_slug \
  --source-feature-set-id reviewed_safe_source_set \
  --source-feature-set-version 1 \
  --model-role primary \
  --requested-by operator_slug \
  --confirm-privacy-safe-historical-reembedding

PYTHONPATH=. python scripts/build_module5_historical_semantic_features.py \
  --tenant-id tenant_slug \
  --source-feature-set-id reviewed_safe_source_set \
  --source-feature-set-version 1 \
  --model-role complementary \
  --requested-by operator_slug \
  --confirm-privacy-safe-historical-reembedding
```

The source-reader transaction is read-only and never selects the stored legacy
embedding column. The writer publishes only new immutable semantic feature
sets. Source freshness is preserved, activation remains false, and no export
or production routing occurs.

The functional input is operator-owned JSON with four sections:

- `goal`: `tenant_id`, opaque `request_id` and `goal_id`, objective,
  `requested_outcomes`, `execution_mode=shadow`, and optional bounded budget;
- `run`: opaque `functional_run_id`, purpose, and shadow execution mode;
- `retrieval`: language, pinned canonicalizer model, pinned primary and
  complementary model bindings, explicit governed constraints, and result
  limit; and
- `authorization_context`: a short-lived bounded-agent principal, read/propose
  scopes, exact capability allowlist, and a fingerprint of the upstream
  authentication assertion. It must not contain a token or credential.

The taxonomy is supplied separately and must have `production_retrieval`
approval scope, native-human review, `approved_for_production_retrieval`
status, an exact canonical fingerprint, and no benchmark-oracle lineage.

Model binding JSON uses the existing `EmbeddingModelSpec` fields. Revisions
must be immutable, dimensions must match the `vector(384)` schema, and the
model registry must contain passed, tenant-approved records. Primary retrieval
uses depth 10 and expansion 30; complementary retrieval uses depth 10 and
expansion 10.

The authoritative historical result is supplied as `--legacy-result`. Raw
source material is not an allowed input. The generated report retains only
minimized counts, decisions, timings, reason codes, and fingerprints.

```bash
PYTHONPATH=. python scripts/run_module5_functional_shadow_evidence.py \
  --source-certification-report "$RUN_DIR/module5-8.json" \
  --functional-input "$RUN_DIR/module5-9-input.json" \
  --taxonomy-file "$RUN_DIR/governed-taxonomy.json" \
  --legacy-result "$RUN_DIR/historical-result.json" \
  --env-file .env \
  --output "$RUN_DIR/module5-9.json"
```

`AUDIENCE_FEATURE_DATABASE_URL` must identify the tenant-scoped read runtime.
The functional runner deliberately uses that same read credential for feature
metadata, model-approval lookup, and pgvector retrieval. It never selects the
writer or migration credential.

## Module 5.10 inputs

The security command evaluates the effective environment, including
production mode, API and signed-tenant authentication, host/CORS restrictions,
TLS requirements, secret-manager injection, non-root/read-only container
controls, request bounds, disabled debug/demo/docs surfaces, mandatory agent
capability authorization, and disabled release-effect flags.

Environment values and credentials are never written into evidence. Only
control names, booleans, evidence codes, counts, and fingerprints are stored.

```bash
PYTHONPATH=. python scripts/run_module5_agent_security_evidence.py \
  --tenant-id tenant_slug \
  --assessment-id "security-posture-$(date +%s)" \
  --certification-id "agent-security-$(date +%s)" \
  --functional-shadow-report "$RUN_DIR/module5-9.json" \
  --env-file .env \
  --posture-output "$RUN_DIR/security-posture.json" \
  --certification-output "$RUN_DIR/module5-10.json" \
  --summary-output "$RUN_DIR/module5-10-summary.json"
```

Exit status `2` means a prerequisite or control gate blocked certification.
That is a valid fail-closed outcome. Do not manufacture the missing
certification file.

## Validate lineage before scale execution

```bash
PYTHONPATH=. python scripts/validate_module5_certification_chain.py \
  --source-certification-report "$RUN_DIR/module5-8.json" \
  --functional-shadow-report "$RUN_DIR/module5-9.json" \
  --agent-security-report "$RUN_DIR/module5-10.json"
```

After a real staging fault driver is configured, run Module 5.12 with the
existing `run_module5_historical_scale_certification.py` command. The output is
also private, atomic, and non-overwriting. Re-run the validator with
`--scale-recovery-report` to verify the full fingerprint chain.

## What this still does not certify

Passing these commands does not certify fresh provider delivery, a production
identity issuer, external penetration testing, activation connectors,
infrastructure failover, backup restore, disaster recovery, or live traffic.
Those remain external staging and rollout gates.
