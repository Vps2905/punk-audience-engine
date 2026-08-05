# Module 2 Full Completion Runbook

## 1. Safety boundary

Module 2 code is engineering-complete, but installing it does not certify or
activate production routing. These flags must remain false until the stated
release gates are independently satisfied:

```dotenv
MODULE2_SHADOW_SERVING_ENABLED=false
MODULE2_MODEL_REGISTRATION_ENABLED=false
MODULE2_PRODUCTION_ROUTING_ENABLED=false
```

No script in this package registers a model, creates an audience proposal,
activates an audience, or exports data.

## 2. Apply migration

Apply `migrations/0011_module2_certification_index_shadow.sql` through the normal
operator-controlled migration process against the separate Punk-owned feature
database. Do not apply it to the provider/source database.

Verify:

- `module2_certification_reports` exists;
- `audience_retrieval_indexes` exists;
- `audience_retrieval_index_events` exists;
- `audience_retrieval_shadow_observations` exists;
- row-level security is enabled and forced;
- only one active index can exist per tenant.

## 3. Native-language review plan

```bash
PYTHONPATH=. python scripts/plan_module2_certification_review.py \
  --taxonomy /path/to/governed-taxonomy.json \
  --language-pack /path/to/reviewed-language-pack.json \
  --output /outside/repo/module2-native-review-plan.json \
  --confirm-no-auto-approval
```

A named native reviewer must review every required language. The release owner
cannot self-approve. Machine translation cannot be auto-approved. Final evidence
must be converted to the `module2-native-language-review-manifest-v1` contract.

## 4. Unsupported calibration

Collect at least 299 independently reviewed unsupported-location observations
for the default 1% false-match target at 95% confidence. The default policy also
requires at least 20 observations per required language.

Each observation stores:

- case ID;
- language;
- unsupported location token fingerprint;
- taxonomy fingerprint;
- expected rejection;
- observed retrieval-ready boolean;
- safe reason code;
- reviewer and UTC review time.

It must not store raw query text, document text, embeddings, raw identifiers, or
precise device/location data.

## 5. Certification evaluation

```bash
PYTHONPATH=. python scripts/evaluate_module2_certification.py \
  --benchmark-report /outside/repo/governed-benchmark.json \
  --native-review-manifest /outside/repo/native-review-manifest.json \
  --unsupported-calibration /outside/repo/unsupported-calibration.json \
  --external-gates /outside/repo/external-gates.json \
  --output /outside/repo/module2-certification-report.json \
  --confirm-evaluation-only
```

Without explicit model, index, shadow, and external evidence flags, the report
may declare `module2_evidence_ready=true` but must keep
`production_certification_ready=false`.

## 6. Index lifecycle rehearsal

Create a `module2-governed-index-manifest-v1` JSON document, then execute each
state explicitly:

```bash
PYTHONPATH=. python scripts/rehearse_module2_index_lifecycle.py \
  --ledger /outside/repo/module2-index-ledger.json \
  --manifest /outside/repo/index-manifest.json \
  --action register \
  --confirm-no-production-routing
```

Repeat with `building`, `built`, `validate`, and `shadow`. Validation requires a
`module2-governed-index-validation-v1` JSON file. Shadow promotion requires a
valid Module 2 certification report with `module2_evidence_ready=true`.

The rehearsal CLI intentionally does not expose active promotion or rollback.
Those operations exist only in the lifecycle service and require explicit
production routing approval plus full release evidence.

## 7. Shadow serving

Integrate `ProductionModule2ShadowServingService` only with retrieval-only
adapters. Set `enabled=false` initially. When enabled in staging:

- incumbent output remains authoritative;
- candidate output is never returned to the caller;
- candidate errors are recorded safely;
- agreement compares routing outcome and canonical constraints, not exact
  ranking identity; incumbent/candidate result signatures preserve ranking
  drift for diagnosis;
- any candidate-ready/incumbent-blocked case is a safety divergence;
- no raw query is written to the observation sink.

Evaluate a safe observation export:

```bash
PYTHONPATH=. python scripts/evaluate_module2_shadow_readiness.py \
  --observations /outside/repo/shadow-observations.json \
  --output /outside/repo/shadow-readiness.json \
  --confirm-shadow-only
```

The production default requires at least 1,000 observations, at least 95%
agreement, zero safety divergence, at most 0.5% candidate errors, and at most
250 ms p95 latency overhead.

## 8. Status endpoint

Set evidence paths only after creating validated reports:

```dotenv
MODULE2_CERTIFICATION_REPORT_PATH=/secure/evidence/module2-certification.json
MODULE2_SHADOW_REPORT_PATH=/secure/evidence/module2-shadow.json
MODULE2_INDEX_RECORD_PATH=/secure/evidence/module2-index.json
```

Call with the Audience API key:

```text
GET /api/audience-intelligence/module-2/status
```

The endpoint returns booleans/status only. It never returns secret values or the
configured evidence paths.

## 9. Required verification

```bash
REQUIRE_AUDIENCE_API_KEY=true PYTHONPATH=. python -m pytest -q \
  tests/test_production_governed_audience_retrieval_service.py \
  tests/test_production_governed_constraint_taxonomy_service.py \
  tests/test_production_governed_retrieval_benchmark_service.py \
  tests/test_production_module2_certification_service.py \
  tests/test_production_module2_index_lifecycle_service.py \
  tests/test_production_module2_shadow_serving_service.py \
  tests/test_production_module2_status_and_migration.py

REQUIRE_AUDIENCE_API_KEY=true PYTHONPATH=. python -m pytest -q
```

Do not stage or commit runtime evidence, review worksheets, calibration data,
shadow logs, `.env`, caches, or files under `data/prompt_runs`.
