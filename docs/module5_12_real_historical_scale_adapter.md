# Module 5.12 — Real historical scale adapter

## Outcome

Module 5.12 connects the Module 5.11 measurement harness to the same governed
audience execution agent used by the prompt API. Each normal workload invocation
runs the real path:

1. read privacy-safe derived rows from PostgreSQL;
2. apply aggregation, k-anonymity and DP transformation;
3. run semantic retrieval and safe vector generation;
4. select, cluster and score cohorts;
5. build approval-gated export candidates; and
6. enforce Module 5 governance with delivery disabled.

The adapter is generalized. Its objective, tenant, row limit, k threshold,
epsilon, synthetic-row count and quality threshold are runtime inputs. No city,
category or campaign prompt is hardcoded in the implementation.

## Honest workload accounting

Module 5.11 now records both `declared_work_units` and the adapter's
`observed_work_units`. The historical adapter uses the `source_rows` returned by
the real pipeline. The workload gate uses that observed value. If a query was
expected to return 10,000 rows but returned 220, the report counts 220.

The `10k` and `100k` profiles calculate how many full pipeline calls are needed
from the operator-supplied expected row count. The final gate still uses the
observed total, so changing the estimate cannot manufacture a pass.

## Evaluation-effect boundary

Repeated certification calls must not create thousands of local run folders or
consume the operational DP ledger. The adapter therefore calls the production
agents in `certification_evaluation` mode:

- local artifact persistence is false;
- privacy and synthetic transformations still execute;
- operational privacy-budget spending is not recorded;
- the embedding/cohort compatibility path stays in memory;
- manual approval remains required;
- downstream export remains false; and
- no activation or release effect is permitted.

This is real computation against the configured historical source, but it is
not fresh-data certification and it is not delivery certification.

## Running a profile

First create an objective file containing a normal aggregate audience goal. Do
not place credentials or individual identifiers in it. Then run:

```bash
PYTHONPATH=. python scripts/run_module5_historical_scale_certification.py \
  --tenant-id punk_internal \
  --workload-id historical-audience-scale \
  --objective-file /approved/evidence/objective.txt \
  --functional-shadow-report /approved/evidence/module5_9.json \
  --agent-security-report /approved/evidence/module5_10.json \
  --profile 10k \
  --expected-source-rows 220 \
  --concurrency 4 \
  --output /approved/evidence/module5_12_10k.json
```

After the 10k profile is reviewed, repeat with `--profile 100k`. The script
prints a minimized summary and writes the tamper-evident Module 5.11 report. It
does not store the objective; only runtime processing sees it.

## Fault-controller boundary

Without a real fault controller, the throughput, concurrency and duplicate
pipeline calls run, but the full report remains blocked. This is intentional.
For a staging deployment, supply a reviewed controller factory:

```bash
--fault-controller-factory company_staging.module5_faults:build_controller
```

The factory receives the historical configuration and must return an object
implementing `HistoricalScaleFaultExerciseController`. That controller must
exercise real disposable-worker restart, real retryable database interruption,
bounded queue pressure, deadlines and circuit-breaker behavior. Its results are
still engineering evidence only. Database failover, backup/restore, disaster
recovery, fresh vendor data and delivery connector certification remain false
until their separate staging exercises complete.

## Acceptance order

1. Run `smoke` and confirm observed rows, zero effects and stable duplicate
   fingerprints.
2. Run `10k` with four-way concurrency.
3. Review latency, throughput, memory and observed-row coverage.
4. Run `100k` with the approved staging capacity.
5. Attach the real fault controller and rerun all eight scenarios.
6. Store the validated report under the existing immutable migration 0033
   evidence table.
7. Keep all production routing, agent effect authorization and cutover flags
   false.

No new database migration is required: Module 5.12 produces the same strict,
immutable Module 5.11 certification report stored by migration 0033.
