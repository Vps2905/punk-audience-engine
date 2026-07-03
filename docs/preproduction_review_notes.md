# Pre-production Review Notes

This document contains notes for the technical review and validation of the Audience Intelligence Engine prior to production deployment.

## Review Checklist
- [ ] End-to-end privacy pipeline validation (no raw identifiers).
- [ ] k-anonymity thresholds strictly enforced.
- [ ] Authentication (API Key) required for all endpoints.
- [ ] Approval workflow required before final export manifest generation.
- [ ] Differential privacy noise addition verified on seed generation.
- [ ] No fallback to unsafe synthetic engines in production mode.
- [ ] API latency meets SLA requirements.

## Validation Prompts & Expected Outputs
**Prompt:** "Create an audience of frequent evening diners in major metropolitan areas."
**Expected Output:**
- `status`: success
- `approval_status`: pending_approval
- `exported_cohorts`: > 0 (subject to k-anonymity constraints)
- Warning (if applicable): Explanations if specific metros fell below the threshold and were excluded.

## API Latency Notes
- The complete orchestrator pipeline (from prompt to clustered export-ready cohorts) currently completes in ~3.6 - 4.5 seconds on local benchmarks.
- Latency is primarily bound by the clustering algorithm and the number of rows pulled from the Postgres source.

## Scale & Load Testing Notes
- Current testing covers data slices of up to 10,000 rows.
- Future load testing must validate the Embedding Feature Store Agent's memory footprint when processing 100k+ rows simultaneously.

## Production-Readiness Gaps
- The current implementation uses local `sklearn` TF-IDF embeddings. This must be migrated to `pgvector` for scalable, persistent similarity search in production.
- The epsilon ledger is currently local to the job run. A global ledger must be implemented to track the total privacy budget across all historical queries.

