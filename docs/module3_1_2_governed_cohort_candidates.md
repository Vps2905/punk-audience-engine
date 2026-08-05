# Module 3.1-3.2: Governed Cohort Candidates

This phase introduces deterministic cohort candidate contracts, an aggregate-only
PostgreSQL control plane, explainable quality scoring, and a no-write historical
evaluation runner.

## Safety boundary

- Minimum cohort size is `k >= 1000`.
- Only retrieval-eligible aggregate feature rows with a safe privacy status are used.
- Sensitive POIs are blocked or routed to human review through the existing
  `SensitivePOIPrivacyRiskService`.
- Historical or stale sources remain preview-only.
- Candidate activation, export, lookalikes, persistence, overlap, and unique-reach
  computation remain disabled.
- The evaluator does not select embeddings, trait text, raw metadata, or identifiers.
- One source feature creates one candidate. Cohort sizes are never summed because
  privacy-safe intersection evidence is not yet available.

## Quality score

The score is transparent and bounded to `[0, 1]`:

- 45% source quality score
- 25% logarithmic size adequacy relative to `k_min`
- 15% freshness
- 10% required-field completeness
- 5% privacy threshold pass

Dwell, repeat visits, unique reach, overlap, campaign lift, and demographic traits
are explicitly marked unavailable and are not inferred.

## Module boundary

Implemented:

- 3.1 contracts, deterministic IDs/versions/fingerprints, schema, RLS
- 3.2 candidate generation, quality disclosure, sensitive-POI decisions
- authenticated status endpoint
- no-write historical evaluation script

Not implemented in this phase:

- 3.3 overlap and deduplication
- 3.4 governed lookalikes
- 3.5 lifecycle approval and monitoring
- 3.6 Punk AI shadow routing
