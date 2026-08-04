# Module 2 benchmark curation gate

## Outcome

Privacy-safe historical features can now be profiled into a deterministic,
content-addressed curation plan before any benchmark query is authored.

## Added

- exact document and taxonomy coverage results;
- explicit supplemental-document slots for real coverage deficits;
- structured hard-negative candidates that differ in one constraint;
- taxonomy distribution and constraint-signature evidence;
- atomic curation-plan CLI;
- regression tests for no generation, no auto-approval and no activation.

## Policy upgrade

`punk-global-embedding-policy-v2` adds minimum distinct location, category and
daypart coverage, minimum cases per language, and multi-constraint cases. This
prevents a benchmark with many repeated examples from presenting itself as a
global evaluation.

## Safety boundary

The planner does not generate queries, documents, gold labels, audience
volume, model scores, model approval, features, audience activation or export.
Every candidate pair remains subject to named human review.

## Curated document review enforcement

Curated synthetic and aggregated document catalogs now require approved named
human review, a UTC review timestamp, an explicit no-audience-volume-claim
attestation, and one unique source-lineage reference per document. Pgvector
catalogs exported from privacy-safe feature sets retain their system lineage
boundary and do not require the curated-catalog review manifest.
