# Module 3.4: Governed Lookalike Review

Module 3.4 consumes validated Module 3.3 overlap/deduplication evidence and
creates deterministic, review-only candidate-to-candidate similarity
suggestions.

## Safety boundary

Module 3.4 does not receive audience membership and therefore cannot compute
member similarity, overlap percentages, unique reach, or a deduplicated
audience.

Only aggregate cohort metadata already present in the governed Module 3
candidate contract is compared.

Candidates requiring Module 3.3 overlap review are excluded by default.

Sensitive or blocked lifecycle candidates are excluded.

The version 2 review policy uses a minimum composite candidate quality of
`0.60` for seeds and targets. This admits high-source-quality historical
evidence whose composite score is reduced by stale freshness, while continuing
to exclude low-quality candidates. Admission is only to offline review; it does
not change freshness, approval, activation, or export eligibility.

Every suggestion remains approval gated and is explicitly ineligible for
activation and export.

## Similarity evidence

The transparent score uses:

- POI-type compatibility: 30%
- daypart compatibility: 20%
- lookback compatibility: 15%
- cohort-quality alignment: 20%
- source-quality alignment: 15%

No cohort-size similarity is used because cohort size is a privacy threshold
and quality signal, not evidence that two audiences contain similar members.

## Production boundary

This phase does not:

- generate audience membership;
- mutate cohort lifecycle state;
- route into Punk AI production traffic;
- activate audiences;
- export audiences;
- claim overlap or unique reach.

Migration `0016_module3_governed_lookalike_evidence.sql` provides immutable,
tenant-scoped evidence storage with row-level security.
