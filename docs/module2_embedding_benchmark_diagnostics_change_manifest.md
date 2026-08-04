# Module 2 Embedding Benchmark Diagnostics Change Manifest

## Purpose

Add a governed offline diagnostics path for failed embedding benchmarks without
weakening production thresholds or permitting model registration.

## Safety boundary

The diagnostics artifact contains only privacy-safe benchmark IDs, canonical
location/category/daypart labels, ranks, and finite similarity scores. It does
not persist query text, document text, embeddings, vectors, raw identifiers,
credentials, activation payloads, or export payloads.

The diagnostics service always emits:

- `registration_allowed=false`
- `recommended_threshold_auto_applied=false`
- `activation_or_export_performed=false`
- `raw_identifiers_read=false`
- `raw_identifiers_stored=false`

## Evidence added

- Exact-ID and semantic-signature hit rates.
- Per-language retrieval diagnostics.
- Privacy-safe top-candidate IDs, canonical traits, ranks, and scores.
- Exact-relevant versus named-hard-negative score margins.
- Existing hard-negative exclusion-at-k reported under an unambiguous name.
- Supported and unsupported score distributions.
- A deterministic model-specific rejection-threshold candidate for human
  review only.
- Unsupported-calibration sample-resolution evidence.
- Exact zero-failure sample count required for a one-sided 95% upper bound at
  the configured false-match target.

For a 1% unsupported false-match target, 20 unsupported cases have a 5%
empirical resolution and cannot certify the production target. With zero
observed false matches, at least 299 independent unsupported cases are needed
for the exact one-sided 95% upper bound to fall below 1%.

## Non-goals

- No policy thresholds are changed.
- No benchmark labels are rewritten automatically.
- No threshold candidate is applied automatically.
- No model is registered, promoted, activated, or exported.
