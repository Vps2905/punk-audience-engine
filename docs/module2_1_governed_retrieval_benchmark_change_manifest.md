# Module 2.1 governed retrieval benchmark

This additive change evaluates the real governed multilingual canonicalizer and
candidate-retrieval pipeline against an immutable offline benchmark dataset.

## Scope

- Uses query text only; benchmark gold constraints are never copied into the
  production request fields.
- Builds two privacy-safe in-memory feature sets with the exact pinned E5 and
  MiniLM revisions already governed by Module 2.1.
- Runs canonicalization, dual-model top-10 generation, reciprocal-rank fusion,
  conditional E5 top-30 expansion, and structured reranking.
- Measures per-field and full canonicalization accuracy, base/final candidate
  recall, exact/semantic top-1 accuracy, clarification accuracy, unsupported
  rejection, expansion rate, per-language success, latency, and process peak
  memory.
- Stores case IDs and canonical evidence only; raw query text, document text,
  vectors, raw identifiers, and activation payloads are not persisted.

## Governance boundaries

- The local approval gate is explicitly scoped to offline benchmarking.
- No production model registration is performed.
- No threshold is applied to runtime routing.
- Automatic proposals, activation, and downstream export remain disabled.
- Engineering pass and production certification are separate. Certification
  remains blocked until native-human signoff and enough independent unsupported
  cases exist for the requested false-match confidence bound.
