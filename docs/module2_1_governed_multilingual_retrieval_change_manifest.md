# Module 2.1 Governed Multilingual Retrieval

## Scope

This additive change introduces an offline-first production boundary for:

- approved multilingual constraint canonicalization;
- exact supported-location handling with no nearest-location fallback;
- per-field resolution status, confidence, margin, and clarification output;
- approved dual-model candidate generation;
- reciprocal-rank fusion without cross-model cosine blending;
- structured constraint reranking;
- conditional primary-model expansion from depth 10 to depth 30;
- fail-closed handling when no candidate satisfies every resolved constraint.

## Deliberately unchanged

The change does not modify:

- `LocalSemanticIntentService`;
- `HybridAudienceRetrievalService`;
- `AudienceFeatureProposalService`;
- `DurableAudienceFeatureProposalService`;
- the Punk AI audience proposal API;
- model-registration data;
- benchmark thresholds;
- activation or export behavior.

## Runtime gates

All canonicalizer and retrieval models must use immutable
`EmbeddingModelSpec` values and pass the existing tenant-scoped production
model-registry gate. The service refuses unapproved models and validates that
each feature set and its lineage match the approved model specification.

## Safety

The service never combines raw cosine scores across models. It does not read or
store raw identifiers. It does not create audience proposals, register models,
change thresholds, activate audiences, or export data. Successful retrieval is
returned only for human review.

## Validation status

The included tests use fake approved registries, encoders, and feature stores.
They validate orchestration and safety behavior without asserting that any real
model has passed production approval. Native multilingual review, real
canonicalizer benchmarking, model approval, unsupported-location calibration,
and production latency/cost validation remain release gates.
