# Module 2.2 Governed Multilingual Taxonomy

## Scope

This change adds a production-oriented, fail-closed taxonomy lifecycle for Module 2 multilingual canonicalization and offline benchmark evaluation.

It does not register models, enable production routing, create proposals, change thresholds, activate audiences, or export data.

## Added

- A governed taxonomy loader that verifies:
  - approved taxonomy status;
  - canonical taxonomy fingerprint;
  - checksum-pinned source-language-pack lineage;
  - source benchmark dataset fingerprint binding;
  - explicit approval scope;
  - native-human review status;
  - production certification status;
  - absence of benchmark oracle-case constraint usage;
  - non-ambiguous aliases through the existing taxonomy contract.
- An engineering taxonomy builder that:
  - derives canonical location, category, and daypart values from privacy-safe benchmark documents only;
  - never reads expected case constraints to construct aliases;
  - requires a checksum-pinned reviewed language pack;
  - requires complete labels for every observed category and daypart in every declared language;
  - binds the taxonomy to the canonical benchmark dataset fingerprint;
  - preserves the immutable benchmark dataset fingerprint;
  - writes taxonomy and dataset-envelope outputs atomically;
  - keeps native-human production signoff pending.
- Optional benchmark CLI taxonomy attachment using `--taxonomy` and a required `--expected-taxonomy-sha256`.
- Taxonomy lineage in the benchmark report and production-certification blockers.

## Safety behavior

Engineering-only taxonomies are accepted only for offline evaluation. A taxonomy claiming production scope is rejected unless native-human review is complete. Any fingerprint or checksum mismatch, cross-dataset attachment, missing lineage, incomplete language coverage, ambiguous alias, or oracle-case lineage fails closed.

Generated taxonomy, envelope, benchmark JSON, and logs remain evidence artifacts outside the repository.

## Example

```bash
PYTHONPATH=. python scripts/build_production_governed_multilingual_taxonomy.py \
  --dataset "$EVIDENCE_DIR/punk-module2-global-benchmark-dataset.json" \
  --language-pack "$HOME/Downloads/module2_embedding_benchmark_language_pack_ai_reviewed.json" \
  --taxonomy-output "$EVIDENCE_DIR/punk-module2-engineering-multilingual-taxonomy.json" \
  --envelope-output "$EVIDENCE_DIR/punk-module2-global-benchmark-with-taxonomy.json" \
  --expected-language-pack-sha256 "$LANGUAGE_PACK_SHA256" \
  --confirm-engineering-benchmark-only
```

The original dataset file is never overwritten.
