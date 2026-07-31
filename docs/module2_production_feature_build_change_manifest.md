# Module 2 production feature-build change manifest

## Scope

This package adds the bounded production path from one privacy-safe canonical
S3 partition to immutable tenant-scoped pgvector features.

It does not:

- read or embed raw MAIDs, device IDs, movement paths, or observations;
- score or activate audiences;
- apply database migrations automatically;
- register or approve a model automatically;
- restart a service;
- modify the historical provider database;
- stage, commit, or push Git changes.

## Added capabilities

- exact S3 object version, size, SHA-256 and row-count validation;
- canonical privacy, rights, purpose and provenance enforcement;
- configurable global location/category/daypart field mappings;
- pinned immutable sentence-transformer revision;
- document/query prefix and normalization consistency;
- no hashing, TF-IDF, mutable-model, or silent model fallback;
- embedding shape, dimension, finiteness and zero-vector checks;
- content-addressed feature-set and feature identities;
- insert-only immutable pgvector publication;
- approved embedding-model registry with benchmark evidence;
- durable idempotent feature-build lifecycle and replay receipt;
- tenant RLS and least-privilege writer grants;
- controlled operator-only migration and model-registration commands;
- production environment validation and runbook.

## Installation verification

The package was built against the current scale/privacy data-plane code.

Scratch verification:

- Module 2 focused tests: 60 passed.
- All repository tests except four existing local-semantic test files:
  615 passed, 3 skipped.
- The excluded files require `sentence-transformers`, which is unavailable in
  the scratch verification environment.
- Python compilation passed.
- Example JSON validation passed.
- Patch whitespace validation passed.

The repository's real `.venv` includes the semantic dependency. After copying
the package, the operator must run the complete repository suite there before
any migration, model registration, indexing, deployment, or restart.

## Required real-repository verification

```bash
REQUIRE_AUDIENCE_API_KEY=true \
PYTHONPATH=. \
python -m pytest -q \
  tests/test_production_*.py \
  tests/test_durable_production_feature_build_service.py \
  tests/test_feature_query_embedding_model_pinning.py \
  tests/test_phase2_feature_role_provisioning_service.py \
  tests/test_environment_validation.py

REQUIRE_AUDIENCE_API_KEY=true \
PYTHONPATH=. \
python -m pytest -q

python -m compileall -q app scripts tests
git diff --check
git diff --cached --name-only
```

Do not apply migration 0008 or enable production feature builds until these
commands pass in the real repository and the database/model changes receive
separate explicit approval.
