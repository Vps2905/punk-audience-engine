# Module 2 Full Completion Change Manifest

## Baseline

- Repository branch: `audience-intelligence-agents`
- Required base commit: `358ceba18aec560f233e54226ad3dd69650d894f`
- Existing Module 2.1/2.2 benchmark result retained:
  - full multilingual canonicalization: 1.0;
  - final candidate recall: 1.0;
  - structured semantic top-1: 1.0;
  - unsupported rejection: 1.0;
  - p95 latency: 44.651895 ms;
  - p99 latency: 45.207718 ms.

## Added: Module 2.3 certification and calibration

- Native-language review decisions with named reviewer identity, native-language
  confirmation, conflict counts, review timestamps, and independent release-owner
  validation.
- Review manifests bound to the exact taxonomy fingerprint and language-pack
  checksum.
- Privacy-safe unsupported-location calibration observations that store only
  fingerprints, language, safe reason codes, and readiness outcomes.
- Exact one-sided zero-failure confidence calculation. The default 1% target at
  95% confidence requires at least 299 zero-failure unsupported cases.
- Per-language calibration coverage requirements.
- Combined engineering, review, calibration, index, shadow, model-registration,
  and external-release gate report.
- Report fingerprint validation and forbidden raw-field checks.
- No automatic human approval or production certification.

## Added: Module 2.4 governed index lifecycle

- Immutable index manifest bound to tenant, taxonomy, three model fingerprints,
  source fingerprint, feature-set version, data-use mode, and embedding dimension.
- Candidate, building, built, validated, shadow, active, retired, and failed states.
- Validation evidence for count, dimension, duplicates, missing/invalid vectors,
  tenant isolation, taxonomy/model/source binding, recall, latency, and checksum.
- Atomic active-index switch and rollback repository operations.
- New immutable version requirement for incremental updates; no in-place vector
  mutation.
- In-memory and atomic JSON repositories for tests/operator rehearsal.
- Transactional PostgreSQL repository for migration 0011.
- Active promotion requires explicit routing approval, full production
  certification, production data-use mode, and passing shadow evidence.

## Added: Module 2.5 shadow serving and release hardening

- Disabled-by-default retrieval-only shadow comparator.
- Candidate output is never returned to the user or substituted for incumbent
  output.
- Privacy-safe observation records contain request/result fingerprints, safe
  statuses, latency, agreement, errors, and safety divergence only.
- Agreement is safe decision-and-canonical-constraint equivalence; exact
  result signatures remain separate so expected model reranking drift does not
  make the 95% release gate impossible to satisfy.
- The legacy proposal boundary now fails closed when any non-empty structured
  location, category, daypart, or exclusion cannot be represented as a canonical
  slug. Multilingual filters must pass through the governed canonicalizer instead
  of being silently dropped and widening retrieval.
- Readiness policy for sample size, agreement, safety divergence, candidate
  errors, and p95 latency overhead.
- Release-gate recommendation report that performs no index mutation or routing.
- Authenticated Module 2 status endpoint reporting configuration/evidence
  presence without returning secret values or filesystem paths.

## Database

Migration `0011_module2_certification_index_shadow.sql` adds:

- immutable certification reports;
- governed retrieval index records and transition events;
- one-active-index-per-tenant partial unique constraint;
- privacy-safe shadow observations;
- immutable identity triggers;
- tenant row-level security and PUBLIC privilege revocation.

## Operator scripts

- `plan_module2_certification_review.py`
- `evaluate_module2_certification.py`
- `rehearse_module2_index_lifecycle.py`
- `evaluate_module2_shadow_readiness.py`

All scripts require an explicit confirmation flag and keep model registration,
production routing, proposal creation, activation, and export disabled.

## Validation

The package-focused suite passes 47 tests, covering the pre-existing governed
retrieval/taxonomy benchmark and the new Module 2.3–2.5 services.

The full repository suite must still be executed in the project virtual
environment after installation because the packaging environment does not carry
all repository runtime dependencies and external services.
