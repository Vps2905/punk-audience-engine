# Production CI and Supply-Chain Gate

## Scope

This gate validates a candidate commit without deploying cloud resources,
enabling production traffic, exporting audiences, or changing any production
feature flag. It is an engineering gate, not live-production certification.

The workflow runs only for the approved audience-intelligence development
branch, pull requests targeting the existing protected branches, or an explicit
manual workflow dispatch.

## Required evidence

The `Production CI` workflow produces or verifies the following evidence:

1. All Python sources in `app`, `scripts`, and `tests` compile.
2. Migration numeric prefixes remain unique.
3. The complete pytest suite runs with a disposable PostgreSQL 16 service.
   This enables the database-backed privacy-budget integration cases that are
   skipped on machines without `AUDIENCE_TEST_DATABASE_URL`.
4. `uv sync --frozen` installs the exact CPU-only runtime and test graph from
   `uv.lock`, and `pip-audit` reports no known vulnerability in its exported
   runtime dependency graph. The CPU-suffixed PyTorch wheel is additionally
   covered by the container scan because PyPI's audit resolver does not publish
   that local-version wheel on its default index.
5. Bandit reports no high-severity Python finding with medium-or-higher
   confidence.
6. `cfn-lint` validates every CloudFormation template.
7. The production Dockerfile builds and the candidate container passes its
   `/health` smoke check with demo routes disabled.
8. Trivy reports no fixed or actionable critical vulnerability in the candidate
   image.
9. A CycloneDX container SBOM and its keyless GitHub-OIDC Sigstore bundle are
   retained as immutable workflow artifacts.
10. Every third-party GitHub Action is pinned by a full commit SHA.

### Immutable semantic-model artifact

The prefetch step resolves the configured model at its reviewed 40-character
commit SHA and materializes the snapshot into a revision-addressed local
directory. It then loads that directory with `local_files_only=True` before
writing `artifact-manifest.json`. The manifest records the model identity,
embedding dimension, every model file's size and SHA-256, and a deterministic
tree SHA-256.

When `LOCAL_SEMANTIC_MODEL_CACHE` is configured, runtime loading requires this
manifest, recomputes all recorded digests, and loads the verified artifact path
without consulting the Hugging Face cache resolver or network. Missing,
mismatched, symlinked, or modified artifacts fail closed. Re-run the prefetch
step after intentionally changing the pinned model revision; do not hand-edit
the artifact or manifest.

Test JUnit evidence is retained for 30 days. Container SBOM evidence is retained
for 90 days. Artifact retention is an operational convenience and does not
replace the future immutable production-certification evidence ledger.

## Fail-closed behavior

Any failed job blocks the container job. No job pushes an image, deploys a
CloudFormation stack, applies a database migration, enables traffic, or mutates
an audience. The candidate image exists only on the temporary GitHub runner.

The container scan initially blocks `CRITICAL` vulnerabilities. Before a live
production release, promote `HIGH` vulnerabilities to blocking after the
current dependency and base-image findings have been reviewed and explicitly
remediated or accepted with an expiry.

## Dependency updates

Dependabot opens weekly grouped updates for Python packages, the Docker base
image, and GitHub Actions. Every update must pass the same tests and security
gates. An automated dependency update is never production approval.

## Local validation

Run the non-container checks before uploading a patch:

```bash
python -m compileall -q app scripts tests
PYTHONPATH=. python -m pytest -q tests/test_production_ci_supply_chain.py
uv sync --frozen
RUNTIME_SITE_PACKAGES="$(
  uv run --frozen --no-dev python -c \
    'import site; print(site.getsitepackages()[0])'
)"
pip-audit --path "$RUNTIME_SITE_PACKAGES" --progress-spinner off
bandit -r app scripts --severity-level high --confidence-level medium --quiet
cfn-lint infrastructure/cloudformation/*.yaml
```

Run the database integration suite with a disposable PostgreSQL database:

```bash
AUDIENCE_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/audience_ci \
REQUIRE_AUDIENCE_API_KEY=true \
PYTHONPATH=. \
python -m pytest -q
```

Do not point `AUDIENCE_TEST_DATABASE_URL` at a shared or production database.
The integration fixture recreates the public schema.

## Remaining supply-chain closure

The runtime graph, CPU-only PyTorch source, Python base image, semantic model,
and CI actions are now immutable inputs. This gate still does not claim live
release closure. The approved ECR publishing path must sign the published image
digest and attach build provenance; the current workflow signs its SBOM but
deliberately does not push or sign an undeployed image.
