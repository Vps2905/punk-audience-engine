# Contributing

Thank you for helping improve the Punk Audience Intelligence Engine. This system
handles privacy, authorization, and delivery decisions, so changes must preserve
fail-closed behavior and produce reviewable evidence.

## Development workflow

1. Create a focused branch from the intended integration branch.
2. Keep each change small enough to review and roll back.
3. Add or update tests for behavior, safety boundaries, and failure paths.
4. Update the relevant contract, runbook, or change manifest.
5. Open a pull request using the repository template.

Do not push directly to `main`, `dev`, `Nightfury`, or another protected branch.
The current audience-intelligence integration branch is
`audience-intelligence-agents`.

## Engineering rules

- Do not hardcode a specific prompt, city, category, provider, tenant, or expected
  audience result to make a test pass.
- Do not commit `.env`, credentials, private keys, database dumps, raw provider
  records, raw identifiers, local caches, or runtime evidence.
- Do not weaken authentication, tenant isolation, k-anonymity, differential
  privacy, freshness, authorization, or approval gates.
- New production effects must remain disabled by default and require an explicit,
  tested authorization path.
- Terminal safety decisions must stop source access and downstream processing.
- Database changes require an ordered, unique migration and rollback/recovery
  consideration.
- External model and action dependencies must be pinned or governed through an
  approved registry.
- Never describe a historical, synthetic, or local test as live-production
  certification.

## Local setup

Use Python 3.11 and the locked dependency graph:

```bash
uv sync --frozen --python 3.11
```

Copy `.env.example` to `.env` and use only local/test credentials and safe data.
See the [README](README.md) for model prefetch and local-server commands.

## Required validation

At minimum, run:

```bash
python -m compileall -q app scripts tests
git diff --check

REQUIRE_AUDIENCE_API_KEY=true \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
LOCAL_SEMANTIC_MODEL_CACHE="$HOME/.cache/punk-audience-semantic-model" \
PYTHONPATH=. \
uv run --frozen python -m pytest -q
```

For migration changes, also run:

```bash
PYTHONPATH=. uv run --frozen python -m pytest -q \
  tests/test_migration_version_uniqueness.py
```

Run focused tests while developing, but do not use them as a replacement for the
full suite before handoff. CI remains the authoritative shared verification.

## Pull-request expectations

A pull request should explain:

- the user or operational outcome;
- the modules and contracts affected;
- privacy, security, tenant, freshness, and activation impact;
- configuration or migration changes;
- tests and manual evidence produced;
- rollout, rollback, and remaining limitations.

Generated evidence should be summarized without committing runtime data or
credentials. Link to approved external evidence storage when appropriate.
