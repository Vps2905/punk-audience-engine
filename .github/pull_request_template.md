## Summary

Describe the outcome and why this change is needed.

## Scope

- Modules/components:
- API, contract, migration, or configuration changes:
- Out of scope:

## Safety and operational impact

- Authentication/authorization:
- Tenant isolation:
- Privacy/k-anonymity/DP:
- Freshness/quality:
- Activation/export:
- Rollout and rollback:

## Validation

List focused tests, the full-suite result, manual checks, and safe evidence.

## Checklist

- [ ] No secrets, raw identifiers, provider records, local caches, or runtime evidence are committed.
- [ ] No prompt, city, category, tenant, provider, or expected result is hardcoded.
- [ ] Authentication, tenant isolation, terminal safety, and approval gates remain fail closed.
- [ ] New production effects are disabled by default and require explicit authorization.
- [ ] Migrations are ordered and unique; recovery/rollback impact is documented.
- [ ] Focused tests and the full suite pass with the locked dependencies.
- [ ] Documentation and operator runbooks reflect the behavior and limitations.
- [ ] Historical, synthetic, or local evidence is not presented as live-production certification.
