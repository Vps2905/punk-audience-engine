# Module 4.1 governed evolution snapshots

Module 4.1 establishes the production evidence foundation for audience
evolution. It snapshots privacy-safe aggregate cohorts from tenant-isolated run
history and classifies monitoring state using freshness, quality, approval and
policy evidence.

The service records baselines only. It does not read audience membership,
infer individual behaviour, mutate cohorts, approve changes, route traffic,
activate audiences or export data. Historical and offline inputs remain
`historical_baseline`; production inputs may be monitored, paused, blocked or
sent for quality review.

Migration `0020_module4_governed_evolution_snapshots.sql` stores immutable
fingerprinted run and cohort evidence with forced PostgreSQL row-level security
through `app.tenant_id`. All release-affecting flags are disabled by default.

## Module 4.2-4.5 control plane

The completed evolution control plane compares consecutive validated snapshots
and produces aggregate drift evidence for quality, freshness, approval, policy,
new cohorts and missing cohorts. Drift is converted into review-only actions:
maintain monitoring, investigate, pause and review, onboarding review, or
rollback review.

Manual review evidence must explicitly reference a recommendation before a
shadow observation is accepted. Shadow observations never route traffic.
Recovery plans cover every reviewed recommendation but always require manual
execution and remain unexecuted evidence.

Migration `0021_module4_evolution_control_plane.sql` stores immutable drift,
recommendation, approval/shadow, and recovery reports. Forced tenant RLS,
append-only triggers, fingerprints and database constraints preserve lineage
and prohibit automatic evolution, approval, routing, activation and export.

Module 4 engineering readiness requires the complete 4.1-4.5 evidence chain.
Live-data calibration and production release approval remain separate gates.
