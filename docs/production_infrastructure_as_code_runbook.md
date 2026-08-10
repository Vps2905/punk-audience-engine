# Production Infrastructure as Code Runbook

This stage defines the missing production runtime plane around the existing
provider distributed data plane. The template is reviewable infrastructure
evidence; nothing in the application creates a CloudFormation stack, executes
a change set, enables production traffic, or authorizes a release.

## Runtime architecture

`infrastructure/cloudformation/production_runtime_plane.yaml` provisions:

- one VPC across two availability zones;
- public subnets only for the TLS application load balancer and one NAT gateway
  per availability zone;
- private application subnets for the API and provider worker ECS tasks;
- isolated private database subnets without an internet route;
- an S3 gateway endpoint for private canonical and provider object access;
- hardened ECS Fargate task definitions using an immutable image digest,
  non-root UID/GID, read-only root filesystem, dropped capabilities and disabled
  ECS Exec;
- Secrets Manager injection without plaintext credentials in the template;
- encrypted CloudWatch log groups and an encrypted load-balancer log bucket;
- HTTPS-only forwarding with HTTP redirect, optional WAF association, invalid
  header rejection and deletion protection;
- an encrypted, private, deletion-protected, Multi-AZ PostgreSQL database with
  forced TLS, 35-day automated backups, final snapshots, enhanced monitoring
  and Performance Insights;
- ECS service circuit-breaker rollback, multi-AZ minimum task counts,
  autoscaling and alarms routed to an operator-owned SNS topic; and
- scoped task roles for the existing provider S3, SQS, KMS and canonical data
  plane.

## Required external resources

The template deliberately accepts existing resource references instead of
creating secrets or production data resources. Before validation, operators
must prepare and independently review:

- an ECR image referenced by `repository@sha256:<digest>`;
- ACM certificate and optional regional WAF web ACL;
- data KMS key whose policy permits RDS, CloudWatch Logs, ECS workers and the
  existing provider data plane;
- secrets KMS key whose policy permits the ECS task execution role to decrypt
  the runtime and database secrets;
- Secrets Manager runtime secret containing the documented JSON keys;
- Secrets Manager database secret containing `username` and `password`;
- a separate runtime database secret containing `api_username`, `api_password`,
  `worker_username` and `worker_password`;
- provider landing and canonical S3 buckets;
- provider SQS queue and DLQ; and
- an SNS alert topic with verified human recipients.

The container bootstrap receives the database endpoint from CloudFormation and
separate API, worker and migration credentials through ECS secret injection. It
URL-encodes the values, requires TLS, removes the component credentials from the
child environment and executes the selected process without printing the
resulting database URL. The administrator secret is never injected into the API
or provider worker. It is used only by the one-shot bootstrap task to apply the
complete immutable migration set and provision bounded runtime database roles.

Never place secret values, database URLs, external IDs or credentials in a
parameter file, template, evidence report, shell history or source control.

## Validation and reviewed change set

Run local compilation and tests first. Then use an authorized AWS operator
session to perform the read-only template validation:

```bash
aws cloudformation validate-template \
  --template-body file://infrastructure/cloudformation/production_runtime_plane.yaml
```

Prepare parameters outside the repository using secret/resource ARNs only.
Create a `CHANGE_SET_TYPE=CREATE` or `UPDATE` change set through the approved
deployment pipeline. Do not execute it. Review the complete replacement list,
IAM changes, security-group paths, subnet routes, database changes, quotas,
monthly cost, backup policy, alarms and rollback plan.

Only after those checks are independently evidenced should
`ProductionInfrastructureAssessmentService` receive all required attestations.
A manual review may then record `approved_for_preproduction_change_set`, but
the resulting report still sets all execution, traffic and release permissions
to false.

## Deployment gates

The safe defaults are:

```text
INFRASTRUCTURE_VALIDATION_ENABLED=false
INFRASTRUCTURE_APPLY_ENABLED=false
INFRASTRUCTURE_PRODUCTION_TRAFFIC_ENABLED=false
```

The status endpoint is:

```text
GET /api/audience-intelligence/infrastructure/status
```

It can report `infrastructure_engineering_evidence_ready` only when a passing
assessment and matching approved preproduction review are present. If apply or
traffic flags are enabled, the service fails closed. Even a ready result is not
live-production certification.

## Handoff to certification

Follow `docs/preproduction_deployment_certification_runbook.md` to deploy the
foundation with zero runtime tasks, run and verify the one-shot migration task,
and then start the isolated runtime through a second reviewed change set.

Do not enable production traffic after IaC review. The next stage must validate load,
autoscaling, queue pressure, dependency failure, zonal disruption, database
restore, rollback, recovery-time and recovery-point objectives against an
isolated preproduction stack. Production traffic remains blocked until those
results and the final operator approvals are complete.
