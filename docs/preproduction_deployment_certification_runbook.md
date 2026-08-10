# Preproduction Deployment Certification Runbook

This gate proves that the Punk Audience runtime can be deployed, migrated and
started in an isolated AWS preproduction environment without fresh provider
data. It does not authorize production traffic, audience activation or export.

## What this stage adds

The runtime template now has two explicit phases:

1. `foundation` creates the isolated network, encrypted database, load balancer,
   ECS cluster, task definitions, alarms and services with zero running tasks;
2. an operator launches the one-shot migration task and verifies a successful
   exit; and
3. `runtime` starts the multi-AZ API and worker services only after migration
   evidence is accepted.

The one-shot bootstrap applies the complete checksum-protected migration set,
creates separate non-admin API and worker database identities, removes inherited
memberships, applies bounded DML grants and binds the worker identity to the
isolated deployment tenant. The API and worker then receive those identities
from `RuntimeDatabaseSecretArn`. The database administrator secret is available
only to RDS and the bootstrap task execution role. Application containers cannot
retrieve it through their task roles.

## External prerequisites

Prepare these resources outside source control:

- an immutable ECR image reference using `repository@sha256:<digest>`;
- reviewed ACM, KMS, S3, SQS, DLQ and SNS resources;
- `DatabaseSecretArn`, containing the RDS administrator `username` and
  `password` keys;
- `RuntimeDatabaseSecretArn`, containing `api_username`, `api_password`,
  `worker_username` and `worker_password` keys; and
- `RuntimeSecretArn`, containing the application and privacy secrets documented
  in the infrastructure runbook.

`DeploymentTenantId` must name the isolated preproduction tenant assigned to the
provider worker. It is an identifier, never a secret or production data value.

Never put secret values or database URLs in CloudFormation parameter files,
shell history, evidence reports or source control.

## Safe deployment sequence

Use an authorized AWS operator role and an isolated preproduction account.
Validate the template and create a reviewed change set with
`DeploymentPhase=foundation`. Do not point production DNS or provider events at
the stack.

After the foundation change set completes, run the emitted
`MigrationTaskDefinitionArn` once in the private application subnets and runtime
security group. The task command is fixed in the template:

```text
python scripts/production_runtime_entrypoint.py migration
```

The task builds a TLS-only database URL in memory, removes component credentials
from the child environment, applies all checksum-protected migrations, creates
and verifies the separated runtime roles and exits.
Stop if the ECS task does not exit with code zero, any migration remains pending,
or an existing migration checksum differs.

Only after migration verification should a second reviewed change set set
`DeploymentPhase=runtime`. The deployment circuit breaker must remain enabled.
Production DNS, provider delivery, activation and downstream export must remain
disabled.

## Measured certification evidence

Collect only sanitized measurements. Do not store stack names, ARNs, endpoints,
secret names, subnet IDs, account IDs, tenant data or credentials. The evidence
object must contain the exact fields accepted by
`ProductionPreproductionDeploymentCertificationService`, including:

- stack and reviewed-change-set binding status;
- running immutable image digest;
- API/worker desired, running, healthy and availability-zone counts;
- database encryption, privacy, Multi-AZ, TLS, backup and deletion protection;
- applied migration head, pending count and checksum mismatch count;
- runtime database-role separation and absence of administrator credentials;
- secret injection and plaintext-secret count;
- alarm-route state and rollback protection; and
- explicit false values for production traffic, export and live provider data.

Evaluate the sanitized evidence:

```bash
PYTHONPATH=. python scripts/evaluate_preproduction_deployment.py \
  --request /secure/operator/preproduction-request.json \
  --observations /secure/operator/preproduction-observations.json \
  --output /secure/operator/preproduction-certification.json
```

The request and observation files belong in an operator-controlled location
outside the repository. The generated report contains fingerprints, aggregate
counts and pass/fail controls only.

Set `PREPRODUCTION_DEPLOYMENT_CERTIFICATION_PATH` to the generated report for
the infrastructure status endpoint. A valid report changes readiness to
`preproduction_deployment_certified`; it still reports
`live_production_certified=false`.

## Failure rules

Certification fails closed when any required observation is missing, unknown or
invalid, or when any of these conditions occurs:

- the image digest differs from the reviewed candidate;
- either runtime service is unhealthy or not spread across two zones;
- the database is public, unencrypted, single-AZ or missing TLS/backup controls;
- migrations are pending or checksum-invalid;
- API and worker share a database identity or receive administrator credentials;
- plaintext secrets are detected;
- alarm routing or rollback is not verified; or
- production traffic, downstream export or live provider delivery is enabled.

Fresh provider data is intentionally not required for this gate. It remains a
separate final vendor acceptance and fresh-data shadow certification.
