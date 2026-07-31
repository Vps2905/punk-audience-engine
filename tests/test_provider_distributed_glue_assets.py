import json
from pathlib import Path

from scripts.provider_distributed_privacy_glue_job import (
    gaussian_private_count,
    parse_s3_ref,
)


ROOT = Path(__file__).resolve().parents[1]
GLUE_JOB = ROOT / "scripts" / "provider_distributed_privacy_glue_job.py"
STATE_MACHINE = (
    ROOT
    / "infrastructure"
    / "stepfunctions"
    / "provider_distributed_processing.asl.json"
)
MIGRATION = (
    ROOT
    / "migrations"
    / "0007_provider_distributed_privacy_releases.sql"
)
CLOUDFORMATION = (
    ROOT
    / "infrastructure"
    / "cloudformation"
    / "provider_distributed_data_plane.yaml"
)


def test_glue_noise_is_stable_per_immutable_release():
    first = gaussian_private_count(
        1000,
        keyed_seed="hmac-output",
        fingerprint="a" * 64,
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
    )
    second = gaussian_private_count(
        1000,
        keyed_seed="hmac-output",
        fingerprint="a" * 64,
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
    )
    other = gaussian_private_count(
        1000,
        keyed_seed="different-hmac-output",
        fingerprint="a" * 64,
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
    )

    assert first == second
    assert first[1] == other[1]
    assert parse_s3_ref("s3://bucket/key") == ("bucket", "key")


def test_glue_job_never_publishes_identifier_columns():
    source = GLUE_JOB.read_text(encoding="utf-8")

    assert "CryptographicHash.apply" in source
    assert "HMAC_SHA256" in source
    assert "row_number().over" in source
    assert "min_cohort_size" in source
    assert ".partitionBy(privacy_window)" in source
    assert ".drop(" in source
    assert "_punk_bounded_count" in source
    assert "output.write.mode(\"errorifexists\")" in source
    assert "complete_entity_partition_validation" in source
    assert "provider_data_rights_requests" in source
    assert "left_anti" in source
    assert "_write_canonical_manifest" in source
    assert "ChecksumAlgorithm" in source


def test_state_machine_runs_glue_sync_and_always_finalizes():
    definition = json.loads(STATE_MACHINE.read_text(encoding="utf-8"))
    states = definition["States"]

    assert states["RunGluePrivacyJob"]["Resource"] == (
        "arn:aws:states:::glue:startJobRun.sync"
    )
    assert (
        states["RunGluePrivacyJob"]["Catch"][0]["Next"]
        == "FinalizeFailure"
    )
    assert states["FinalizeSuccess"]["Parameters"]["Payload"]["action"] == (
        "finalize"
    )
    assert states["FinalizeFailure"]["Parameters"]["Payload"]["action"] == (
        "fail"
    )


def test_privacy_release_migration_has_budget_and_idempotency_guards():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "provider_privacy_releases" in sql
    assert "ingestion_id text not null" in sql
    assert "fingerprint char(64) not null" in sql
    assert "unique (ingestion_id, dispatch_attempt)" in sql
    assert "budget_scope" in sql
    assert "check (epsilon > 0)" in sql
    assert "canonical_ref is not null" in sql


def test_cloudformation_enforces_short_lived_encrypted_staging_and_roles():
    template = CLOUDFORMATION.read_text(encoding="utf-8")

    assert "DeleteRawDistributedStaging" in template
    assert "ExpirationInDays: 2" in template
    assert "SSEAlgorithm: aws:kms" in template
    assert "BlockPublicAcls: true" in template
    assert "s3:GetObjectVersion" in template
    assert "secretsmanager:GetSecretValue" in template
    assert "glue:StartJobRun" in template
    assert "StateMachineType: STANDARD" in template
    assert "DistributedReconciliationSchedule" in template
    assert "states:DescribeExecution" in template
    assert "AWSGlueServiceRole" not in template
    assert "IncludeExecutionData: false" in template
    assert "TracingConfiguration:" in template
