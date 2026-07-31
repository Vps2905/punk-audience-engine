"""
AWS Glue/Spark data plane for large provider objects.

Deployment requirements:

* AWS Glue 4.0+ Spark job;
* this repository packaged through ``--extra-py-files``;
* tokenization and DP seed Secrets Manager ARNs configured as Glue job
  default/non-overridable arguments;
* a workload role restricted to the staged input, canonical output, result
  prefix, the two secrets, CloudWatch, and the configured KMS keys.

The script never collects event rows on the driver and never writes entity
identifiers, including hashed identifiers, to canonical output.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from typing import Any, Dict, Tuple
from urllib.parse import unquote, urlparse

from app.models.provider_scale_contracts import (
    ProviderDistributedJobRequest,
    ProviderDistributedJobResult,
)


PRIVACY_CONTROLS = (
    "exact_version_staging",
    "hmac_sha256_tokenization",
    "daily_contribution_bounding",
    "k_anonymity",
    "deterministic_per_release_gaussian_dp",
    "no_identifier_output",
    "encrypted_parquet_output",
    "complete_entity_partition_validation",
)


class ControlledDistributedPrivacyError(RuntimeError):
    def __init__(self, status: str, reason_code: str) -> None:
        super().__init__(reason_code)
        self.status = status
        self.reason_code = reason_code


def parse_s3_ref(value: str) -> Tuple[str, str]:
    normalized = str(value or "").strip()
    if not normalized.startswith("s3://"):
        raise ValueError("Expected an s3:// reference")
    remainder = normalized[5:]
    bucket, separator, key = remainder.partition("/")
    if not separator or not bucket or not key:
        raise ValueError("S3 reference must contain a bucket and key")
    if ".." in key.split("/"):
        raise ValueError("S3 reference contains an invalid path segment")
    return bucket, key


def gaussian_private_count(
    count: int,
    *,
    keyed_seed: str,
    fingerprint: str,
    epsilon: float,
    delta: float,
    sensitivity: float,
) -> Tuple[int, float]:
    """
    Stable noise for one immutable privacy release.

    ``keyed_seed`` is already the output of the Glue HMAC transform. The
    underlying secret is never present in this function or in Spark output.
    Replaying the same immutable object returns the same release instead of
    spending privacy budget for a second independently noised answer.
    """

    sigma = (
        sensitivity
        * math.sqrt(2.0 * math.log(1.25 / delta))
        / epsilon
    )
    seed = int(
        hashlib.sha256(
            f"{fingerprint}\x1f{keyed_seed}".encode("utf-8")
        ).hexdigest(),
        16,
    )
    noise = random.Random(seed).gauss(0.0, sigma)
    return max(0, int(round(float(count) + noise))), sigma


def _read_input(spark: Any, source_ref: str, data_format: str) -> Any:
    if data_format == "csv":
        return (
            spark.read.option("header", "true")
            .option("mode", "FAILFAST")
            .option("enforceSchema", "false")
            .csv(source_ref)
        )
    if data_format == "jsonl":
        return (
            spark.read.option("mode", "FAILFAST")
            .option("multiLine", "false")
            .json(source_ref)
        )
    if data_format == "parquet":
        return spark.read.parquet(source_ref)
    raise ControlledDistributedPrivacyError(
        "quarantined",
        "unsupported_distributed_data_format",
    )


def _write_result(
    *,
    client: Any,
    request: ProviderDistributedJobRequest,
    result_ref: str,
    result: ProviderDistributedJobResult,
) -> None:
    bucket, key = parse_s3_ref(result_ref)
    put_request: Dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": json.dumps(
            result.to_safe_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        "ContentType": "application/json",
        "ServerSideEncryption": (
            request.canonical_target.server_side_encryption
        ),
        "Metadata": {
            "ingestion-id": result.ingestion_id,
            "source-fingerprint": result.fingerprint,
            "result-contract": result.contract_version,
        },
    }
    if request.canonical_target.server_side_encryption in {
        "aws:kms",
        "aws:kms:dsse",
    }:
        put_request["SSEKMSKeyId"] = request.canonical_target.kms_key_id
    client.put_object(**put_request)


def _write_canonical_manifest(
    *,
    client: Any,
    request: ProviderDistributedJobRequest,
    canonical_ref: str,
    input_rows: int,
    output_rows: int,
) -> Tuple[str, str]:
    """Write a deterministic inventory for an immutable Parquet attempt."""
    bucket, prefix = parse_s3_ref(canonical_ref)
    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = str(item.get("Key") or "")
            if not key or key.endswith("/_MANIFEST.json"):
                continue
            head = client.head_object(
                Bucket=bucket,
                Key=key,
                ChecksumMode="ENABLED",
            )
            objects.append(
                {
                    "key": key,
                    "size": int(item.get("Size") or 0),
                    "etag": str(item.get("ETag") or "").strip('"'),
                    "version_id": head.get("VersionId"),
                    "checksum_sha256": head.get("ChecksumSHA256"),
                }
            )
    if not objects:
        raise ControlledDistributedPrivacyError(
            "failed", "canonical_output_manifest_empty"
        )
    manifest = {
        "contract_version": "provider-canonical-manifest-v1",
        "tenant_id": request.contract.tenant_id,
        "provider_id": request.contract.provider_id,
        "dataset_id": request.contract.dataset_id,
        "source_fingerprint": request.fingerprint,
        "delivery_window_id": request.manifest.delivery_window_id,
        "partition_index": request.manifest.partition_index,
        "partition_count": request.manifest.partition_count,
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "privacy_controls": list(PRIVACY_CONTROLS),
        "objects": sorted(objects, key=lambda row: row["key"]),
    }
    body = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    checksum = hashlib.sha256(body).hexdigest()
    manifest_key = f"{prefix.rstrip('/')}/_MANIFEST.json"
    put_request: Dict[str, Any] = {
        "Bucket": bucket,
        "Key": manifest_key,
        "Body": body,
        "ContentType": "application/json",
        "ChecksumAlgorithm": "SHA256",
        "ServerSideEncryption": (
            request.canonical_target.server_side_encryption
        ),
        "Metadata": {
            "source-fingerprint": request.fingerprint,
            "manifest-sha256": checksum,
        },
    }
    if request.canonical_target.server_side_encryption in {
        "aws:kms",
        "aws:kms:dsse",
    }:
        put_request["SSEKMSKeyId"] = request.canonical_target.kms_key_id
    client.put_object(**put_request)
    return f"s3://{bucket}/{manifest_key}", checksum


def _jdbc_secret_config(secret_payload: str) -> Dict[str, str]:
    """Convert the managed database secret into bounded Spark JDBC options."""
    try:
        payload = json.loads(secret_payload)
    except json.JSONDecodeError:
        payload = {"database_url": secret_payload}
    database_url = str(payload.get("database_url") or "").strip()
    if database_url:
        parsed = urlparse(database_url.replace("postgres://", "postgresql://", 1))
        if parsed.scheme not in {"postgresql", "postgresql+psycopg"}:
            raise ValueError("Suppression database must be PostgreSQL")
        if not parsed.hostname or not parsed.path.lstrip("/"):
            raise ValueError("Suppression database secret is incomplete")
        return {
            "url": (
                f"jdbc:postgresql://{parsed.hostname}:"
                f"{parsed.port or 5432}/{parsed.path.lstrip('/')}"
            ),
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
        }
    required = {
        "host": payload.get("host"),
        "database": payload.get("dbname") or payload.get("database"),
        "user": payload.get("username") or payload.get("user"),
        "password": payload.get("password"),
    }
    if any(not str(value or "").strip() for value in required.values()):
        raise ValueError("Suppression database secret is incomplete")
    return {
        "url": (
            f"jdbc:postgresql://{required['host']}:"
            f"{int(payload.get('port') or 5432)}/{required['database']}"
        ),
        "user": str(required["user"]),
        "password": str(required["password"]),
    }


def _read_suppression_tokens(
    *,
    spark: Any,
    secrets_client: Any,
    secret_arn: str,
    request: ProviderDistributedJobRequest,
) -> Any:
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError("Suppression database secret is unavailable")
    config = _jdbc_secret_config(str(secret_string))
    for value in (config["user"], config["password"]):
        if not value:
            raise RuntimeError("Suppression database credentials are incomplete")
    tenant = request.contract.tenant_id
    provider = request.contract.provider_id
    dataset = request.contract.dataset_id
    query = (
        "(SELECT subject_token_sha256 "
        "FROM provider_data_rights_requests "
        f"WHERE tenant_id = '{tenant}' "
        f"AND provider_id = '{provider}' "
        f"AND dataset_id = '{dataset}' "
        "AND status = 'applied') AS active_suppression"
    )
    return (
        spark.read.format("jdbc")
        .option("url", config["url"])
        .option("dbtable", query)
        .option("user", config["user"])
        .option("password", config["password"])
        .option("driver", "org.postgresql.Driver")
        .option("sessionInitStatement", f"SET app.tenant_id = '{tenant}'")
        .load()
        .select("subject_token_sha256")
        .dropDuplicates()
    )


def run_job() -> None:
    # Runtime-only imports keep unit tests independent of the Glue image.
    import boto3
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from awsgluedi.transforms import pii
    from pyspark.context import SparkContext
    from pyspark.sql import functions as F
    from pyspark.sql.types import LongType, StructField, StructType
    from pyspark.sql.window import Window

    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "JOB_RUN_ID",
            "request_json",
            "staged_source_ref",
            "canonical_prefix",
            "result_ref",
            "privacy_budget_scope",
            "tokenization_secret_arn",
            "dp_seed_secret_arn",
            "suppression_database_secret_arn",
        ],
    )
    request = ProviderDistributedJobRequest.from_safe_dict(
        json.loads(args["request_json"])
    )
    parse_s3_ref(args["staged_source_ref"])
    canonical_bucket, _ = parse_s3_ref(args["canonical_prefix"])
    result_bucket, _ = parse_s3_ref(args["result_ref"])
    if (
        canonical_bucket != request.canonical_target.bucket
        or result_bucket != request.canonical_target.bucket
    ):
        raise ValueError(
            "Distributed output references do not match the canonical target"
        )

    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)
    s3 = boto3.client("s3")
    secrets = boto3.client("secretsmanager")

    input_rows = 0
    try:
        frame = _read_input(
            spark,
            args["staged_source_ref"],
            request.contract.data_format,
        )
        required = [
            request.contract.entity_id_column,
            request.contract.timestamp_column,
            *request.contract.cohort_columns,
        ]
        if len(frame.columns) != len(set(frame.columns)):
            raise ControlledDistributedPrivacyError(
                "quarantined",
                "duplicate_columns",
            )
        missing = [value for value in required if value not in frame.columns]
        if missing:
            raise ControlledDistributedPrivacyError(
                "quarantined",
                "privacy_required_columns_missing",
            )

        input_rows = int(frame.count())
        if input_rows < 1:
            raise ControlledDistributedPrivacyError(
                "blocked",
                "payload_has_no_rows",
            )
        if input_rows > request.contract.max_rows_per_object:
            raise ControlledDistributedPrivacyError(
                "quarantined",
                "row_limit_exceeded",
            )
        if (
            request.manifest.row_count is not None
            and input_rows != int(request.manifest.row_count)
        ):
            raise ControlledDistributedPrivacyError(
                "quarantined",
                "row_count_mismatch",
            )

        timestamp = "_punk_event_timestamp"
        privacy_window = "privacy_window"
        working = frame.withColumn(
            timestamp,
            F.to_timestamp(F.col(request.contract.timestamp_column)),
        )
        invalid_required = F.col(request.contract.entity_id_column).isNull()
        invalid_required = invalid_required | F.col(timestamp).isNull()
        for column in request.contract.cohort_columns:
            invalid_required = invalid_required | F.col(column).isNull()
        if working.filter(invalid_required).limit(1).count():
            raise ControlledDistributedPrivacyError(
                "quarantined",
                "invalid_required_value",
            )

        if request.contract.distributed_partition_strategy == "entity_hash_v1":
            if request.manifest.delivery_window_id is None:
                raise ControlledDistributedPrivacyError(
                    "quarantined",
                    "privacy_partition_manifest_required",
                )
            expected_partition = F.pmod(
                F.xxhash64(
                    F.col(request.contract.entity_id_column).cast("string")
                ),
                F.lit(int(request.manifest.partition_count)),
            )
            wrong_partition = expected_partition != F.lit(
                int(request.manifest.partition_index)
            )
            if working.filter(wrong_partition).limit(1).count():
                raise ControlledDistributedPrivacyError(
                    "quarantined",
                    "privacy_partition_membership_mismatch",
                )

        hashed = pii.CryptographicHash.apply(
            data_frame=working,
            spark_context=sc,
            source_columns=[request.contract.entity_id_column],
            secret_id=args["tokenization_secret_arn"],
            algorithm="HMAC_SHA256",
            output_format="BASE64",
        )
        hashed_column = f"{request.contract.entity_id_column}_hashed"
        if hashed_column not in hashed.columns:
            raise RuntimeError("Glue HMAC transform did not produce its output")
        working = hashed.drop(request.contract.entity_id_column).withColumn(
            privacy_window,
            F.to_date(F.col(timestamp)),
        )
        working = working.withColumn(
            "_punk_subject_token_sha256",
            F.sha2(F.col(hashed_column).cast("string"), 256),
        )
        suppression = _read_suppression_tokens(
            spark=spark,
            secrets_client=secrets,
            secret_arn=args["suppression_database_secret_arn"],
            request=request,
        )
        working = working.join(
            suppression,
            working["_punk_subject_token_sha256"]
            == suppression["subject_token_sha256"],
            "left_anti",
        ).drop("_punk_subject_token_sha256")

        bound_partition = Window.partitionBy(
            hashed_column,
            privacy_window,
            *request.contract.cohort_columns,
        ).orderBy(F.col(timestamp).asc())
        bounded = (
            working.withColumn(
                "_punk_contribution_rank",
                F.row_number().over(bound_partition),
            )
            .filter(F.col("_punk_contribution_rank") == 1)
        )

        group_columns = [
            *request.contract.cohort_columns,
            privacy_window,
        ]
        aggregate = (
            bounded.groupBy(*group_columns)
            .agg(F.count(F.lit(1)).alias("_punk_bounded_count"))
            .filter(
                F.col("_punk_bounded_count")
                >= F.lit(request.contract.min_cohort_size)
            )
        )
        safe_count = int(aggregate.count())
        if safe_count < 1:
            raise ControlledDistributedPrivacyError(
                "blocked",
                "k_anonymity_threshold_not_met",
            )

        aggregate = aggregate.withColumn(
            "cohort_key",
            F.concat_ws(
                "|",
                *[
                    F.coalesce(F.col(value).cast("string"), F.lit("unknown"))
                    for value in group_columns
                ],
            ),
        )
        seeded = pii.CryptographicHash.apply(
            data_frame=aggregate,
            spark_context=sc,
            source_columns=["cohort_key"],
            secret_id=args["dp_seed_secret_arn"],
            algorithm="HMAC_SHA256",
            output_format="BASE64",
        )
        if "cohort_key_hashed" not in seeded.columns:
            raise RuntimeError("Glue DP seed transform did not produce output")

        private_count_schema = StructType(
            [
                StructField("private_count", LongType(), nullable=False),
                StructField("sigma_micros", LongType(), nullable=False),
            ]
        )

        @F.udf(returnType=private_count_schema)
        def private_count_udf(count: int, keyed_seed: str):
            private_count, sigma = gaussian_private_count(
                int(count),
                keyed_seed=str(keyed_seed),
                fingerprint=request.fingerprint,
                epsilon=request.contract.epsilon,
                delta=request.contract.delta,
                sensitivity=request.contract.sensitivity,
            )
            return (private_count, int(round(sigma * 1_000_000)))

        output = (
            seeded.withColumn(
                "_punk_dp",
                private_count_udf(
                    F.col("_punk_bounded_count"),
                    F.col("cohort_key_hashed"),
                ),
            )
            .withColumn(
                "dp_noisy_count",
                F.col("_punk_dp.private_count"),
            )
            .withColumn(
                "dp_sigma",
                F.col("_punk_dp.sigma_micros") / F.lit(1_000_000.0),
            )
            .withColumn("dp_epsilon", F.lit(request.contract.epsilon))
            .withColumn("dp_delta", F.lit(request.contract.delta))
            .withColumn("dp_mechanism", F.lit("gaussian"))
            .withColumn(
                "tenant_id",
                F.lit(request.contract.tenant_id),
            )
            .withColumn(
                "provider_id",
                F.lit(request.contract.provider_id),
            )
            .withColumn(
                "dataset_id",
                F.lit(request.contract.dataset_id),
            )
            .withColumn(
                "schema_version",
                F.lit(request.contract.schema_version),
            )
            .withColumn(
                "source_fingerprint",
                F.lit(request.fingerprint),
            )
            .withColumn(
                "rights_policy_id",
                F.lit(request.manifest.rights_policy_id),
            )
            .withColumn(
                "processing_purpose",
                F.lit(request.manifest.purpose),
            )
            .withColumn(
                "delivery_window_id",
                F.lit(
                    request.manifest.delivery_window_id
                    or f"single_{request.fingerprint}"
                ),
            )
            .withColumn(
                "partition_index",
                F.lit(int(request.manifest.partition_index or 0)),
            )
            .withColumn(
                "partition_count",
                F.lit(int(request.manifest.partition_count or 1)),
            )
            .withColumn(
                "delivery_type",
                F.lit(request.manifest.delivery_type),
            )
            .drop(
                "_punk_bounded_count",
                "_punk_dp",
                "cohort_key_hashed",
            )
        )
        forbidden = {
            request.contract.entity_id_column,
            hashed_column,
            request.contract.timestamp_column,
            timestamp,
        }
        output = output.drop(*[value for value in forbidden if value in output.columns])

        attempt_ref = (
            f"{args['canonical_prefix'].rstrip('/')}/attempts/"
            f"{request.manifest.delivery_window_id or 'single-object'}/"
            f"partition={request.manifest.partition_index or 0}/"
            f"{args['JOB_RUN_ID']}/"
        )
        (
            output.write.mode("errorifexists")
            .partitionBy(privacy_window)
            .parquet(attempt_ref)
        )
        manifest_ref, manifest_checksum = _write_canonical_manifest(
            client=s3,
            request=request,
            canonical_ref=attempt_ref,
            input_rows=input_rows,
            output_rows=safe_count,
        )
        result = ProviderDistributedJobResult(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            status="completed",
            input_rows=input_rows,
            output_rows=safe_count,
            privacy_job_id=args["JOB_RUN_ID"],
            output_manifest_ref=manifest_ref,
            canonical_ref=attempt_ref,
            canonical_checksum_sha256=manifest_checksum,
            privacy_controls=PRIVACY_CONTROLS,
        )
        _write_result(
            client=s3,
            request=request,
            result_ref=args["result_ref"],
            result=result,
        )
        job.commit()
    except ControlledDistributedPrivacyError as exc:
        result = ProviderDistributedJobResult(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            status=exc.status,
            input_rows=input_rows,
            output_rows=0,
            privacy_job_id=args["JOB_RUN_ID"],
            output_manifest_ref=args["result_ref"],
            reason_code=exc.reason_code,
            privacy_controls=PRIVACY_CONTROLS,
        )
        _write_result(
            client=s3,
            request=request,
            result_ref=args["result_ref"],
            result=result,
        )
        job.commit()
    except Exception:
        result = ProviderDistributedJobResult(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            status="failed",
            input_rows=input_rows,
            output_rows=0,
            privacy_job_id=args["JOB_RUN_ID"],
            output_manifest_ref=args["result_ref"],
            reason_code="distributed_privacy_job_failed",
            privacy_controls=PRIVACY_CONTROLS,
        )
        try:
            _write_result(
                client=s3,
                request=request,
                result_ref=args["result_ref"],
                result=result,
            )
        finally:
            raise


if __name__ == "__main__":
    run_job()
