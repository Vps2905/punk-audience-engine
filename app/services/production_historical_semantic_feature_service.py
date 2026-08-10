from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.schema_validator import SafeSchemaValidator
from app.models.audience_feature_contracts import (
    NON_ACTIVATABLE_DATA_MODES,
    SAFE_PRIVACY_STATUSES,
    CanonicalAudienceFeature,
    CanonicalFeatureSet,
    normalize_taxonomy_value,
    stable_digest,
    validate_embedding,
)
from app.models.production_historical_semantic_feature_contracts import (
    HistoricalSemanticFeatureBuildRequest,
)
from app.services.production_feature_embedding_service import (
    FeatureBatchEncoder,
    SentenceTransformerFeatureBatchEncoder,
)


_SAFE_RIGHTS_STATUSES = {
    "permitted",
    "historical_internal_only",
    "offline_evaluation_only",
}


class ApprovedModelRegistry(Protocol):
    def require_approved(self, *, tenant_id: str, model: Any) -> dict[str, Any]: ...


class HistoricalFeatureWriter(Protocol):
    def save_feature_set(self, feature_set: CanonicalFeatureSet) -> dict[str, Any]: ...


class ProductionHistoricalSafeFeatureReader:
    """Read safe traits without stored vectors or individual identifiers."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine

    def read(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> dict[str, Any]:
        clean_tenant = normalize_taxonomy_value(tenant_id)
        if not clean_tenant:
            raise ValueError("tenant_id is required.")
        engine = self._engine()
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.execute(
                    text(
                        """
                        SELECT set_config(
                            'app.tenant_id',
                            :tenant_id,
                            true
                        )
                        """
                    ),
                    {"tenant_id": clean_tenant},
                )
                feature_set = connection.execute(
                    text(
                        """
                        SELECT
                            tenant_id,
                            feature_set_id,
                            version,
                            status,
                            source_mode,
                            data_use_mode,
                            source_ref,
                            source_version,
                            source_fingerprint,
                            source_latest_at,
                            freshness_status,
                            stale_after_hours,
                            privacy_policy_version,
                            rights_policy_id,
                            purpose,
                            eligible_for_retrieval,
                            eligible_for_activation,
                            feature_count,
                            lineage
                        FROM audience_feature_sets
                        WHERE tenant_id = :tenant_id
                          AND feature_set_id = :feature_set_id
                          AND version = :feature_set_version
                        """
                    ),
                    {
                        "tenant_id": clean_tenant,
                        "feature_set_id": feature_set_id,
                        "feature_set_version": int(feature_set_version),
                    },
                ).mappings().first()
                if not feature_set:
                    raise FileNotFoundError(
                        "Historical source feature set was not found."
                    )
                rows = connection.execute(
                    text(
                        """
                        SELECT
                            feature_id,
                            location_name,
                            primary_poi_type,
                            created_day_part,
                            lookback_bucket,
                            cohort_size,
                            quality_score,
                            privacy_status,
                            rights_status,
                            purpose,
                            source_latest_at,
                            freshness_status,
                            data_use_mode,
                            eligible_for_retrieval,
                            eligible_for_activation,
                            trait_text,
                            metadata
                        FROM audience_feature_vectors
                        WHERE tenant_id = :tenant_id
                          AND feature_set_id = :feature_set_id
                          AND feature_set_version = :feature_set_version
                        ORDER BY feature_id
                        """
                    ),
                    {
                        "tenant_id": clean_tenant,
                        "feature_set_id": feature_set_id,
                        "feature_set_version": int(feature_set_version),
                    },
                ).mappings().all()
            finally:
                transaction.rollback()
        return {
            "feature_set": dict(feature_set),
            "features": [dict(row) for row in rows],
            "read_only_transaction_verified": True,
            "stored_embeddings_read": False,
            "raw_identifiers_read": False,
        }

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        database_url = (
            self._database_url
            or os.getenv("AUDIENCE_FEATURE_DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError("AUDIENCE_FEATURE_DATABASE_URL is required.")
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)


class ProductionHistoricalSemanticFeatureService:
    """Re-embed only existing privacy-safe historical cohort traits."""

    def __init__(
        self,
        *,
        source_reader: Any,
        model_registry: ApprovedModelRegistry,
        feature_writer: HistoricalFeatureWriter,
        encoder: FeatureBatchEncoder | None = None,
    ) -> None:
        self._source_reader = source_reader
        self._model_registry = model_registry
        self._feature_writer = feature_writer
        self._encoder = encoder or SentenceTransformerFeatureBatchEncoder()
        self._schema_validator = SafeSchemaValidator()

    def build(
        self,
        request: HistoricalSemanticFeatureBuildRequest,
        *,
        privacy_safe_historical_reembedding_confirmed: bool,
    ) -> dict[str, Any]:
        if not privacy_safe_historical_reembedding_confirmed:
            raise RuntimeError(
                "Explicit privacy-safe historical re-embedding confirmation "
                "is required."
            )
        approval = self._model_registry.require_approved(
            tenant_id=request.tenant_id,
            model=request.model,
        )
        if approval.get("approved") is not True:
            raise RuntimeError("Embedding model is not approved.")
        snapshot = self._source_reader.read(
            tenant_id=request.tenant_id,
            feature_set_id=request.source_feature_set_id,
            feature_set_version=request.source_feature_set_version,
        )
        source, rows = self._validate_snapshot(snapshot, request)
        vectors = self._encode(rows, request)
        feature_set = self._build_feature_set(
            source=source,
            rows=rows,
            vectors=vectors,
            request=request,
        )
        receipt = self._feature_writer.save_feature_set(feature_set)
        return {
            "status": "historical_semantic_features_ready",
            "tenant_id": request.tenant_id,
            "request_fingerprint": request.request_fingerprint,
            "source_feature_set_id": request.source_feature_set_id,
            "source_feature_set_version": request.source_feature_set_version,
            "semantic_feature_set_id": feature_set.feature_set_id,
            "semantic_feature_set_version": feature_set.version,
            "feature_count": feature_set.feature_count,
            "model_fingerprint": request.model.fingerprint,
            "model_name": request.model.model_name,
            "model_revision": request.model.model_revision,
            "freshness_status": feature_set.freshness_status,
            "storage_receipt": dict(receipt),
            "eligible_for_module5_functional_shadow": True,
            "eligible_for_activation": False,
            "production_ready": False,
            "live_cutover_authorized": False,
            "stored_source_embeddings_read": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "activation_or_export_performed": False,
        }

    def _validate_snapshot(
        self,
        snapshot: Mapping[str, Any],
        request: HistoricalSemanticFeatureBuildRequest,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        source = dict(snapshot.get("feature_set") or {})
        rows = [dict(row) for row in snapshot.get("features") or []]
        if snapshot.get("read_only_transaction_verified") is not True:
            raise RuntimeError("Historical source read-only mode was not verified.")
        if snapshot.get("stored_embeddings_read") is not False:
            raise RuntimeError("Historical source embeddings must not be read.")
        if snapshot.get("raw_identifiers_read") is not False:
            raise RuntimeError("Historical source identifiers must not be read.")
        if source.get("tenant_id") != request.tenant_id:
            raise ValueError("Historical feature tenant lineage mismatch.")
        if source.get("feature_set_id") != request.source_feature_set_id:
            raise ValueError("Historical feature-set lineage mismatch.")
        if int(source.get("version") or 0) != request.source_feature_set_version:
            raise ValueError("Historical feature-set version mismatch.")
        if source.get("data_use_mode") not in NON_ACTIVATABLE_DATA_MODES:
            raise RuntimeError("Historical re-embedding requires non-activatable data.")
        if source.get("eligible_for_retrieval") is not True:
            raise RuntimeError("Historical source is not retrieval-eligible.")
        if source.get("eligible_for_activation") is not False:
            raise RuntimeError("Activation-eligible sources are not accepted here.")
        if not rows or len(rows) != int(source.get("feature_count") or 0):
            raise RuntimeError("Historical safe feature count does not match lineage.")
        frame = pd.DataFrame(rows)
        self._schema_validator.validate_safe_cohort_dataframe(
            frame,
            context="historical semantic re-embedding input",
        )
        for row in rows:
            if (
                normalize_taxonomy_value(row.get("privacy_status"))
                not in SAFE_PRIVACY_STATUSES
            ):
                raise RuntimeError("Historical feature row is not privacy-safe.")
            if (
                normalize_taxonomy_value(row.get("rights_status"))
                not in _SAFE_RIGHTS_STATUSES
            ):
                raise RuntimeError("Historical feature rights are not permitted.")
            if row.get("data_use_mode") not in NON_ACTIVATABLE_DATA_MODES:
                raise RuntimeError("Historical feature row has an unsafe data mode.")
            if row.get("data_use_mode") != source.get("data_use_mode"):
                raise RuntimeError("Historical feature data-mode lineage mismatch.")
            if row.get("freshness_status") != source.get("freshness_status"):
                raise RuntimeError("Historical feature freshness lineage mismatch.")
            if normalize_taxonomy_value(row.get("purpose")) != (
                normalize_taxonomy_value(source.get("purpose"))
            ):
                raise RuntimeError("Historical feature purpose lineage mismatch.")
            if row.get("eligible_for_retrieval") is not True:
                raise RuntimeError("Historical feature row is not retrieval-eligible.")
            if row.get("eligible_for_activation") is not False:
                raise RuntimeError("Historical feature row is activation-eligible.")
            if int(row.get("cohort_size") or 0) < request.minimum_cohort_size:
                raise RuntimeError("Historical feature row is below k-anonymity.")
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise TypeError("Historical feature metadata must be an object.")
            self._validate_metadata_keys(metadata)
        return source, rows

    def _validate_metadata_keys(self, value: Mapping[str, Any]) -> None:
        stack: list[Mapping[str, Any]] = [value]
        keys: list[str] = []
        while stack:
            current = stack.pop()
            for key, child in current.items():
                keys.append(str(key))
                if isinstance(child, Mapping):
                    stack.append(child)
                elif isinstance(child, list):
                    stack.extend(
                        item for item in child if isinstance(item, Mapping)
                    )
        self._schema_validator.validate_no_blocked_columns(
            keys,
            context="historical feature metadata",
        )

    def _encode(
        self,
        rows: Sequence[Mapping[str, Any]],
        request: HistoricalSemanticFeatureBuildRequest,
    ) -> np.ndarray:
        texts = [" ".join(str(row.get("trait_text") or "").split()) for row in rows]
        if any(not value for value in texts):
            raise ValueError("Every historical feature requires safe trait_text.")
        vectors = np.asarray(
            self._encoder.encode(
                texts,
                model=request.model,
                batch_size=request.batch_size,
            ),
            dtype=np.float64,
        )
        expected = (len(rows), request.model.dimension)
        if vectors.shape != expected:
            raise RuntimeError(
                f"Historical embedding matrix must have shape {expected}."
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("Historical embedding matrix is not finite.")
        if np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise RuntimeError("Historical embedding matrix contains zero vectors.")
        return vectors

    def _build_feature_set(
        self,
        *,
        source: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        vectors: np.ndarray,
        request: HistoricalSemanticFeatureBuildRequest,
    ) -> CanonicalFeatureSet:
        row_manifest = [
            {
                "feature_id": row["feature_id"],
                "location_name": row["location_name"],
                "primary_poi_type": row["primary_poi_type"],
                "created_day_part": row["created_day_part"],
                "lookback_bucket": row.get("lookback_bucket"),
                "cohort_size": int(row["cohort_size"]),
                "quality_score": float(row["quality_score"]),
                "trait_text": row["trait_text"],
                "metadata": dict(row.get("metadata") or {}),
            }
            for row in rows
        ]
        embedding_digest = stable_digest(
            {"vectors": np.asarray(vectors, dtype=np.float32).tolist()}
        )
        build_fingerprint = stable_digest(
            {
                "contract_version": request.contract_version,
                "request": request.to_safe_dict(),
                "source_fingerprint": source["source_fingerprint"],
                "rows": row_manifest,
                "embedding_digest": embedding_digest,
            }
        )
        feature_set_id = "feature_set_" + build_fingerprint[:24]
        features = []
        for position, row in enumerate(rows):
            feature_id = "feature_" + stable_digest(
                {
                    "build_fingerprint": build_fingerprint,
                    "source_feature_id": row["feature_id"],
                }
            )[:24]
            embedding = validate_embedding(
                vectors[position],
                expected_dimension=request.model.dimension,
            )
            features.append(
                CanonicalAudienceFeature(
                    tenant_id=request.tenant_id,
                    feature_set_id=feature_set_id,
                    feature_set_version=1,
                    feature_id=feature_id,
                    location_name=row["location_name"],
                    primary_poi_type=row["primary_poi_type"],
                    created_day_part=row["created_day_part"],
                    lookback_bucket=row.get("lookback_bucket"),
                    cohort_size=int(row["cohort_size"]),
                    quality_score=float(row["quality_score"]),
                    privacy_status=normalize_taxonomy_value(
                        row["privacy_status"]
                    ),
                    rights_status=normalize_taxonomy_value(row["rights_status"]),
                    purpose=normalize_taxonomy_value(row["purpose"]),
                    source_latest_at=row.get("source_latest_at"),
                    freshness_status=row["freshness_status"],
                    data_use_mode=row["data_use_mode"],
                    eligible_for_retrieval=True,
                    eligible_for_activation=False,
                    trait_text=row["trait_text"],
                    embedding=embedding,
                    metadata={
                        **dict(row.get("metadata") or {}),
                        "historical_semantic_reembedding": True,
                        "source_feature_set_fingerprint": source[
                            "source_fingerprint"
                        ],
                    },
                )
            )
        return CanonicalFeatureSet(
            tenant_id=request.tenant_id,
            feature_set_id=feature_set_id,
            version=1,
            status="ready_for_historical_preview",
            source_mode="historical_safe_semantic_reembedding",
            data_use_mode=source["data_use_mode"],
            source_ref=source["source_ref"],
            source_version=(
                f"{source['source_version']}:semantic:"
                f"{request.model.fingerprint[:16]}"
            ),
            source_fingerprint=build_fingerprint,
            source_latest_at=source.get("source_latest_at"),
            freshness_status=source["freshness_status"],
            stale_after_hours=int(source["stale_after_hours"]),
            model_backend=request.model.backend,
            model_name=request.model.model_name,
            model_version=request.model.model_revision,
            embedding_dimension=request.model.dimension,
            privacy_policy_version=source["privacy_policy_version"],
            rights_policy_id=source["rights_policy_id"],
            purpose=source["purpose"],
            eligible_for_retrieval=True,
            eligible_for_activation=False,
            feature_count=len(features),
            lineage={
                "historical_semantic_feature_contract_version": (
                    request.contract_version
                ),
                "request_fingerprint": request.request_fingerprint,
                "source_feature_set_id": request.source_feature_set_id,
                "source_feature_set_version": (
                    request.source_feature_set_version
                ),
                "source_feature_set_fingerprint": source[
                    "source_fingerprint"
                ],
                "embedding_model_fingerprint": request.model.fingerprint,
                "embedding_model_spec": request.model.to_safe_dict(),
                "embedding_content_fingerprint": embedding_digest,
                "minimum_cohort_size": request.minimum_cohort_size,
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "raw_observations_read": False,
                "stored_source_embeddings_read": False,
                "activation_blocked": True,
                "production_routing_enabled": False,
            },
            features=features,
        )
