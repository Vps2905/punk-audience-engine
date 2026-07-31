from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd

from app.core.schema_validator import SafeSchemaValidator
from app.models.audience_feature_contracts import (
    CanonicalAudienceFeature,
    CanonicalFeatureSet,
    normalize_taxonomy_value,
    stable_digest,
    validate_embedding,
)
from app.models.production_feature_build_contracts import (
    EmbeddingModelSpec,
    ProductionFeatureBuildRequest,
)


class FeatureBatchEncoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray: ...


class SentenceTransformerFeatureBatchEncoder:
    """
    Pinned sentence-transformer inference for privacy-safe feature text.

    Model name and immutable revision are supplied by the build contract.
    Production never falls back to hashing or an unversioned model.
    """

    def __init__(self, *, device: str | None = None) -> None:
        self._device = (
            device
            or os.getenv("FEATURE_EMBEDDING_DEVICE")
            or "cpu"
        )
        self._models: dict[tuple[str, str, str], Any] = {}

    def encode(
        self,
        texts: Sequence[str],
        *,
        model: EmbeddingModelSpec,
        batch_size: int,
    ) -> np.ndarray:
        if model.backend != "sentence_transformers":
            raise RuntimeError(
                "The configured encoder cannot serve this embedding backend."
            )
        from sentence_transformers import SentenceTransformer

        key = (model.model_name, model.model_revision, self._device)
        loaded = self._models.get(key)
        if loaded is None:
            loaded = SentenceTransformer(
                model.model_name,
                revision=model.model_revision,
                device=self._device,
                trust_remote_code=False,
            )
            self._models[key] = loaded
        prefixed = [
            f"{model.document_prefix}{text}"
            for text in texts
        ]
        vectors = loaded.encode(
            prefixed,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=model.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class ProductionCanonicalFeatureEmbeddingService:
    """
    Canonical privacy-safe rows -> versioned semantic feature set.

    This boundary accepts only already aggregated canonical records. It never
    accepts, embeds, stores, or returns entity identifiers or raw observations.
    Module 2 output is deliberately not activation-eligible; quality scoring,
    cohort lifecycle checks, approval, and activation remain downstream.
    """

    def __init__(
        self,
        *,
        encoder: FeatureBatchEncoder | None = None,
        stale_after_hours: int | None = None,
        now_fn: Any = None,
    ) -> None:
        self._encoder = encoder or SentenceTransformerFeatureBatchEncoder()
        self._stale_after_hours = int(
            stale_after_hours
            or os.getenv("DATA_FRESHNESS_STALE_HOURS", "48")
        )
        if self._stale_after_hours < 1:
            raise ValueError("stale_after_hours must be at least 1.")
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._schema_validator = SafeSchemaValidator()

    def build(
        self,
        request: ProductionFeatureBuildRequest,
        rows: Sequence[dict[str, Any]],
    ) -> CanonicalFeatureSet:
        source = request.source
        if len(rows) != int(source.expected_row_count):
            raise ValueError(
                "Canonical row count does not match the source manifest."
            )
        if len(rows) > int(request.max_features):
            raise ValueError(
                "Canonical feature count exceeds the bounded build limit."
            )
        frame = pd.DataFrame([dict(row) for row in rows])
        self._schema_validator.validate_safe_cohort_dataframe(
            frame,
            context="production canonical feature input",
        )
        required_columns = {
            source.location_column,
            source.category_column,
            source.cohort_size_column,
            *request.metadata_fields,
        }
        for optional_column in (
            source.daypart_column,
            source.lookback_bucket_column,
            source.privacy_window_column,
        ):
            if optional_column:
                required_columns.add(optional_column)
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(
                "Canonical feature input is missing mapped columns: "
                + ", ".join(missing)
            )
        self._validate_provenance(frame, request)

        projections = [
            self._project_row(request, dict(row))
            for row in rows
        ]
        self._assert_unique_feature_keys(projections)
        projections = sorted(
            projections,
            key=lambda value: json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
        canonical_content_fingerprint = stable_digest(
            {
                "rows": projections,
            }
        )
        texts = [projection["trait_text"] for projection in projections]
        vectors = np.asarray(
            self._encoder.encode(
                texts,
                model=request.model,
                batch_size=request.batch_size,
            ),
            dtype=np.float64,
        )
        self._validate_matrix(
            vectors,
            row_count=len(projections),
            dimension=request.model.dimension,
        )
        embedding_content_fingerprint = self._embedding_digest(vectors)
        build_fingerprint = stable_digest(
            {
                "request_fingerprint": request.request_fingerprint,
                "canonical_content_fingerprint": (
                    canonical_content_fingerprint
                ),
                "embedding_content_fingerprint": (
                    embedding_content_fingerprint
                ),
            }
        )
        feature_set_id = "feature_set_" + build_fingerprint[:24]
        freshness_status = self._freshness_status(source.source_latest_at)

        features = [
            self._feature(
                request=request,
                projection=projection,
                embedding=vectors[position],
                feature_set_id=feature_set_id,
                build_fingerprint=build_fingerprint,
                freshness_status=freshness_status,
            )
            for position, projection in enumerate(projections)
        ]
        return CanonicalFeatureSet(
            tenant_id=source.tenant_id,
            feature_set_id=feature_set_id,
            version=1,
            status="ready_for_quality_scoring",
            source_mode="canonical_s3_privacy_safe_features",
            data_use_mode=source.data_use_mode,
            source_ref=source.source_ref,
            source_version=source.source_version,
            # Migration 0004 has a tenant/source uniqueness constraint. The
            # build fingerprint combines the immutable source, model and
            # actual safe content while lineage preserves the source digest.
            source_fingerprint=build_fingerprint,
            source_latest_at=source.source_latest_at,
            freshness_status=freshness_status,
            stale_after_hours=self._stale_after_hours,
            model_backend=request.model.backend,
            model_name=request.model.model_name,
            model_version=request.model.model_revision,
            embedding_dimension=request.model.dimension,
            privacy_policy_version=source.privacy_policy_version,
            rights_policy_id=source.rights_policy_id,
            purpose=source.purpose,
            eligible_for_retrieval=True,
            eligible_for_activation=False,
            feature_count=len(features),
            lineage={
                "feature_build_contract_version": request.contract_version,
                "feature_build_id": request.build_id,
                "request_fingerprint": request.request_fingerprint,
                "canonical_source_fingerprint": source.source_fingerprint,
                "canonical_content_fingerprint": (
                    canonical_content_fingerprint
                ),
                "embedding_model_fingerprint": request.model.fingerprint,
                "embedding_model_spec": request.model.to_safe_dict(),
                "embedding_content_fingerprint": (
                    embedding_content_fingerprint
                ),
                "provider_id": source.provider_id,
                "dataset_id": source.dataset_id,
                "schema_version": source.schema_version,
                "privacy_controls": list(source.privacy_controls),
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "raw_observations_read": False,
                "quality_status": "not_scored",
                "activation_blocked_until_module3": True,
                "transformations": [
                    "validate_privacy_safe_canonical_schema",
                    "validate_source_and_policy_provenance",
                    "normalize_global_feature_taxonomy",
                    "create_model_input_trait_text",
                    "encode_with_pinned_model_revision",
                    "validate_embedding_shape_and_finiteness",
                    "assign_content_addressed_feature_identity",
                ],
            },
            features=features,
        )

    def _validate_provenance(
        self,
        frame: pd.DataFrame,
        request: ProductionFeatureBuildRequest,
    ) -> None:
        source = request.source
        expected = {
            "tenant_id": source.tenant_id,
            "provider_id": source.provider_id,
            "dataset_id": source.dataset_id,
            "schema_version": source.schema_version,
            "source_fingerprint": source.source_fingerprint,
            "rights_policy_id": source.rights_policy_id,
            "processing_purpose": source.purpose,
        }
        for column, expected_value in expected.items():
            if column not in frame.columns:
                raise ValueError(
                    f"Canonical feature input is missing provenance column: {column}"
                )
            observed = {
                normalize_taxonomy_value(value)
                if column != "source_fingerprint"
                else str(value or "").strip().lower()
                for value in frame[column].dropna().tolist()
            }
            normalized_expected = (
                expected_value
                if column == "source_fingerprint"
                else normalize_taxonomy_value(expected_value)
            )
            if observed != {normalized_expected}:
                raise ValueError(
                    f"Canonical provenance mismatch for {column}."
                )

    def _project_row(
        self,
        request: ProductionFeatureBuildRequest,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        source = request.source
        location = normalize_taxonomy_value(row.get(source.location_column))
        category = normalize_taxonomy_value(row.get(source.category_column))
        daypart = (
            normalize_taxonomy_value(row.get(source.daypart_column))
            if source.daypart_column
            else "all_day"
        )
        if not location or not category or not daypart:
            raise ValueError(
                "Canonical features require location, category and daypart values."
            )
        cohort_size = self._nonnegative_integer(
            row.get(source.cohort_size_column),
            source.cohort_size_column,
        )
        lookback_bucket = (
            normalize_taxonomy_value(
                row.get(source.lookback_bucket_column)
            )
            if source.lookback_bucket_column
            else None
        ) or None
        privacy_window = (
            str(row.get(source.privacy_window_column) or "").strip()
            if source.privacy_window_column
            else ""
        )
        metadata = {
            field: self._safe_scalar(row.get(field))
            for field in request.metadata_fields
        }
        trait_parts = [
            f"location {location.replace('_', ' ')}",
            f"category {category.replace('_', ' ')}",
            f"daypart {daypart.replace('_', ' ')}",
        ]
        if lookback_bucket:
            trait_parts.append(
                f"lookback {lookback_bucket.replace('_', ' ')}"
            )
        for field in sorted(metadata):
            normalized_value = normalize_taxonomy_value(metadata[field])
            if normalized_value:
                trait_parts.append(
                    f"{field.replace('_', ' ')} "
                    f"{normalized_value.replace('_', ' ')}"
                )
        return {
            "location_name": location,
            "primary_poi_type": category,
            "created_day_part": daypart,
            "lookback_bucket": lookback_bucket,
            "privacy_window": privacy_window,
            "cohort_size": cohort_size,
            "metadata": metadata,
            "trait_text": " ".join(trait_parts),
        }

    def _feature(
        self,
        *,
        request: ProductionFeatureBuildRequest,
        projection: dict[str, Any],
        embedding: Sequence[float],
        feature_set_id: str,
        build_fingerprint: str,
        freshness_status: str,
    ) -> CanonicalAudienceFeature:
        source = request.source
        feature_identity = {
            "build_fingerprint": build_fingerprint,
            "location_name": projection["location_name"],
            "primary_poi_type": projection["primary_poi_type"],
            "created_day_part": projection["created_day_part"],
            "lookback_bucket": projection["lookback_bucket"],
            "privacy_window": projection["privacy_window"],
            "metadata": projection["metadata"],
        }
        feature_id = "feature_" + stable_digest(feature_identity)[:24]
        safe_metadata = {
            **projection["metadata"],
            "provider_id": source.provider_id,
            "dataset_id": source.dataset_id,
            "schema_version": source.schema_version,
            "privacy_window": projection["privacy_window"],
            "privacy_controls": list(source.privacy_controls),
            "quality_status": "not_scored",
            "canonical_source_fingerprint": source.source_fingerprint,
        }
        return CanonicalAudienceFeature(
            tenant_id=source.tenant_id,
            feature_set_id=feature_set_id,
            feature_set_version=1,
            feature_id=feature_id,
            location_name=projection["location_name"],
            primary_poi_type=projection["primary_poi_type"],
            created_day_part=projection["created_day_part"],
            lookback_bucket=projection["lookback_bucket"],
            cohort_size=projection["cohort_size"],
            quality_score=0.0,
            privacy_status=source.privacy_status,
            rights_status=source.rights_status,
            purpose=source.purpose,
            source_latest_at=source.source_latest_at,
            freshness_status=freshness_status,
            data_use_mode=source.data_use_mode,
            eligible_for_retrieval=True,
            eligible_for_activation=False,
            trait_text=projection["trait_text"],
            embedding=validate_embedding(
                embedding,
                expected_dimension=request.model.dimension,
            ),
            metadata=safe_metadata,
        )

    def _validate_matrix(
        self,
        vectors: np.ndarray,
        *,
        row_count: int,
        dimension: int,
    ) -> None:
        if vectors.shape != (row_count, dimension):
            raise ValueError(
                "Embedding output shape does not match the canonical feature build."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding output contains non-finite values.")
        magnitudes = np.linalg.norm(vectors, axis=1)
        if np.any(np.isclose(magnitudes, 0.0)):
            raise ValueError("Embedding output contains a zero-magnitude vector.")

    def _assert_unique_feature_keys(
        self,
        projections: Sequence[dict[str, Any]],
    ) -> None:
        seen: set[str] = set()
        for projection in projections:
            key = stable_digest(
                {
                    "location_name": projection["location_name"],
                    "primary_poi_type": projection["primary_poi_type"],
                    "created_day_part": projection["created_day_part"],
                    "lookback_bucket": projection["lookback_bucket"],
                    "privacy_window": projection["privacy_window"],
                    "metadata": projection["metadata"],
                }
            )
            if key in seen:
                raise ValueError(
                    "Canonical input contains a duplicate feature identity."
                )
            seen.add(key)

    def _embedding_digest(self, vectors: np.ndarray) -> str:
        normalized = np.asarray(vectors, dtype="<f4", order="C")
        return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()

    def _freshness_status(self, source_latest_at: datetime) -> str:
        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        age_hours = max(
            (
                now.astimezone(timezone.utc)
                - source_latest_at.astimezone(timezone.utc)
            ).total_seconds()
            / 3600.0,
            0.0,
        )
        return (
            "fresh"
            if age_hours <= self._stale_after_hours
            else "stale"
        )

    def _nonnegative_integer(self, value: Any, label: str) -> int:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.") from None
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"{label} must be finite and nonnegative.")
        return int(round(parsed))

    def _safe_scalar(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "item"):
            return self._safe_scalar(value.item())
        if hasattr(value, "isoformat"):
            return value.isoformat()
        raise ValueError("Canonical metadata fields must contain scalar values.")
