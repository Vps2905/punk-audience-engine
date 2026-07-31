from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
    EmbeddingModelSpec,
    ProductionFeatureBuildRequest,
)
from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)
from app.services.durable_production_feature_build_service import (
    DurableProductionFeatureBuildService,
)
from app.services.production_canonical_feature_source_service import (
    ProductionCanonicalFeatureSourceService,
)
from app.services.production_embedding_model_registry_service import (
    ProductionEmbeddingModelRegistryService,
)
from app.services.production_feature_build_pipeline_service import (
    ProductionFeatureBuildPipelineService,
)
from app.services.production_feature_build_state_service import (
    ProductionFeatureBuildStateService,
)
from app.services.production_feature_embedding_service import (
    ProductionCanonicalFeatureEmbeddingService,
)
from app.services.provider_object_store_service import S3ProviderObjectStore


ROOT = Path(__file__).resolve().parents[1]


def load_request(path: Path) -> ProductionFeatureBuildRequest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    source_payload = dict(payload.pop("source"))
    model_payload = dict(payload.pop("model"))
    for key in ("privacy_controls",):
        if key in source_payload:
            source_payload[key] = tuple(source_payload[key])
    if "metadata_fields" in payload:
        payload["metadata_fields"] = tuple(payload["metadata_fields"])
    return ProductionFeatureBuildRequest(
        source=CanonicalFeatureSourceManifest(**source_payload),
        model=EmbeddingModelSpec(**model_payload),
        **payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one bounded privacy-safe canonical S3 feature partition "
            "with an approved immutable embedding model."
        )
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--confirm-privacy-safe-canonical-input",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_privacy_safe_canonical_input:
        raise SystemExit(
            "STOP: explicit privacy-safe canonical input confirmation "
            "is required"
        )

    load_dotenv(ROOT / ".env", override=False)
    if str(
        os.getenv("PRODUCTION_FEATURE_BUILDS_ENABLED") or ""
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit(
            "STOP: PRODUCTION_FEATURE_BUILDS_ENABLED is not enabled"
        )
    writer_url = str(
        os.getenv("AUDIENCE_FEATURE_WRITER_DATABASE_URL") or ""
    ).strip()
    if not writer_url:
        raise SystemExit(
            "STOP: AUDIENCE_FEATURE_WRITER_DATABASE_URL is not configured"
        )

    request = load_request(args.request)
    rows = ProductionCanonicalFeatureSourceService(
        object_store=S3ProviderObjectStore()
    ).read(request.source)
    result = DurableProductionFeatureBuildService(
        pipeline=ProductionFeatureBuildPipelineService(
            embedding_service=ProductionCanonicalFeatureEmbeddingService(),
            feature_store=PgvectorAudienceFeatureStore(
                database_url=writer_url
            ),
            model_registry=ProductionEmbeddingModelRegistryService(
                database_url=writer_url
            ),
        ),
        state_service=ProductionFeatureBuildStateService(
            database_url=writer_url
        ),
    ).execute(request, rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
