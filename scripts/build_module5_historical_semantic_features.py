from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.models.production_historical_semantic_feature_contracts import (
    HistoricalSemanticFeatureBuildRequest,
)
from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)
from app.services.production_dual_model_candidate_retrieval_service import (
    ProductionDualModelCandidateRetrievalService,
)
from app.services.production_embedding_model_registry_service import (
    ProductionEmbeddingModelRegistryService,
)
from app.services.production_historical_semantic_feature_service import (
    ProductionHistoricalSafeFeatureReader,
    ProductionHistoricalSemanticFeatureService,
)


ROOT = Path(__file__).resolve().parents[1]


def _model(role: str) -> EmbeddingModelSpec:
    if role == "primary":
        return EmbeddingModelSpec(
            backend="sentence_transformers",
            model_name=(
                ProductionDualModelCandidateRetrievalService.PRIMARY_MODEL_NAME
            ),
            model_revision=(
                ProductionDualModelCandidateRetrievalService.PRIMARY_MODEL_REVISION
            ),
            dimension=384,
            normalize_embeddings=True,
            document_prefix="passage: ",
            query_prefix="query: ",
        )
    return EmbeddingModelSpec(
        backend="sentence_transformers",
        model_name=(
            ProductionDualModelCandidateRetrievalService.COMPLEMENTARY_MODEL_NAME
        ),
        model_revision=(
            ProductionDualModelCandidateRetrievalService.COMPLEMENTARY_MODEL_REVISION
        ),
        dimension=384,
        normalize_embeddings=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build one approved pinned semantic feature set from existing "
            "privacy-safe historical cohort traits."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--source-feature-set-id", required=True)
    parser.add_argument("--source-feature-set-version", type=int, required=True)
    parser.add_argument(
        "--model-role",
        choices=("primary", "complementary"),
        required=True,
    )
    parser.add_argument("--minimum-cohort-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument(
        "--confirm-privacy-safe-historical-reembedding",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_privacy_safe_historical_reembedding:
        raise SystemExit(
            "STOP: --confirm-privacy-safe-historical-reembedding is required"
        )

    load_dotenv(ROOT / ".env", override=False)
    reader_url = str(os.getenv("AUDIENCE_FEATURE_DATABASE_URL") or "").strip()
    writer_url = str(
        os.getenv("AUDIENCE_FEATURE_WRITER_DATABASE_URL") or ""
    ).strip()
    if not reader_url or not writer_url:
        raise SystemExit(
            "STOP: reader and writer feature database URLs are required"
        )

    model = _model(args.model_role)
    result = ProductionHistoricalSemanticFeatureService(
        source_reader=ProductionHistoricalSafeFeatureReader(reader_url),
        model_registry=ProductionEmbeddingModelRegistryService(reader_url),
        feature_writer=PgvectorAudienceFeatureStore(writer_url),
    ).build(
        HistoricalSemanticFeatureBuildRequest(
            tenant_id=args.tenant_id,
            source_feature_set_id=args.source_feature_set_id,
            source_feature_set_version=args.source_feature_set_version,
            model=model,
            minimum_cohort_size=args.minimum_cohort_size,
            batch_size=args.batch_size,
            requested_by=args.requested_by,
        ),
        privacy_safe_historical_reembedding_confirmed=True,
    )
    result["model_role"] = args.model_role
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
