from __future__ import annotations

import argparse
import json

from app.services.audience_feature_proposal_service import (
    FeatureQueryEmbeddingService,
)
from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)


def _csv_values(value: str) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").split(",")
        if item.strip()
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HNSW retrieval with an exact pgvector baseline "
            "for one privacy-safe audience query."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--feature-set-id")
    parser.add_argument("--feature-set-version", type=int)
    parser.add_argument("--locations", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--dayparts", default="")
    parser.add_argument("--exclusions", default="")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--execution-mode",
        choices=["historical_preview", "production"],
        default="historical_preview",
    )
    return parser


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    store = PgvectorAudienceFeatureStore()
    feature_set = store.get_feature_set(
        tenant_id=args.tenant_id,
        feature_set_id=args.feature_set_id,
        version=args.feature_set_version,
        data_use_mode=args.execution_mode,
    )
    embedding = FeatureQueryEmbeddingService().encode(
        args.query,
        feature_set,
    )
    search_kwargs = {
        "tenant_id": args.tenant_id,
        "feature_set_id": feature_set["feature_set_id"],
        "feature_set_version": int(feature_set["version"]),
        "query_text": args.query,
        "query_embedding": embedding,
        "execution_mode": args.execution_mode,
        "locations": _csv_values(args.locations),
        "categories": _csv_values(args.categories),
        "dayparts": _csv_values(args.dayparts),
        "exclusions": _csv_values(args.exclusions),
        "top_k": max(1, min(int(args.top_k), 100)),
    }
    approximate = store.hybrid_search(
        **search_kwargs,
        search_strategy="ann",
    )
    exact = store.hybrid_search(
        **search_kwargs,
        search_strategy="exact",
    )
    approximate_ids = {
        str(row["feature_id"])
        for row in approximate
    }
    exact_ids = {
        str(row["feature_id"])
        for row in exact
    }
    overlap = approximate_ids & exact_ids
    denominator = max(len(exact_ids), 1)
    return {
        "status": "completed",
        "tenant_id": args.tenant_id,
        "feature_set_id": feature_set["feature_set_id"],
        "feature_set_version": int(feature_set["version"]),
        "execution_mode": args.execution_mode,
        "top_k": search_kwargs["top_k"],
        "ann_result_count": len(approximate_ids),
        "exact_result_count": len(exact_ids),
        "overlap_count": len(overlap),
        "recall_at_k": round(len(overlap) / denominator, 6),
        "activation_or_export_performed": False,
    }


def main() -> None:
    print(
        json.dumps(
            benchmark(_parser().parse_args()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
