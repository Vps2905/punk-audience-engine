from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.agents.maid_swarm_pipeline_agent import MaidSwarmPipelineAgent


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(clean_json(data), indent=2, default=str, allow_nan=False))


def copy_file(src: str | Path, dst: Path) -> None:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    shutil.copy2(src, dst)


def fake_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    print("Running real DB five-module demo pipeline...")

    agent = MaidSwarmPipelineAgent()
    result = agent.run(
        schema_name="public",
        table_name="maid_extractions",
        limit=10000,
        k_min=1000,
        epsilon=1.0,
        synthetic_rows=1000,
    )

    run_id = result["run_id"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    package_dir = Path("data/review_packages") / f"five_module_demo_{ts}_{run_id}"

    m1 = package_dir / "01_ingestion_privacy"
    m2 = package_dir / "02_embeddings_feature_store"
    m3 = package_dir / "03_cohort_management"
    m4 = package_dir / "04_meta_safe_export"
    m5 = package_dir / "05_simple_conversational_trigger"

    for folder in [m1, m2, m3, m4, m5]:
        folder.mkdir(parents=True, exist_ok=True)

    outputs = result["outputs"]
    cohort_metadata_path = Path(outputs["safe_cohort_metadata"])
    vector_path = Path(outputs["cohort_vectors"])
    synthetic_path = Path(outputs["synthetic_safe_seed_profiles"])
    export_manifest_path = Path(outputs["meta_safe_export_manifest"])

    cohorts = pd.read_csv(cohort_metadata_path)
    vectors = np.load(vector_path)
    synthetic = pd.read_csv(synthetic_path)

    # ------------------------------------------------------------------
    # MODULE 1: Ingestion & Privacy
    # ------------------------------------------------------------------
    clean_cols = [
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
        "sessions",
        "noisy_maid_volume",
        "privacy_status",
        "quality_score",
    ]
    cohorts[[c for c in clean_cols if c in cohorts.columns]].to_csv(
        m1 / "clean_feature_table.csv", index=False
    )

    write_json(
        m1 / "privacy_report.json",
        {
            "module": "Module 1 - Ingestion & Privacy Layer",
            "status": "completed",
            "source": result["source"],
            "privacy_controls": {
                "session_deduplication_enabled": True,
                "aggregation_enabled": True,
                "k_anonymity_enabled": True,
                "k_min": result["privacy"]["k_min"],
                "basic_dp_noise_enabled": result["privacy"]["dp_noise"],
                "epsilon": result["privacy"]["epsilon"],
            },
            "export_safety": {
                "raw_maids_exported": False,
                "hashed_real_maids_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "aggregated_only": True,
            },
            "result": {
                "raw_rows_loaded": result["source"]["raw_rows_loaded"],
                "deduped_sessions": result["source"]["deduped_sessions"],
                "safe_cohort_count": result["summary"]["safe_cohort_count"],
            },
        },
    )

    write_json(
        m1 / "hashing_manifest.json",
        {
            "module": "Hashing & Privacy Manifest",
            "status": "completed_as_privacy_manifest",
            "note": "Real MAID-level hashes are intentionally not exported because hashed MAIDs are still pseudonymous identifiers.",
            "production_hashing_strategy": {
                "recommended_algorithm": "HMAC-SHA256 or salted SHA-256",
                "salt_storage": "secret manager or environment variable",
                "raw_identifier_export": False,
                "hashed_real_identifier_review_export": False,
            },
            "sensitive_fields_detected": [
                "maids",
                "observations.lat",
                "observations.lng",
                "pois.lat",
                "pois.lng",
                "center.latitude",
                "center.longitude",
            ],
        },
    )

    pd.DataFrame(
        [
            {"raw_dummy_id": "dummy_maid_001", "hashed_dummy_id": fake_hash("dummy_maid_001")},
            {"raw_dummy_id": "dummy_maid_002", "hashed_dummy_id": fake_hash("dummy_maid_002")},
            {"raw_dummy_id": "dummy_maid_003", "hashed_dummy_id": fake_hash("dummy_maid_003")},
        ]
    ).to_csv(m1 / "dummy_hashing_demo.csv", index=False)

    k_cols = [
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
        "total_maid_volume",
        "noisy_maid_volume",
        "privacy_status",
        "quality_score",
    ]
    cohorts[[c for c in k_cols if c in cohorts.columns]].head(50).to_csv(
        m1 / "k_anonymity_dp_report.csv", index=False
    )

    write_json(
        m1 / "lineage_report.json",
        {
            "module": "Lineage Logging",
            "source": "Postgres public.maid_extractions",
            "transformations": [
                "load_rows_from_postgres",
                "dedupe_by_session_id_keep_latest",
                "parse_maids_json_for_counting_only",
                "parse_pois_json_to_safe_category_traits",
                "parse_center_json_to_location_name_only",
                "aggregate_safe_traits",
                "apply_k_anonymity",
                "apply_laplace_dp_noise",
                "create_clean_feature_table",
            ],
            "outputs": {
                "clean_feature_table": "clean_feature_table.csv",
                "privacy_report": "privacy_report.json",
                "hashing_manifest": "hashing_manifest.json",
                "k_anonymity_dp_report": "k_anonymity_dp_report.csv",
            },
        },
    )

    (m1 / "README.txt").write_text(
        "Module 1 output: ingestion, privacy, hashing manifest, k-anonymity, DP noise, lineage.\n"
        "No raw MAIDs, raw observations, raw lat/lng, email, phone, or DB credentials included.\n"
    )

    # ------------------------------------------------------------------
    # MODULE 2: Embedding & Feature Store
    # ------------------------------------------------------------------
    copy_file(vector_path, m2 / "cohort_vectors.npy")
    copy_file(cohort_metadata_path, m2 / "cohort_metadata.csv")

    preview = []
    max_rows = min(20, len(cohorts))
    max_dims = min(12, vectors.shape[1])

    for i in range(max_rows):
        row = {
            "row_index": i,
            "location_name": cohorts.iloc[i].get("location_name"),
            "primary_poi_type": cohorts.iloc[i].get("primary_poi_type"),
            "cluster_id": int(cohorts.iloc[i].get("cluster_id", -1)),
            "quality_score": float(cohorts.iloc[i].get("quality_score", 0)),
            "trait_text": cohorts.iloc[i].get("trait_text"),
        }
        for dim in range(max_dims):
            row[f"vector_dim_{dim}"] = float(vectors[i][dim])
        preview.append(row)

    pd.DataFrame(preview).to_csv(m2 / "vector_preview.csv", index=False)

    write_json(
        m2 / "embedding_manifest.json",
        {
            "module": "Module 2 - Embedding & Feature Store",
            "status": "completed",
            "current_engine": "TF-IDF vectorizer",
            "planned_production_engine": "sentence-transformers or OpenAI-compatible embeddings",
            "vector_store_current": "local NumPy file",
            "vector_store_planned": "Weaviate / Pinecone / Qdrant / pgvector",
            "vector_count": int(vectors.shape[0]),
            "vector_dimension": int(vectors.shape[1]),
            "files": {
                "cohort_vectors": "cohort_vectors.npy",
                "cohort_metadata": "cohort_metadata.csv",
                "vector_preview": "vector_preview.csv",
            },
        },
    )

    query_vector = vectors[0].reshape(1, -1)
    sims = cosine_similarity(query_vector, vectors).flatten()
    top_idx = sims.argsort()[::-1][:5]

    write_json(
        m2 / "similarity_search_demo.json",
        {
            "module": "Similarity Search Demo",
            "status": "completed_basic_v1",
            "query": "montreal restaurant evening audience",
            "note": "Current demo uses existing cohort vector as query vector. Production version will embed query text directly.",
            "top_matches": [
                {
                    "row_index": int(idx),
                    "similarity_score": float(sims[idx]),
                    "location_name": cohorts.iloc[idx].get("location_name"),
                    "primary_poi_type": cohorts.iloc[idx].get("primary_poi_type"),
                    "created_day_part": cohorts.iloc[idx].get("created_day_part"),
                    "quality_score": float(cohorts.iloc[idx].get("quality_score", 0)),
                }
                for idx in top_idx
            ],
        },
    )

    (m2 / "README.txt").write_text(
        "Module 2 output: vectors, vector preview, cohort metadata, embedding manifest, similarity demo.\n"
    )

    # ------------------------------------------------------------------
    # MODULE 3: Cohort Management & Lookalike
    # ------------------------------------------------------------------
    cohorts.head(30).to_csv(m3 / "top_cohorts.csv", index=False)

    cluster_summary = (
        cohorts.groupby("cluster_id", dropna=False)
        .agg(
            cohort_count=("cluster_id", "count"),
            avg_quality_score=("quality_score", "mean"),
            total_noisy_maid_volume=("noisy_maid_volume", "sum"),
        )
        .reset_index()
        .sort_values("cohort_count", ascending=False)
    )
    cluster_summary.to_csv(m3 / "cluster_summary.csv", index=False)

    write_json(
        m3 / "cohort_quality_report.json",
        {
            "module": "Module 3 - Cohort Management & Lookalike",
            "status": "completed_basic_v1",
            "safe_cohort_count": int(len(cohorts)),
            "cluster_count": int(cohorts["cluster_id"].nunique()),
            "quality": {
                "max": float(cohorts["quality_score"].max()),
                "min": float(cohorts["quality_score"].min()),
                "avg": float(cohorts["quality_score"].mean()),
            },
            "cohort_storage": "aggregated traits + synthetic representatives",
            "privacy_output": "aggregated/synthetic only",
        },
    )

    source_idx = 0
    source_cluster = cohorts.iloc[source_idx]["cluster_id"]
    similar = cohorts[(cohorts["cluster_id"] == source_cluster) & (cohorts.index != source_idx)].head(5)

    write_json(
        m3 / "lookalike_demo.json",
        {
            "module": "Basic Lookalike Demo",
            "status": "completed_basic_v1",
            "source_cohort": cohorts.iloc[source_idx].to_dict(),
            "method": "same_cluster_plus_safe_trait_similarity_current_v1",
            "note": "Dedicated vector-based LookalikeAgent is planned as production upgrade.",
            "similar_cohorts": similar.to_dict(orient="records"),
        },
    )

    (m3 / "README.txt").write_text(
        "Module 3 output: top cohorts, cluster summary, quality report, basic lookalike demo.\n"
    )

    # ------------------------------------------------------------------
    # MODULE 4: Meta Safe Export
    # ------------------------------------------------------------------
    copy_file(synthetic_path, m4 / "synthetic_safe_seed_profiles.csv")
    copy_file(export_manifest_path, m4 / "meta_safe_export_manifest.json")

    write_json(
        m4 / "export_package_summary.json",
        {
            "module": "Module 4 - Meta Safe Export",
            "status": "created",
            "platform": "Meta Advantage+",
            "approval_status": result["export"]["approval_status"],
            "safe_export_guarantees": {
                "contains_raw_maids": False,
                "contains_raw_email": False,
                "contains_raw_phone": False,
                "contains_raw_lat_lng": False,
                "contains_individual_user_data": False,
                "aggregated_traits_only": True,
                "synthetic_seed_profiles": True,
                "requires_manual_approval_before_upload": True,
            },
            "files": {
                "synthetic_safe_seed_profiles": "synthetic_safe_seed_profiles.csv",
                "meta_safe_export_manifest": "meta_safe_export_manifest.json",
            },
        },
    )

    write_json(
        m4 / "approval_request.json",
        {
            "module": "Export Approval Step",
            "status": "pending_approval",
            "reason": "No export should be uploaded externally until reviewed.",
            "review_required_for": [
                "privacy guarantees",
                "cohort quality",
                "platform compatibility",
                "business approval",
            ],
        },
    )

    (m4 / "README.txt").write_text(
        "Module 4 output: synthetic safe seed profiles, Meta-safe export manifest, approval request.\n"
    )

    # ------------------------------------------------------------------
    # MODULE 5: Simple Conversational Trigger
    # ------------------------------------------------------------------
    write_json(
        m5 / "sample_user_requests.json",
        {
            "module": "Module 5 - Simple Conversational Trigger",
            "status": "demo_ready",
            "sample_user_requests": [
                "create montreal restaurant evening cohort",
                "generate fitness evening audience",
                "prepare safe export for top cohort",
                "find similar audience for restaurant visitors",
            ],
        },
    )

    write_json(
        m5 / "chat_orchestration_demo.json",
        {
            "input": "create montreal restaurant evening cohort and prepare safe export",
            "parsed_intent": "create_cohort_and_export",
            "extracted_fields": {
                "location": "montreal",
                "poi_type": "restaurant",
                "day_part": "evening",
                "export_platform": "meta",
            },
            "called_modules": [
                "ingestion_privacy",
                "embedding_feature_store",
                "cohort_management",
                "meta_safe_export",
            ],
            "result": {
                "safe_cohort_count": result["summary"]["safe_cohort_count"],
                "cluster_count": result["summary"]["cluster_count"],
                "synthetic_rows": result["summary"]["synthetic_rows"],
                "approval_status": result["export"]["approval_status"],
            },
        },
    )

    write_json(
        m5 / "endpoint_mapping.json",
        {
            "module": "API Endpoint Mapping",
            "available_or_demo_endpoints": {
                "/ingest": "CSV/sample ingestion flow",
                "/status/{job_id}": "Job status lookup",
                "/synthetic/generate": "Synthetic generation",
                "/embed": "Embedding endpoint",
                "/search/similar": "Similarity search endpoint",
                "/cluster": "Clustering endpoint",
                "/cohort/create": "Cohort creation",
                "/cohort/lookalike": "Lookalike endpoint",
                "/cohort/query": "Cohort query",
                "/export/meta/{cohort_id}": "Safe export",
                "/agents/run-maid-swarm": "Real DB v1 endpoint used for current demo outputs",
            },
            "advanced_orchestration_note": "Full multi-agent orchestration and background swarm evolution are intentionally not included in this five-module demo package.",
        },
    )

    (m5 / "README.txt").write_text(
        "Module 5 output: sample requests, parsed trigger demo, endpoint mapping.\n"
    )

    # ------------------------------------------------------------------
    # MASTER PACKAGE
    # ------------------------------------------------------------------
    write_json(
        package_dir / "PACKAGE_INDEX.json",
        {
            "package_name": "Five Module Audience Intelligence Demo Outputs",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "source": "Postgres public.maid_extractions",
            "summary": {
                "raw_rows_loaded": result["source"]["raw_rows_loaded"],
                "deduped_sessions": result["source"]["deduped_sessions"],
                "safe_cohort_count": result["summary"]["safe_cohort_count"],
                "cluster_count": result["summary"]["cluster_count"],
                "synthetic_rows": result["summary"]["synthetic_rows"],
                "approval_status": result["export"]["approval_status"],
            },
            "covered_modules": [
                "Module 1 - Ingestion & Privacy Layer",
                "Module 2 - Embedding & Feature Store",
                "Module 3 - Cohort Management & Lookalike",
                "Module 4 - Meta Safe Export",
                "Module 5 - Simple Conversational Trigger",
            ],
            "not_included_in_this_demo": [
                "advanced multi-agent orchestration",
                "background swarm evolution",
                "production auth",
                "real external upload",
            ],
            "privacy_note": "No raw MAIDs, raw observations, raw lat/lng, email, phone, DB credentials, or individual-level rows are included.",
        },
    )

    (package_dir / "README.txt").write_text(
        f"""Five Module Audience Intelligence Demo Outputs

Run ID:
{run_id}

Source:
Postgres public.maid_extractions

Summary:
- Raw DB rows loaded: {result["source"]["raw_rows_loaded"]}
- Deduped sessions: {result["source"]["deduped_sessions"]}
- Safe cohorts: {result["summary"]["safe_cohort_count"]}
- Clusters: {result["summary"]["cluster_count"]}
- Synthetic seed profiles: {result["summary"]["synthetic_rows"]}
- Export approval status: {result["export"]["approval_status"]}

Covered Modules:
1. Ingestion & Privacy Layer
2. Embedding & Feature Store
3. Cohort Management & Lookalike
4. Meta Safe Export
5. Simple Conversational Trigger

Privacy:
This package does not include raw MAIDs, raw observations, raw lat/lng, email, phone, DB credentials, or individual-level records.
"""
    )

    print("\nDONE")
    print(f"Package folder: {package_dir}")
    print(f"Run ID: {run_id}")
    print(f"Raw rows: {result['source']['raw_rows_loaded']}")
    print(f"Deduped sessions: {result['source']['deduped_sessions']}")
    print(f"Safe cohorts: {result['summary']['safe_cohort_count']}")
    print(f"Clusters: {result['summary']['cluster_count']}")
    print(f"Synthetic rows: {result['summary']['synthetic_rows']}")


if __name__ == "__main__":
    main()
