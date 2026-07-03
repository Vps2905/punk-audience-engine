from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent
from app.agents.embedding_feature_store_agent import EmbeddingFeatureStoreAgent
from app.agents.cohort_management_agent import CohortManagementAgent
from app.agents.safe_export_agent import SafeExportAgent
from app.agents.synthetic_engine_agent import SyntheticEngineAgent
from app.agents.privacy_layer_agent import PrivacyLayerAgent

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence/modules",
    tags=["Audience Intelligence Module APIs"],
    dependencies=[Depends(require_audience_api_key)],
)


class IngestRequest(BaseModel):
    records: Optional[List[Dict[str, Any]]] = None
    csv_path: Optional[str] = None
    output_dir: str = "data/module_api_runs/ingest"
    run_id: Optional[str] = None
    identifier_column: str = "session_id"
    count_column: str = "maid_count"
    timestamp_column: str = "created_at"
    location_column: str = "location_name"
    poi_column: str = "primary_poi_type"
    k_min: int = 1000
    epsilon: float = 1.0


class SyntheticGenerateRequest(BaseModel):
    cohort_path: str
    output_dir: str = "data/module_api_runs/synthetic"
    run_id: Optional[str] = None
    rows: int = 1000
    epsilon: float = 1.0
    k_min: int = 1000
    engine: str = "dp_aggregate"


class EmbedRequest(BaseModel):
    cohort_path: str
    output_dir: str = "data/module_api_runs/embeddings"
    run_id: Optional[str] = None
    embedding_provider: str = "sklearn_tfidf"
    max_features: int = 384


class SimilarSearchRequest(BaseModel):
    metadata_path: str
    vectors_path: str
    query: str
    top_k: int = 5


class ClusterRequest(BaseModel):
    metadata_path: str
    vectors_path: str
    output_dir: str = "data/module_api_runs/cohort_management"
    run_id: Optional[str] = None
    min_clusters: int = 2
    max_clusters: int = 8
    top_n: int = 25
    lookalike_top_k: int = 3
    min_export_quality: float = 0.25


class CohortQueryRequest(BaseModel):
    cohorts_path: str
    location: Optional[str] = None
    poi_type: Optional[str] = None
    daypart: Optional[str] = None
    min_quality: float = 0.0
    limit: int = 25


class CohortLookalikeRequest(BaseModel):
    lookalikes_path: str
    seed_cohort_index: Optional[int] = None
    min_similarity: float = 0.0
    limit: int = 25


class MetaExportRequest(BaseModel):
    safe_export_dir: str
    approved_only: bool = True
    output_filename: str = "meta_advantage_plus_seed_payload.json"


def _new_run_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _read_df_from_request(records: Optional[List[Dict[str, Any]]], csv_path: Optional[str]) -> pd.DataFrame:
    if records:
        return pd.DataFrame(records)

    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"CSV path not found: {csv_path}")
        return pd.read_csv(path)

    raise HTTPException(status_code=400, detail="Either records or csv_path is required.")


def _safe_preview(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    safe = df.head(limit).copy()
    blocked = ["maid", "raw_maid", "device_id", "email", "phone", "lat", "lng", "latitude", "longitude", "hashed"]
    for col in list(safe.columns):
        lower = col.lower()
        if any(token in lower for token in blocked):
            safe = safe.drop(columns=[col])
    return json.loads(safe.to_json(orient="records"))


@router.post("/ingest")
def ingest(request: IngestRequest) -> Dict[str, Any]:
    try:
        df = _read_df_from_request(request.records, request.csv_path)
        output_dir = Path(request.output_dir) / (request.run_id or _new_run_id("ingest"))

        agent = PrivacyLayerAgent()
        helper = AudienceIntelligenceOrchestratorAgent()

        result = helper._call_agent_method(
            agent=agent,
            method_names=["run", "process", "build", "execute", "transform", "anonymize"],
            kwargs={
                "raw_observations": df,
                "observations": df,
                "input_df": df,
                "df": df,
                "data": df,
                "cohorts": df,
                "output_dir": output_dir,
                "run_id": request.run_id or output_dir.name,
                "identifier_column": request.identifier_column,
                "count_column": request.count_column,
                "timestamp_column": request.timestamp_column,
                "location_column": request.location_column,
                "poi_column": request.poi_column,
                "k_min": request.k_min,
                "epsilon": request.epsilon,
            },
        )

        return {
            "status": result.get("status", "completed"),
            "output_dir": str(output_dir),
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
def module_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "modules": {
            "privacy_layer": "available",
            "synthetic_generation": "available_dp_aggregate",
            "dpgc_synthesizer": "optional_not_required",
            "embedding": "available_sklearn_tfidf",
            "cohort_management": "available",
            "safe_export": "available",
            "meta_export_adapter": "safe_payload_only_no_upload",
        },
    }


@router.post("/synthetic/generate")
def synthetic_generate(request: SyntheticGenerateRequest) -> Dict[str, Any]:
    try:
        cohort_path = Path(request.cohort_path)
        if not cohort_path.exists():
            raise HTTPException(status_code=404, detail=f"cohort_path not found: {request.cohort_path}")

        cohorts = pd.read_csv(cohort_path)
        output_dir = Path(request.output_dir) / (request.run_id or _new_run_id("synthetic"))

        agent = SyntheticEngineAgent()
        helper = AudienceIntelligenceOrchestratorAgent()

        result = helper._call_agent_method(
            agent=agent,
            method_names=["generate"],
            kwargs={
                "cohorts": cohorts,
                "df": cohorts,
                "data": cohorts,
                "output_dir": output_dir,
                "run_id": request.run_id or output_dir.name,
                "engine_requested": request.engine,
                "engine": request.engine,
                "production_mode": True,
                "allow_fallback": False,
                "synthetic_rows": request.rows,
                "rows": request.rows,
                "epsilon": request.epsilon,
                "k_min": request.k_min,
            },
        )

        return {
            "status": result.get("status", "completed"),
            "output_dir": str(output_dir),
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/embed")
def embed(request: EmbedRequest) -> Dict[str, Any]:
    try:
        cohort_path = Path(request.cohort_path)
        if not cohort_path.exists():
            raise HTTPException(status_code=404, detail=f"cohort_path not found: {request.cohort_path}")

        cohorts = pd.read_csv(cohort_path)
        output_dir = Path(request.output_dir) / (request.run_id or _new_run_id("embedding"))

        agent = EmbeddingFeatureStoreAgent()
        result = agent.build(
            cohorts=cohorts,
            output_dir=output_dir,
            embedding_provider=request.embedding_provider,
            run_id=request.run_id or output_dir.name,
            max_features=request.max_features,
        )

        return {
            "status": result["status"],
            "output_dir": str(output_dir),
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/search/similar")
def search_similar(request: SimilarSearchRequest) -> Dict[str, Any]:
    try:
        metadata_path = Path(request.metadata_path)
        vectors_path = Path(request.vectors_path)

        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"metadata_path not found: {request.metadata_path}")
        if not vectors_path.exists():
            raise HTTPException(status_code=404, detail=f"vectors_path not found: {request.vectors_path}")

        metadata = pd.read_csv(metadata_path)
        vectors = np.load(vectors_path)

        agent = EmbeddingFeatureStoreAgent()
        result = agent.search_similar(
            query_text=request.query,
            metadata=metadata,
            vectors=vectors,
            top_k=request.top_k,
        )

        return {
            "status": "completed",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cluster")
def cluster(request: ClusterRequest) -> Dict[str, Any]:
    try:
        metadata_path = Path(request.metadata_path)
        vectors_path = Path(request.vectors_path)

        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"metadata_path not found: {request.metadata_path}")
        if not vectors_path.exists():
            raise HTTPException(status_code=404, detail=f"vectors_path not found: {request.vectors_path}")

        output_dir = Path(request.output_dir) / (request.run_id or _new_run_id("cohort_management"))

        agent = CohortManagementAgent()
        result = agent.run_from_artifacts(
            metadata_path=metadata_path,
            vectors_path=vectors_path,
            output_dir=output_dir,
            run_id=request.run_id or output_dir.name,
            min_clusters=request.min_clusters,
            max_clusters=request.max_clusters,
            top_n=request.top_n,
            lookalike_top_k=request.lookalike_top_k,
            min_export_quality=request.min_export_quality,
        )

        return {
            "status": result["status"],
            "output_dir": str(output_dir),
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cohort/create")
def cohort_create(request: ClusterRequest) -> Dict[str, Any]:
    return cluster(request)


@router.post("/cohort/query")
def cohort_query(request: CohortQueryRequest) -> Dict[str, Any]:
    try:
        path = Path(request.cohorts_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"cohorts_path not found: {request.cohorts_path}")

        df = pd.read_csv(path)

        mask = pd.Series(True, index=df.index)

        if request.location and "location_name" in df.columns:
            mask = mask & df["location_name"].astype(str).str.lower().str.contains(request.location.lower(), na=False)

        if request.poi_type and "primary_poi_type" in df.columns:
            mask = mask & df["primary_poi_type"].astype(str).str.lower().str.contains(request.poi_type.lower(), na=False)

        if request.daypart and "created_day_part" in df.columns:
            mask = mask & df["created_day_part"].astype(str).str.lower().eq(request.daypart.lower())

        quality_col = "management_quality_score" if "management_quality_score" in df.columns else "quality_score"
        if quality_col in df.columns:
            mask = mask & (pd.to_numeric(df[quality_col], errors="coerce").fillna(0) >= request.min_quality)
            df = df.sort_values(quality_col, ascending=False)

        filtered = df[mask].head(request.limit).copy()

        return {
            "status": "completed",
            "matched": int(len(filtered)),
            "results": _safe_preview(filtered, limit=request.limit),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cohort/lookalike")
def cohort_lookalike(request: CohortLookalikeRequest) -> Dict[str, Any]:
    try:
        path = Path(request.lookalikes_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"lookalikes_path not found: {request.lookalikes_path}")

        df = pd.read_csv(path)

        if request.seed_cohort_index is not None and "seed_cohort_index" in df.columns:
            df = df[df["seed_cohort_index"] == request.seed_cohort_index]

        if "similarity_score" in df.columns:
            df = df[pd.to_numeric(df["similarity_score"], errors="coerce").fillna(0) >= request.min_similarity]
            df = df.sort_values("similarity_score", ascending=False)

        df = df.head(request.limit)

        return {
            "status": "completed",
            "matched": int(len(df)),
            "results": _safe_preview(df, limit=request.limit),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/export/meta/{cohort_id}")
def export_meta_seed_payload(cohort_id: str, request: MetaExportRequest) -> Dict[str, Any]:
    """
    Safe Meta Advantage+ seed payload generator.

    This does NOT upload to Meta. It creates a gated payload only after approval.
    Real Meta upload should be a separate explicit connector with credentials.
    """
    try:
        safe_export_dir = Path(request.safe_export_dir)

        manifest_path = safe_export_dir / "safe_export_manifest.json"
        cohorts_path = safe_export_dir / "safe_export_cohorts.csv"
        synthetic_path = safe_export_dir.parent / "02_synthetic" / "synthetic_safe_seed_profiles.csv"

        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"safe_export_manifest.json not found in {safe_export_dir}")
        if not cohorts_path.exists():
            raise HTTPException(status_code=404, detail=f"safe_export_cohorts.csv not found in {safe_export_dir}")

        manifest = json.loads(manifest_path.read_text())
        cohorts = pd.read_csv(cohorts_path)

        if request.approved_only and manifest.get("approval_status") != "approved":
            raise HTTPException(
                status_code=403,
                detail="Export is blocked until approval_status is approved.",
            )

        if "export_cohort_id" not in cohorts.columns:
            raise HTTPException(status_code=400, detail="safe_export_cohorts.csv missing export_cohort_id")

        selected = cohorts[cohorts["export_cohort_id"].astype(str) == cohort_id]

        if selected.empty:
            raise HTTPException(status_code=404, detail=f"cohort_id not found: {cohort_id}")

        row = selected.iloc[0].to_dict()

        synthetic_preview: List[Dict[str, Any]] = []
        if synthetic_path.exists():
            synthetic = pd.read_csv(synthetic_path)
            synthetic_preview = _safe_preview(synthetic, limit=25)

            # Keep preview approval state aligned with the approved export package.
            for item in synthetic_preview:
                item["approval_status"] = manifest.get("approval_status", "pending_approval")

        payload = {
            "package_type": "meta_advantage_plus_safe_seed_payload",
            "cohort_id": cohort_id,
            "meta_upload_performed": False,
            "requires_explicit_meta_connector": True,
            "approval_status": manifest.get("approval_status"),
            "downstream_export_enabled": manifest.get("downstream_export_enabled"),
            "audience": {
                "name": row.get("audience_name"),
                "location_name": row.get("location_name"),
                "primary_poi_type": row.get("primary_poi_type"),
                "created_day_part": row.get("created_day_part"),
                "lookback_bucket": row.get("lookback_bucket"),
                "management_quality_score": row.get("management_quality_score"),
                "privacy_mode": row.get("privacy_mode"),
                "data_safety_status": row.get("data_safety_status"),
            },
            "safe_seed_strategy": {
                "use_aggregated_traits": True,
                "use_synthetic_seed_profiles": bool(synthetic_preview),
                "raw_maids_included": False,
                "hashed_identifiers_included": False,
                "raw_lat_lng_included": False,
                "individual_user_rows_included": False,
            },
            "synthetic_seed_preview": synthetic_preview,
        }

        output_path = safe_export_dir / request.output_filename
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

        return {
            "status": "completed",
            "message": "Safe Meta seed payload generated. No Meta upload was performed.",
            "output_path": str(output_path),
            "payload": payload,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
