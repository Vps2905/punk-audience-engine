from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from app.agents.postgres_source_agent import PostgresSourceAgent
from app.agents.schema_profiler_agent import SchemaProfilerAgent
from app.agents.swarm_coordinator import SwarmCoordinator
from app.agents.workflow_planner_agent import WorkflowPlannerAgent


router = APIRouter(prefix="/agents", tags=["Adaptive Agents"])


class PostgresPlanRequest(BaseModel):
    schema_name: str = Field(default="public")
    table_name: str
    limit: int = Field(default=10000, ge=1, le=100000)


@router.post("/plan-csv")
async def plan_csv_workflow(file: UploadFile = File(...)):
    suffix = Path(file.filename or "input.csv").suffix or ".csv"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    coordinator = SwarmCoordinator()
    return coordinator.plan_from_csv(tmp_path)


@router.get("/postgres/tables")
async def list_postgres_tables(schema_name: str = "public"):
    """
    Lists available Postgres tables using ECHO_DATABASE_URL.

    Do not pass DB secrets in URL.
    Store DB connection in .env / environment.
    """

    source_agent = PostgresSourceAgent()
    return source_agent.list_tables(schema_name=schema_name)


@router.post("/plan-postgres")
async def plan_postgres_workflow(request: PostgresPlanRequest):
    """
    Samples a Postgres table and creates:
    - schema profile
    - adaptive workflow plan
    - swarm architecture plan

    This does not export anything.
    """

    source_agent = PostgresSourceAgent()
    profiler = SchemaProfilerAgent()
    planner = WorkflowPlannerAgent()

    df = source_agent.sample_table(
        schema_name=request.schema_name,
        table_name=request.table_name,
        limit=request.limit,
    )

    profile = profiler.profile_dataframe(
        df=df,
        source_name=f"postgres:{request.schema_name}.{request.table_name}",
    )

    plan = planner.create_plan(profile)

    return {
        "agent": "swarm_coordinator",
        "status": "ready_for_adaptive_execution",
        "source": {
            "type": "postgres",
            "schema_name": request.schema_name,
            "table_name": request.table_name,
            "sample_limit": request.limit,
            "sample_rows_loaded": len(df),
        },
        "schema_profile": profile,
        "workflow_plan": plan,
        "swarm_architecture": {
            "mode": "multi_agent_swarm",
            "agents": [
                "postgres_source_agent",
                "schema_profiler_agent",
                "workflow_planner_agent",
                "privacy_agent",
                "synthetic_agent",
                "embedding_agent",
                "cohort_agent",
                "export_agent",
                "swarm_monitor_agent",
            ],
            "coordination_strategy": "supervisor_plans_then_agents_execute",
        },
    }


# -------------------------------------------------------------------
# MAID Extraction Adaptive Execution
# -------------------------------------------------------------------

from app.agents.maid_extraction_execution_agent import MaidExtractionExecutionAgent


class MaidExtractionExecuteRequest(BaseModel):
    schema_name: str = Field(default="public")
    table_name: str = Field(default="maid_extractions")
    limit: int = Field(default=10000, ge=1, le=100000)
    k_min: int = Field(default=1000, ge=1)
    epsilon: float = Field(default=1.0, gt=0)


@router.post("/execute-maid-extractions")
async def execute_maid_extraction_pipeline(request: MaidExtractionExecuteRequest):
    """
    Executes privacy-safe cohort generation from Postgres maid_extractions.

    Current real data source:
    - public.maid_extractions

    Privacy:
    - raw MAIDs are not exported
    - raw observations are not exported
    - raw lat/lng is not exported
    - output is aggregated only
    """

    execution_agent = MaidExtractionExecutionAgent()
    return execution_agent.execute(
        schema_name=request.schema_name,
        table_name=request.table_name,
        limit=request.limit,
        k_min=request.k_min,
        epsilon=request.epsilon,
    )


# -------------------------------------------------------------------
# Full MAID Swarm Pipeline
# -------------------------------------------------------------------

from app.agents.maid_swarm_pipeline_agent import MaidSwarmPipelineAgent


class MaidSwarmRunRequest(BaseModel):
    schema_name: str = Field(default="public")
    table_name: str = Field(default="maid_extractions")
    limit: int = Field(default=10000, ge=1, le=100000)
    k_min: int = Field(default=1000, ge=1)
    epsilon: float = Field(default=1.0, gt=0)
    synthetic_rows: int = Field(default=1000, ge=1, le=100000)


@router.post("/run-maid-swarm")
async def run_maid_swarm_pipeline(request: MaidSwarmRunRequest):
    """
    Full real-data swarm execution:

    Postgres maid_extractions
    -> privacy-safe cohorts
    -> embeddings
    -> clustering
    -> synthetic seed profiles
    -> Meta-safe export package
    """

    agent = MaidSwarmPipelineAgent()
    return agent.run(
        schema_name=request.schema_name,
        table_name=request.table_name,
        limit=request.limit,
        k_min=request.k_min,
        epsilon=request.epsilon,
        synthetic_rows=request.synthetic_rows,
    )


# -------------------------------------------------------------------
# Swarm Monitor Agent
# -------------------------------------------------------------------

from app.agents.swarm_monitor_agent import SwarmMonitorAgent


class SwarmMonitorRequest(BaseModel):
    schema_name: str = Field(default="public")
    table_name: str = Field(default="maid_extractions")


@router.post("/monitor-swarm-source")
async def monitor_swarm_source(request: SwarmMonitorRequest):
    """
    Checks whether the source table changed and recommends whether
    the swarm pipeline should rerun.
    """

    agent = SwarmMonitorAgent()
    return agent.monitor_table(
        schema_name=request.schema_name,
        table_name=request.table_name,
    )

# ---------------------------------------------------------------------
# Audience Intelligence Agents - Five Module Demo Output UI endpoints
# ---------------------------------------------------------------------
import os as _os
import sys as _sys
import json as _json
import subprocess as _subprocess
from pathlib import Path as _Path

from fastapi import HTTPException as _HTTPException
from fastapi.responses import FileResponse as _FileResponse


_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_REVIEW_ROOT = _PROJECT_ROOT / "data" / "review_packages"

_MODULE_ZIP_NAMES = [
    "MODULE_1_INGESTION_PRIVACY.zip",
    "MODULE_2_EMBEDDINGS_FEATURE_STORE.zip",
    "MODULE_3_COHORT_MANAGEMENT.zip",
    "MODULE_4_META_SAFE_EXPORT.zip",
    "MODULE_5_CONVERSATIONAL_TRIGGER.zip",
    "ALL_MODULES_COMBINED.zip",
]


def _latest_five_module_package() -> _Path:
    if not _REVIEW_ROOT.exists():
        raise _HTTPException(status_code=404, detail="No review package found yet.")

    packages = [
        p for p in _REVIEW_ROOT.glob("five_module_demo_*")
        if p.is_dir()
    ]

    if not packages:
        raise _HTTPException(status_code=404, detail="No five_module_demo package found yet.")

    return sorted(packages, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _read_package_index(package_dir: _Path) -> dict:
    index_path = package_dir / "PACKAGE_INDEX.json"
    if not index_path.exists():
        return {}

    try:
        return _json.loads(index_path.read_text())
    except Exception:
        return {}


def _build_module_zip_list(package_dir: _Path) -> list[dict]:
    zip_dir = package_dir / "separate_module_zips"
    items = []

    for name in _MODULE_ZIP_NAMES:
        path = zip_dir / name
        items.append(
            {
                "name": name,
                "exists": path.exists(),
                "path": str(path.relative_to(_PROJECT_ROOT)) if path.exists() else None,
                "download_url": f"/agents/download-five-module-demo/{name}",
            }
        )

    return items


@router.post("/generate-five-module-demo-output")
def generate_five_module_demo_output():
    """
    Runs the local five-module demo output generator and creates separate module ZIPs.
    This is intended for local demo/review only.
    """
    generator = _PROJECT_ROOT / "scripts" / "generate_five_module_demo_outputs.py"
    splitter = _PROJECT_ROOT / "scripts" / "split_module_outputs.py"

    if not generator.exists():
        raise _HTTPException(status_code=500, detail=f"Missing script: {generator}")

    if not splitter.exists():
        raise _HTTPException(status_code=500, detail=f"Missing script: {splitter}")

    env = _os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)

    commands = [
        [_sys.executable, str(generator)],
        [_sys.executable, str(splitter)],
    ]

    logs = []

    for cmd in commands:
        completed = _subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )

        logs.append(
            {
                "command": " ".join(cmd),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )

        if completed.returncode != 0:
            raise _HTTPException(
                status_code=500,
                detail={
                    "message": "Five-module output generation failed.",
                    "logs": logs,
                },
            )

    package_dir = _latest_five_module_package()
    package_index = _read_package_index(package_dir)

    return {
        "status": "completed",
        "latest_package": str(package_dir.relative_to(_PROJECT_ROOT)),
        "summary": package_index.get("summary", {}),
        "covered_modules": package_index.get("covered_modules", []),
        "module_zips": _build_module_zip_list(package_dir),
        "logs": logs,
    }


@router.get("/latest-five-module-demo-output")
def latest_five_module_demo_output():
    package_dir = _latest_five_module_package()
    package_index = _read_package_index(package_dir)

    return {
        "status": "found",
        "latest_package": str(package_dir.relative_to(_PROJECT_ROOT)),
        "summary": package_index.get("summary", {}),
        "covered_modules": package_index.get("covered_modules", []),
        "module_zips": _build_module_zip_list(package_dir),
    }


@router.get("/download-five-module-demo/{artifact_name}")
def download_five_module_demo_artifact(artifact_name: str):
    if artifact_name not in _MODULE_ZIP_NAMES and artifact_name != "MASTER_SUMMARY.json":
        raise _HTTPException(status_code=400, detail="Unsupported artifact name.")

    package_dir = _latest_five_module_package()
    artifact_path = package_dir / "separate_module_zips" / artifact_name

    if not artifact_path.exists():
        raise _HTTPException(status_code=404, detail="Artifact not found.")

    return _FileResponse(
        artifact_path,
        filename=artifact_name,
        media_type="application/octet-stream",
    )
