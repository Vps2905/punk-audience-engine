# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings, ensure_data_dirs
from app.api.ingest_routes import router as ingest_router
from app.api.synthetic_routes import router as synthetic_router
from app.api.embedding_routes import router as embedding_router
from app.api.cohort_routes import router as cohort_router
from app.api.export_routes import router as export_router
from app.api.chat_routes import router as chat_router
from app.api.full_pipeline_routes import router as full_pipeline_router

ensure_data_dirs()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Privacy-safe audience ingestion, embeddings, cohorts, lookalikes, and Meta-safe export engine."
)

# Product-facing simple flow
app.include_router(full_pipeline_router)

# Engineering / advanced module APIs
app.include_router(ingest_router)
app.include_router(synthetic_router)
app.include_router(embedding_router)
app.include_router(cohort_router)
app.include_router(export_router)
app.include_router(chat_router)

app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")


@app.get("/")
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "ui": "http://127.0.0.1:8000/ui",
        "docs": "http://127.0.0.1:8000/docs",
        "simple_flow": "POST /audience/generate",
        "modules": [
            "ingestion_privacy",
            "synthetic_data",
            "embeddings",
            "cohorts_lookalike",
            "meta_export",
            "chat_orchestration"
        ]
    }


# Adaptive agent routes
from app.api.agent_routes import router as agent_router
app.include_router(agent_router)


# ---------------------------------------------------------------------
# Audience Intelligence Agents demo UI
# ---------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles as _StaticFiles
from fastapi.responses import FileResponse as _FileResponse

if not any(getattr(route, "path", None) == "/static" for route in app.routes):
    app.mount("/static", _StaticFiles(directory="app/static"), name="static")

@app.get("/ui/audience-agents")
def audience_agents_ui():
    return _FileResponse("app/static/audience_agents.html")

# ---------------------------------------------------------------------
# Audience Intelligence Agents UI
# ---------------------------------------------------------------------
from fastapi.responses import FileResponse as _AudienceFileResponse
from fastapi.staticfiles import StaticFiles as _AudienceStaticFiles

try:
    app.mount("/static", _AudienceStaticFiles(directory="app/static"), name="static")
except RuntimeError:
    pass

@app.get("/ui/audience-agents")
def audience_agents_ui():
    return _AudienceFileResponse("app/static/audience_agents.html")

from app.api.audience_intelligence_prompt import router as audience_intelligence_prompt_router
app.include_router(audience_intelligence_prompt_router)

from app.api.audience_intelligence_jobs import router as audience_intelligence_jobs_router
app.include_router(audience_intelligence_jobs_router)


from app.api.audience_intelligence_run_history import router as audience_intelligence_run_history_router
app.include_router(audience_intelligence_run_history_router)

from app.api.audience_intelligence_modules import router as audience_intelligence_modules_router
app.include_router(audience_intelligence_modules_router)

from app.api.audience_intelligence_swarm import router as audience_intelligence_swarm_router
app.include_router(audience_intelligence_swarm_router)
