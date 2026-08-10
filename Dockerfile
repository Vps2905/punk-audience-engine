ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.33
ARG PYTHON_BASE_IMAGE=python:3.11.15-slim@sha256:90edbeb8e4efce8dfe102f24c5ea1c8a1d770ff3d99c9c565d89ec97145f0fea

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PYTHONPATH=/app
ENV PATH=/app/.venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV HF_HOME=/opt/models
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV LOCAL_SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV LOCAL_SEMANTIC_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ENV LOCAL_SEMANTIC_MODEL_CACHE=/opt/models

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked --no-dev --no-install-project

COPY app /app/app
COPY scripts/prefetch_local_semantic_model.py /app/scripts/prefetch_local_semantic_model.py

# The runtime is network-independent: materialize the exact reviewed revision
# as a checksummed local artifact, then force offline loading from that path.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    python scripts/prefetch_local_semantic_model.py --cache-dir /opt/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Database migration and approved operator assets.
COPY migrations ./migrations
COPY scripts/apply_database_migrations.py ./scripts/apply_database_migrations.py
COPY scripts/run_provider_ingestion_worker.py ./scripts/run_provider_ingestion_worker.py
COPY scripts/provider_distributed_control_plane_lambda.py ./scripts/provider_distributed_control_plane_lambda.py
COPY scripts/provider_distributed_privacy_glue_job.py ./scripts/provider_distributed_privacy_glue_job.py
COPY scripts/reconcile_provider_distributed_jobs.py ./scripts/reconcile_provider_distributed_jobs.py
COPY scripts/register_provider_contract.py ./scripts/register_provider_contract.py
COPY scripts/replay_provider_ingestion.py ./scripts/replay_provider_ingestion.py
COPY scripts/index_historical_audience_features.py ./scripts/index_historical_audience_features.py
COPY scripts/benchmark_pgvector_recall.py ./scripts/benchmark_pgvector_recall.py
COPY scripts/preflight_phase2_database.py ./scripts/preflight_phase2_database.py
COPY scripts/inventory_historical_vector_snapshots.py ./scripts/inventory_historical_vector_snapshots.py
COPY scripts/apply_phase2_feature_migration.py ./scripts/apply_phase2_feature_migration.py
COPY scripts/provision_phase2_feature_roles.py ./scripts/provision_phase2_feature_roles.py
COPY scripts/apply_phase2_proposal_ledger_migration.py ./scripts/apply_phase2_proposal_ledger_migration.py
COPY scripts/provision_phase2_proposal_runtime_role.py ./scripts/provision_phase2_proposal_runtime_role.py
COPY scripts/apply_production_feature_build_migration.py ./scripts/apply_production_feature_build_migration.py
COPY scripts/register_production_embedding_model.py ./scripts/register_production_embedding_model.py
COPY scripts/build_production_audience_features.py ./scripts/build_production_audience_features.py
COPY scripts/production_runtime_entrypoint.py ./scripts/production_runtime_entrypoint.py

RUN groupadd --system --gid 10001 appgroup \
    && useradd --system --uid 10001 --gid appgroup \
       --home-dir /app --shell /usr/sbin/nologin appuser \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R appuser:appgroup /app /opt/models

USER 10001:10001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
