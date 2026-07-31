FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Production container must be CPU-only.
# Avoid pulling huge CUDA/NVIDIA wheels through torch dependencies.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
    && grep -v -E '^(torch|torchvision|torchaudio|nvidia-|triton)([=<>~! ]|$)' requirements.txt > /tmp/requirements.runtime.txt \
    && python -m pip install -r /tmp/requirements.runtime.txt

COPY app /app/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Database migration assets
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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
