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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
