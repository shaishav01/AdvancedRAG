# AdvancedRAG API image (Cloud Run). Python 3.13 matches the version the project was developed and tested on.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Patch OS-level CVEs, then install system deps for native packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the pip layer is cached; it only re-runs when requirements-prod.txt changes.
COPY requirements-prod.txt .
RUN pip install --prefer-binary -r requirements-prod.txt

# Bake the FlashRank reranker model into the image so the first request doesn't download it (cold-start latency).
ENV FLASHRANK_CACHE_DIR=/opt/flashrank
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/opt/flashrank')"

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser && chown appuser:appuser /app

# Only the app package goes in — evals/, ui/, DATA/, docs/ stay out (see .dockerignore)
COPY --chown=appuser:appuser app/ ./app/

USER appuser

# Cloud Run injects PORT (default 8080); secrets/config come from runtime env vars, never from the image.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
