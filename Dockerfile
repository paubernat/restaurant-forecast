# syntax=docker/dockerfile:1.7
# Multi-stage build. The image is CPU-only and carries NO TimesFM checkpoint or torch:
# TimesFM is served over HTTP by a Hugging Face Space (see space/ and docs/05-timesfm-hybrid.md),
# so the app only needs `requests`. The deps layer is cached; application source is copied
# LAST so editing code never invalidates it.

# ---- builder: install deps --------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build

ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
# Install deps first (cached unless pyproject changes). Falls back to the non-dev set if
# the dev extras fail to resolve in a minimal builder.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install ".[dev]" || pip install --prefix=/install "."

# ---- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

COPY --from=builder /install /usr/local

# Bundled data + config copied before source so they cache independently.
COPY data ./data
COPY pyproject.toml Makefile ./

# Source LAST — code edits rebuild only this layer.
COPY src ./src

CMD ["python", "-m", "forecasting", "evaluate"]
