# syntax=docker/dockerfile:1.6
# ─── Base layer with uv + Python 3.11 ─────────────────────────────────────────
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ─── Build venv from project deps ─────────────────────────────────────────────
FROM base AS deps
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# ─── Final image ──────────────────────────────────────────────────────────────
FROM base AS runtime
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY src/ /app/src/
COPY pyproject.toml /app/
ENV PYTHONPATH=/app/src

# Default to Layer 1; override CMD in derived images or compose.
EXPOSE 8000
CMD ["python", "-c", "raise SystemExit('Override CMD with the layer to run.')"]
