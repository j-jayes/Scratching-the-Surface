FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── uv (fast Python package manager) ─────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ── Quarto CLI ─────────────────────────────────────────────────────────────
ARG QUARTO_VERSION=1.5.57
RUN wget -qO /tmp/quarto.deb \
      "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && dpkg -i /tmp/quarto.deb \
    && rm /tmp/quarto.deb

# ── Azure CLI ─────────────────────────────────────────────────────────────────
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# ── Project dependencies (installed at build time for fast container start) ──
WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# Install dev dependencies for the devcontainer
RUN uv sync

# Set up pre-commit
RUN uv run pre-commit install --install-hooks 2>/dev/null || true

USER vscode
