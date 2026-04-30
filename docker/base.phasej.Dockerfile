# Phase J overlay: re-uses cascade-base:latest already in ACR.
FROM cascadedevacr6ya7a3.azurecr.io/cascade-base:latest
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN /usr/local/bin/uv sync --no-dev --frozen --no-install-project
COPY src/ /app/src/
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-c", "raise SystemExit('Override CMD with the layer to run.')"]
