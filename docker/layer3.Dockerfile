# Layer 3 — Oracle (Azure OpenAI gpt-4.1-mini)
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

# Few-shot seed images are baked in (small).
COPY data/splits/seed /app/data/splits/seed
ENV FEW_SHOT_SEED_DIR=/app/data/splits/seed

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.layer3_gpt4o.app:app", "--host", "0.0.0.0", "--port", "8000"]
