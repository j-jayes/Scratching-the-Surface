# Layer 1 — Convolutional Autoencoder gatekeeper
# Image is small (~1 GB) since torch CPU wheel is used.
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

# Models are mounted via Azure Blob Storage on startup; baked-in default for
# offline smoke tests:
COPY models/autoencoder/best.pt /app/models/autoencoder/best.pt
ENV MODEL_PATH=/app/models/autoencoder/best.pt \
    MSE_THRESHOLD=0.0013

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.layer1_autoencoder.app:app", "--host", "0.0.0.0", "--port", "8000"]
