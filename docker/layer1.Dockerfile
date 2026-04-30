# Layer 1 — PatchCore-lite gatekeeper (Phase L production weights).
# Image is small (~1 GB) since torch CPU wheel is used.
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

# Bake Phase-L PatchCore banks + summary (per-domain calibrated knee).
# Legacy autoencoder weights are kept for back-compat smoke tests but are
# only used when SCORER=ae (default in this image is patchcore).
COPY models/patchcore_metal/ /app/models/patchcore_metal/
COPY models/autoencoder/best.pt /app/models/autoencoder/best.pt

ENV SCORER=patchcore \
    PATCHCORE_DIR=/app/models/patchcore_metal \
    MODEL_PATH=/app/models/autoencoder/best.pt \
    Z_THRESHOLD=3.0 \
    Z_THRESHOLD_SEVERSTAL=-0.5 \
    Z_THRESHOLD_KSDD2=1.0 \
    DEFAULT_DOMAIN=severstal \
    IMAGE_SIZE=224 \
    MSE_THRESHOLD=0.0013

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.layer1_autoencoder.app:app", "--host", "0.0.0.0", "--port", "8000"]
