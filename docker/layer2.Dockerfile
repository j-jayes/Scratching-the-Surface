# Layer 2 — YOLOv8 specialist (CPU build for v1; swap base image for GPU later).
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

COPY models/yolo/best.pt /app/models/yolo/best.pt
ENV YOLO_MODEL_PATH=/app/models/yolo/best.pt \
    CONF_THRESHOLD=0.7

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.layer2_yolo.app:app", "--host", "0.0.0.0", "--port", "8000"]
