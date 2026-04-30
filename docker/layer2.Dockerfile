# Layer 2 — YOLOv8n metal-defect specialist (Phase L production weights:
# 50 ep / 640 px / T4, mAP50 0.50). CPU build; swap base for GPU later.
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

COPY models/yolo_metal/best.pt /app/models/yolo_metal/best.pt
ENV YOLO_MODEL_PATH=/app/models/yolo_metal/best.pt \
    CONF_THRESHOLD=0.5

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.layer2_yolo.app:app", "--host", "0.0.0.0", "--port", "8000"]
