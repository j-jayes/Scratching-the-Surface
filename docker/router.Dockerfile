# Cascade router — orchestrates the three layers behind a single ingress.
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.router:app", "--host", "0.0.0.0", "--port", "8000"]
