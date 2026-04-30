# Cascade router — orchestrates the three layers behind a single ingress.
# Also serves the public demo HTML page from /app/static.
ARG REGISTRY=local
ARG BASE_TAG=latest
FROM ${REGISTRY}/cascade-base:${BASE_TAG}

# Static demo assets (HTML + JS + CSS + 3 example images).
COPY src/cascade_defect/static/ /app/static/
ENV STATIC_DIR=/app/static

EXPOSE 8000
CMD ["uvicorn", "cascade_defect.router:app", "--host", "0.0.0.0", "--port", "8000"]
