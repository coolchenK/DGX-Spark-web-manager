# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim AS frontend-builder

ENV PNPM_HOME=/pnpm
ENV PATH=$PNPM_HOME:$PATH
RUN corepack enable
WORKDIR /build/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

RUN groupadd --gid 10001 manager && useradd --uid 10001 --gid manager --create-home manager
WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY backend/ ./backend/
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
RUN python -m pip install . && chmod +x ./scripts/entrypoint.sh && mkdir -p /app/data && chown -R manager:manager /app
COPY --from=frontend-builder --chown=manager:manager /build/frontend/dist ./frontend/dist

USER manager
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/api/health', timeout=3)"]
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
