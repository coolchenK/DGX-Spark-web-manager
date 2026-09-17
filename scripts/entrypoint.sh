#!/usr/bin/env sh
set -eu

mkdir -p /app/data
alembic -c /app/alembic.ini upgrade head

exec uvicorn app.main:app \
  --host "${DGX_LISTEN_HOST:-0.0.0.0}" \
  --port "${DGX_LISTEN_PORT:-3000}" \
  --proxy-headers
