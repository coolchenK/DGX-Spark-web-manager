#!/usr/bin/env sh
set -eu

mkdir -p /app/data
alembic -c /app/alembic.ini upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port 3000 --proxy-headers
