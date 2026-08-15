#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
docker compose down
if [[ "${1:-}" == "--purge" ]]; then
  TARGET="$(realpath "$ROOT_DIR/data")"
  [[ "$TARGET" == "$ROOT_DIR/data" ]] || { echo "Refusing unsafe purge target" >&2; exit 1; }
  rm -rf -- "$TARGET"
  echo "Manager database and audit history removed. Model caches were retained."
else
  echo "Manager stopped. Data and all model files were retained."
fi
