#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <backup.tar.gz>" >&2; exit 1; }
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$(realpath "$1")"
if tar -tzf "$ARCHIVE" | grep -Eq '^(/|.*\.\./)'; then
  echo "Unsafe archive paths" >&2
  exit 1
fi
cd "$ROOT_DIR"
docker compose stop manager
tar -xzf "$ARCHIVE"
mv -f data/manager.backup.db data/manager.db
docker compose up -d
