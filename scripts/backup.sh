#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${1:-$ROOT_DIR/backups}"
mkdir -p "$BACKUP_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cd "$ROOT_DIR"
docker compose exec -T manager python -c "import sqlite3; src=sqlite3.connect('/app/data/manager.db'); dst=sqlite3.connect('/app/data/manager.backup.db'); src.backup(dst); dst.close(); src.close()"
tar -czf "$BACKUP_DIR/dgx-manager-$STAMP.tar.gz" data/manager.backup.db .env
rm -f data/manager.backup.db
echo "$BACKUP_DIR/dgx-manager-$STAMP.tar.gz"
