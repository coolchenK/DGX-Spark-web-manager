#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv-native"
ENV_FILE="$ROOT_DIR/data/native.env"
[[ -x "$VENV_DIR/bin/python" && -f "$ENV_FILE" ]] || { echo "Native installation was not found" >&2; exit 1; }

mkdir -p "$ROOT_DIR/backups"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
set -a
source "$ENV_FILE"
set +a
"$VENV_DIR/bin/python" - <<PY
import sqlite3
source = sqlite3.connect("$ROOT_DIR/data/manager.db")
target = sqlite3.connect("$ROOT_DIR/backups/manager-$STAMP.db")
source.backup(target)
target.close()
source.close()
PY

"$VENV_DIR/bin/python" -m pip install --upgrade "$ROOT_DIR"
(
  cd "$ROOT_DIR/frontend"
  corepack pnpm install --frozen-lockfile
  corepack pnpm build
)
"$VENV_DIR/bin/alembic" -c "$ROOT_DIR/alembic.ini" upgrade head
systemctl --user restart dgx-spark-web-manager.service
systemctl --user --no-pager status dgx-spark-web-manager.service
