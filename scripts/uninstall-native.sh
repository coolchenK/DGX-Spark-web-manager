#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_FILE="$SERVICE_DIR/dgx-spark-web-manager.service"
APPLY=false
PURGE=false
for argument in "$@"; do
  [[ "$argument" == "--apply" ]] && APPLY=true
  [[ "$argument" == "--purge" ]] && PURGE=true
done

echo "Native uninstall plan"
echo "  Stop and remove: $SERVICE_FILE and $ROOT_DIR/.venv-native"
echo "  Retain:          Hugging Face cache and model directories"
echo "  Manager data:    $([[ "$PURGE" == true ]] && echo remove || echo retain)"
if ! $APPLY; then
  echo "Preview only. Add --apply to continue."
  exit 0
fi

systemctl --user disable --now dgx-spark-web-manager.service 2>/dev/null || true
rm -f -- "$SERVICE_FILE"
systemctl --user daemon-reload
rm -rf -- "$ROOT_DIR/.venv-native"
if $PURGE; then
  DATA_DIR="$(realpath "$ROOT_DIR/data")"
  [[ "$DATA_DIR" == "$ROOT_DIR/data" ]] || { echo "Refusing unsafe purge target" >&2; exit 1; }
  rm -rf -- "$DATA_DIR"
fi
echo "Native manager installation removed; model files were retained."
