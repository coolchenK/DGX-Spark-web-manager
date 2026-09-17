#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="dgx-spark-web-manager.service"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
ENV_FILE="$ROOT_DIR/data/native.env"
VENV_DIR="$ROOT_DIR/.venv-native"
APPLY=false
[[ "${1:-}" == "--apply" ]] && APPLY=true

HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
MODEL_HOME_HOST="${MODEL_HOME_HOST:-$HOME/models}"
HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
HOST_IP="${HOST_IP:-127.0.0.1}"

echo "DGX Spark Web Manager native installation plan"
echo "  Architecture:     $(uname -m)"
echo "  Service:          user systemd unit $SERVICE_FILE"
echo "  Python runtime:   $VENV_DIR"
echo "  Persistent data: $ROOT_DIR/data"
echo "  HF cache:         $HF_HOME_HOST"
echo "  Model root:       $MODEL_HOME_HOST"
echo "  Estimated disk:   1-2 GiB for Python/Node dependencies; model files excluded"
echo "  Commands:         python venv + pip install, pnpm install/build, alembic upgrade, systemctl --user enable --now"
echo "  Rollback:         ./scripts/uninstall-native.sh --apply (data and models retained)"

if ! $APPLY; then
  echo
  echo "Preview only. Run ./scripts/install-native.sh --apply to continue."
  exit 0
fi

ARCH="$(uname -m)"
[[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] || { echo "Unsupported architecture: $ARCH" >&2; exit 1; }
for command in python3 node corepack docker systemctl openssl curl; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done
python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || { echo "Python 3.11+ is required" >&2; exit 1; }
docker info >/dev/null || { echo "The current user must be able to access Docker" >&2; exit 1; }

mkdir -p "$ROOT_DIR/data" "$HF_HOME_HOST/hub" "$MODEL_HOME_HOST" "$SERVICE_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
  SECRET_KEY="$(openssl rand -base64 48 | tr -d '\n')"
  ADMIN_PASSWORD="$(openssl rand -base64 24 | tr -d '\n')"
  umask 077
  cat > "$ENV_FILE" <<EOF
DGX_SECRET_KEY=$SECRET_KEY
DGX_ADMIN_USERNAME=admin
DGX_ADMIN_PASSWORD=$ADMIN_PASSWORD
DGX_COOKIE_SECURE=false
DGX_ALLOWED_ORIGINS=http://$HOST_IP:3000
DGX_DATABASE_URL=sqlite:///$ROOT_DIR/data/manager.db
DGX_DATA_DIR=$ROOT_DIR/data
DGX_STATIC_DIR=$ROOT_DIR/frontend/dist
DGX_HOST_OS_RELEASE=/etc/os-release
DGX_MODEL_ROOTS=$MODEL_HOME_HOST,$HF_HOME_HOST/hub
DGX_HF_CACHE_DIR=$HF_HOME_HOST/hub
DGX_AUTO_DISCOVERY=true
EOF
  chmod 600 "$ENV_FILE"
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "$ROOT_DIR"
(
  cd "$ROOT_DIR/frontend"
  corepack pnpm install --frozen-lockfile
  corepack pnpm build
)

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=DGX Spark Web Manager
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
EnvironmentFile=$ENV_FILE
ExecStartPre=$VENV_DIR/bin/alembic -c $ROOT_DIR/alembic.ini upgrade head
ExecStart=$VENV_DIR/bin/python -m uvicorn app.main:app --app-dir $ROOT_DIR/backend --host 0.0.0.0 --port 3000
Restart=on-failure
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:3000/api/health >/dev/null; then
    echo "Manager is healthy: http://$HOST_IP:3000"
    echo "Administrator username: $(grep '^DGX_ADMIN_USERNAME=' "$ENV_FILE" | cut -d= -f2-)"
    echo "Administrator password: $(grep '^DGX_ADMIN_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)"
    exit 0
  fi
  sleep 2
done

journalctl --user -u "$SERVICE_NAME" -n 100 --no-pager
echo "Manager did not become healthy; inspect the journal above." >&2
exit 1
