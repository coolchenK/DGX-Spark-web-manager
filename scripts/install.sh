#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APPLY=false
[[ "${1:-}" == "--apply" ]] && APPLY=true

ARCH="$(uname -m)"
DOCKER_GID="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || true)"
PUID="$(id -u)"
PGID="$(id -g)"
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
MODEL_HOME_HOST="${MODEL_HOME_HOST:-$HOME/models}"
HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
HOST_IP="${HOST_IP:-127.0.0.1}"

echo "DGX Spark Web Manager installation plan"
echo "  Architecture: $ARCH"
echo "  URL:          http://$HOST_IP:3000"
echo "  HF cache:     $HF_HOME_HOST"
echo "  Model root:   $MODEL_HOME_HOST"
echo "  Data:         $ROOT_DIR/data"
echo "  Changes:      build one ARM64 image, create one manager container, create .env/data"
echo "  Commands:     docker compose build; docker compose up -d; health check"
echo "  Estimated:    2-5 GiB temporary/build image space; model downloads excluded"
echo "  Rollback:     ./scripts/uninstall.sh (model files and existing inference containers remain)"

[[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] || { echo "Unsupported architecture: $ARCH" >&2; exit 1; }
command -v docker >/dev/null || { echo "Docker is required" >&2; exit 1; }
docker compose version >/dev/null || { echo "Docker Compose is required" >&2; exit 1; }
[[ -n "$DOCKER_GID" ]] || { echo "Cannot access /var/run/docker.sock" >&2; exit 1; }

if ! $APPLY; then
  echo
  echo "Preview only. Run ./scripts/install.sh --apply to continue."
  exit 0
fi

mkdir -p data "$HF_HOME_HOST" "$MODEL_HOME_HOST"
if [[ ! -f .env ]]; then
  SECRET_KEY="$(openssl rand -base64 48 | tr -d '\n')"
  ADMIN_PASSWORD="$(openssl rand -base64 24 | tr -d '\n')"
  umask 077
  cat > .env <<EOF
DGX_SECRET_KEY=$SECRET_KEY
DGX_ADMIN_USERNAME=admin
DGX_ADMIN_PASSWORD=$ADMIN_PASSWORD
DGX_COOKIE_SECURE=false
DGX_ALLOWED_ORIGINS=http://$HOST_IP:3000
PUID=$PUID
PGID=$PGID
DOCKER_GID=$DOCKER_GID
HF_HOME_HOST=$HF_HOME_HOST
MODEL_HOME_HOST=$MODEL_HOME_HOST
EOF
  echo "Generated administrator credentials in $ROOT_DIR/.env"
fi

docker compose build
docker compose up -d

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:3000/api/health" >/dev/null; then
    echo "Manager is healthy: http://$HOST_IP:3000"
    echo "Administrator username: $(grep '^DGX_ADMIN_USERNAME=' .env | cut -d= -f2-)"
    echo "Administrator password: $(grep '^DGX_ADMIN_PASSWORD=' .env | cut -d= -f2-)"
    exit 0
  fi
  sleep 2
done

docker compose logs --tail=100 manager
echo "Manager did not become healthy; inspect the logs above." >&2
exit 1
