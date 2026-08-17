#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
  echo "$*" >&2
  exit 1
}

upsert_env() {
  local file="$1"
  local name="$2"
  local value="$3"
  [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ ]] || fail "Invalid environment key: $name"
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || fail "Invalid environment value for $name"
  local temporary
  temporary="$(mktemp "${file}.tmp.XXXXXX")"
  local found=false
  local line
  {
    while IFS= read -r line || [[ -n "$line" ]]; do
      if [[ "$line" == "$name="* ]]; then
        if [[ "$found" == false ]]; then
          printf '%s=%s\n' "$name" "$value"
          found=true
        fi
      else
        printf '%s\n' "$line"
      fi
    done
    if [[ "$found" == false ]]; then
      printf '%s=%s\n' "$name" "$value"
    fi
  } < "$file" > "$temporary"
  chmod 0600 "$temporary"
  mv -f -- "$temporary" "$file"
}

ensure_env_value() {
  local file="$1"
  local name="$2"
  local value="$3"
  if ! grep -q "^${name}=" "$file"; then
    upsert_env "$file" "$name" "$value"
  fi
}

main() {
  local apply=false
  case "$#:${1:-}" in
    0:) ;;
    1:--apply) apply=true ;;
    *) echo "Unknown argument: ${1:-}" >&2; return 2 ;;
  esac

  cd "$ROOT_DIR"
  local arch
  local docker_gid
  local puid
  local pgid
  local hf_home_host
  local model_home_host
  local host_ip
  arch="$(uname -m)"
  docker_gid="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || true)"
  puid="$(id -u)"
  pgid="$(id -g)"
  hf_home_host="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
  model_home_host="${MODEL_HOME_HOST:-$HOME/models}"
  host_ip="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
  host_ip="${host_ip:-127.0.0.1}"

  echo "DGX Spark Web Manager installation plan"
  echo "  Architecture: $arch"
  echo "  URL:          http://$host_ip:3000"
  echo "  HF cache:     $hf_home_host"
  echo "  Model root:   $model_home_host"
  echo "  Data:         $ROOT_DIR/data"
  echo "  Host Agent:   signed socket at /run/dgx-spark-manager/ops-agent.sock"
  echo "  Changes:      install the Host Agent, build one ARM64 image, create .env/data"
  echo "  Commands:     sudo Agent install; build and start the Compose manager"
  echo "  Estimated:    2-5 GiB temporary/build image space; model downloads excluded"
  echo "  Rollback:     ./scripts/uninstall.sh and sudo ./scripts/uninstall-ops-agent.sh --apply"
  echo
  ./scripts/install-ops-agent.sh

  if [[ "$apply" != true ]]; then
    echo
    echo "Preview only. Run ./scripts/install.sh --apply to continue."
    return 0
  fi

  [[ "$arch" == "aarch64" || "$arch" == "arm64" ]] || fail "Unsupported architecture: $arch"
  local dependency
  for dependency in awk cut curl docker getent grep mktemp mv openssl seq sudo tr; do
    command -v "$dependency" >/dev/null 2>&1 || fail "$dependency is required"
  done
  docker compose version >/dev/null || fail "Docker Compose is required"
  [[ -n "$docker_gid" ]] || fail "Cannot access /var/run/docker.sock"

  sudo ./scripts/install-ops-agent.sh --apply
  local ops_agent_gid
  ops_agent_gid="$(getent group dgx-spark-ops | awk -F: 'NR == 1 { print $3 }')"
  [[ "$ops_agent_gid" =~ ^[0-9]+$ ]] || fail "Cannot resolve dgx-spark-ops group GID"

  mkdir -p data "$hf_home_host" "$model_home_host"
  local ENV_FILE="$ROOT_DIR/.env"
  umask 077
  if [[ ! -e "$ENV_FILE" ]]; then
    local temporary_env
    temporary_env="$(mktemp "$ROOT_DIR/.env.tmp.XXXXXX")"
    : > "$temporary_env"
    chmod 0600 "$temporary_env"
    mv -- "$temporary_env" "$ENV_FILE"
  fi
  [[ -f "$ENV_FILE" && ! -L "$ENV_FILE" ]] || fail ".env must be a regular file, not a symlink"
  chmod 0600 "$ENV_FILE"

  ensure_env_value "$ENV_FILE" DGX_SECRET_KEY "$(openssl rand -base64 48 | tr -d '\n')"
  ensure_env_value "$ENV_FILE" DGX_ADMIN_USERNAME admin
  ensure_env_value "$ENV_FILE" DGX_ADMIN_PASSWORD "$(openssl rand -base64 24 | tr -d '\n')"
  ensure_env_value "$ENV_FILE" DGX_COOKIE_SECURE false
  ensure_env_value "$ENV_FILE" DGX_ALLOWED_ORIGINS "http://$host_ip:3000"

  upsert_env "$ENV_FILE" "PUID" "$puid"
  upsert_env "$ENV_FILE" "PGID" "$pgid"
  upsert_env "$ENV_FILE" "DOCKER_GID" "$docker_gid"
  upsert_env "$ENV_FILE" "OPS_AGENT_GID" "$ops_agent_gid"
  upsert_env "$ENV_FILE" "HF_HOME_HOST" "$hf_home_host"
  upsert_env "$ENV_FILE" "MODEL_HOME_HOST" "$model_home_host"
  upsert_env "$ENV_FILE" DGX_OPS_AGENT_SOCKET /run/dgx-spark-manager/ops-agent.sock
  upsert_env "$ENV_FILE" DGX_OPS_AGENT_KEY_FILE /run/secrets/ops-agent.key
  upsert_env "$ENV_FILE" DGX_OPS_AGENT_CONNECT_TIMEOUT_SECONDS 3
  upsert_env "$ENV_FILE" DGX_OPS_AGENT_READ_TIMEOUT_SECONDS 30
  upsert_env "$ENV_FILE" DGX_OPS_AGENT_OUTPUT_LIMIT_BYTES 1000000
  chmod 0600 "$ENV_FILE"

  docker compose build
  docker compose up -d

  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:3000/api/health" >/dev/null; then
      echo "Manager is healthy: http://$host_ip:3000"
      echo "Administrator username: $(grep '^DGX_ADMIN_USERNAME=' "$ENV_FILE" | cut -d= -f2-)"
      echo "Administrator password is stored in $ENV_FILE"
      return 0
    fi
    sleep 2
  done

  docker compose logs --tail=100 manager
  fail "Manager did not become healthy; inspect the logs above"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
