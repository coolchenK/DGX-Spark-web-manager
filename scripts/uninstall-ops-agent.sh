#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR=/usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent
KEY_FILE=/etc/dgx-spark-manager/ops-agent.key
JOBS_DIR=/var/lib/dgx-spark-ops-agent/jobs
SERVICE_UNIT=/etc/systemd/system/dgx-spark-ops-agent.service
SOCKET_UNIT=/etc/systemd/system/dgx-spark-ops-agent.socket
SOCKET_PATH=/run/dgx-spark-manager/ops-agent.sock
INSTALL_ROOT=""
if [[ -n "${DGX_OPS_AGENT_INSTALL_ROOT:-}" ]]; then
  [[ "${DGX_OPS_AGENT_TESTING:-}" == "1" ]] || {
    echo "DGX_OPS_AGENT_INSTALL_ROOT is only available to the test harness" >&2
    exit 2
  }
  INSTALL_ROOT="${DGX_OPS_AGENT_INSTALL_ROOT%/}"
  [[ -n "$INSTALL_ROOT" && "$INSTALL_ROOT" == /* ]] || {
    echo "Test install root must be an absolute path" >&2
    exit 2
  }
  PACKAGE_DIR="$INSTALL_ROOT$PACKAGE_DIR"
  KEY_FILE="$INSTALL_ROOT$KEY_FILE"
  JOBS_DIR="$INSTALL_ROOT$JOBS_DIR"
  SERVICE_UNIT="$INSTALL_ROOT$SERVICE_UNIT"
  SOCKET_UNIT="$INSTALL_ROOT$SOCKET_UNIT"
  SOCKET_PATH="$INSTALL_ROOT$SOCKET_PATH"
fi

fail() {
  echo "$*" >&2
  exit 1
}

confirm_exact() {
  local expected="$1"
  local reply
  echo "Type '$expected' to continue:" >&2
  read -r reply
  [[ "$reply" == "$expected" ]] || fail "Confirmation did not match; no purge was performed"
}

assert_trusted_path_chain() {
  local path="$1"
  local expected="$2"
  local expected_type="$3"
  [[ "$path" == "$INSTALL_ROOT$expected" && "$path" == /* && "$path" != "/" ]] || {
    fail "Refusing unsafe removal path: $path"
  }
  local trusted_root="${INSTALL_ROOT:-/}"
  [[ -d "$trusted_root" && ! -L "$trusted_root" ]] || {
    fail "Refusing unsafe trusted root: $trusted_root"
  }

  local current="${INSTALL_ROOT:-}"
  local relative="${expected#/}"
  local component
  local -a components
  IFS='/' read -r -a components <<< "$relative"
  local last_index=$((${#components[@]} - 1))
  local index
  for index in "${!components[@]}"; do
    component="${components[$index]}"
    current="$current/$component"
    [[ ! -L "$current" ]] || fail "Refusing symlinked removal path: $current"
    if (( index < last_index )); then
      if [[ -e "$current" && ! -d "$current" ]]; then
        fail "Refusing non-directory removal ancestor: $current"
      fi
      [[ -e "$current" ]] || return 0
    fi
  done
  if [[ -e "$path" ]]; then
    case "$expected_type" in
      directory) [[ -d "$path" ]] || fail "Refusing non-directory removal target: $path" ;;
      file) [[ -f "$path" ]] || fail "Refusing non-file removal target: $path" ;;
      *) fail "Invalid removal target type" ;;
    esac
  fi
}

effective_uid() {
  if [[ "${DGX_OPS_AGENT_TESTING:-}" == "1" && -n "${DGX_OPS_AGENT_TEST_EUID:-}" ]]; then
    printf '%s\n' "$DGX_OPS_AGENT_TEST_EUID"
  else
    id -u
  fi
}

usage() {
  cat <<'EOF'
Usage: ./scripts/uninstall-ops-agent.sh [--apply] [--purge-key] [--purge-jobs]

The default is a read-only preview. Job logs and the authentication key are
preserved unless their explicit purge flags are supplied and confirmed.
EOF
}

main() {
  local apply=false
  local purge_key=false
  local purge_jobs=false
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --apply) apply=true ;;
      --purge-key) purge_key=true ;;
      --purge-jobs) purge_jobs=true ;;
      *) echo "Unknown argument: $1" >&2; usage >&2; return 2 ;;
    esac
    shift
  done
  if [[ "$apply" != true ]]; then
    echo "DGX Spark host operations agent uninstall plan"
    echo "  Remove service, socket unit, runtime socket, and installed package"
    echo "  Preserving job logs: $JOBS_DIR"
    echo "  Preserving Agent key: $KEY_FILE"
    $purge_key && echo "  Requested key purge (requires confirmation with --apply)"
    $purge_jobs && echo "  Requested job log purge (requires confirmation with --apply)"
    echo "Preview only. Add --apply to continue."
    return 0
  fi
  [[ "$(effective_uid)" == "0" ]] || fail "Uninstallation requires effective root"

  if $purge_key; then
    confirm_exact "PURGE OPS AGENT KEY"
  fi
  if $purge_jobs; then
    confirm_exact "PURGE OPS AGENT JOBS"
  fi

  systemctl disable --now dgx-spark-ops-agent.socket >/dev/null 2>&1 || true
  systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || true
  assert_trusted_path_chain "$PACKAGE_DIR" /usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent directory
  rm -rf --one-file-system -- "$PACKAGE_DIR"
  rm -f -- "$SERVICE_UNIT" "$SOCKET_UNIT"
  if [[ -S "$SOCKET_PATH" ]]; then
    rm -f -- "$SOCKET_PATH"
  fi
  systemctl daemon-reload

  if $purge_key; then
    assert_trusted_path_chain "$KEY_FILE" /etc/dgx-spark-manager/ops-agent.key file
    rm -f -- "$KEY_FILE"
  else
    echo "Preserving Agent key: $KEY_FILE"
  fi
  if $purge_jobs; then
    assert_trusted_path_chain "$JOBS_DIR" /var/lib/dgx-spark-ops-agent/jobs directory
    rm -rf --one-file-system -- "$JOBS_DIR"
  else
    echo "Preserving job logs: $JOBS_DIR"
  fi
  echo "DGX Spark host operations agent uninstalled"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
