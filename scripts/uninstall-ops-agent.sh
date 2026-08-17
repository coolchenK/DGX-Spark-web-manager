#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR=/usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent
KEY_FILE=/etc/dgx-spark-manager/ops-agent.key
JOBS_DIR=/var/lib/dgx-spark-ops-agent/jobs
SERVICE_UNIT=/etc/systemd/system/dgx-spark-ops-agent.service
SOCKET_UNIT=/etc/systemd/system/dgx-spark-ops-agent.socket
SOCKET_PATH=/run/dgx-spark-manager/ops-agent.sock
SOCKET_DIR=/run/dgx-spark-manager
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
  SOCKET_DIR="$INSTALL_ROOT$SOCKET_DIR"
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
      if [[ ! -e "$current" ]]; then
        [[ ! -e "$path" && ! -L "$path" ]] || {
          fail "Refusing target below a missing removal ancestor: $path"
        }
        return 0
      fi
    fi
  done
  if [[ -e "$path" ]]; then
    case "$expected_type" in
      directory) [[ -d "$path" ]] || fail "Refusing non-directory removal target: $path" ;;
      file) [[ -f "$path" ]] || fail "Refusing non-file removal target: $path" ;;
      socket) [[ -S "$path" ]] || fail "Refusing non-socket removal target: $path" ;;
      *) fail "Invalid removal target type" ;;
    esac
  fi
}

validate_all_removal_targets() {
  assert_trusted_path_chain "$PACKAGE_DIR" /usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent directory
  assert_trusted_path_chain "$KEY_FILE" /etc/dgx-spark-manager/ops-agent.key file
  assert_trusted_path_chain "$JOBS_DIR" /var/lib/dgx-spark-ops-agent/jobs directory
  assert_trusted_path_chain "$SERVICE_UNIT" /etc/systemd/system/dgx-spark-ops-agent.service file
  assert_trusted_path_chain "$SOCKET_UNIT" /etc/systemd/system/dgx-spark-ops-agent.socket file
  assert_trusted_path_chain "$SOCKET_PATH" /run/dgx-spark-manager/ops-agent.sock socket
  assert_trusted_path_chain "$SOCKET_DIR" /run/dgx-spark-manager directory
}

validate_systemd_state() {
  local operation="$1"
  local unit_type="$2"
  local value="$3"
  local status="$4"
  case "$operation:$unit_type:$value:$status" in
    is-enabled:socket:enabled:0|is-enabled:socket:enabled-runtime:0|\
    is-enabled:socket:disabled:1|is-enabled:socket:not-found:1|\
    is-enabled:socket:not-found:4|\
    is-enabled:service:enabled:0|is-enabled:service:enabled-runtime:0|\
    is-enabled:service:disabled:1|is-enabled:service:static:0|\
    is-enabled:service:not-found:1|is-enabled:service:not-found:4|\
    is-active:socket:active:0|is-active:socket:inactive:3|\
    is-active:socket:not-found:3|is-active:socket:not-found:4|\
    is-active:service:active:0|is-active:service:inactive:3|\
    is-active:service:not-found:3|is-active:service:not-found:4)
      return 0
      ;;
    *)
      echo "Unsafe systemd state response: operation=$operation unit_type=$unit_type value='$value' status=$status" >&2
      return 1
      ;;
  esac
}

query_systemd_state() {
  local operation="$1"
  local unit="$2"
  local result_variable="$3"
  local unit_type="${unit##*.}"
  local value
  local status
  if value="$(systemctl "$operation" "$unit" 2>/dev/null)"; then
    status=0
  else
    status=$?
  fi
  [[ -n "$value" && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || {
    echo "Invalid systemd state output for $unit" >&2
    return 1
  }
  validate_systemd_state "$operation" "$unit_type" "$value" "$status" || return 1
  printf -v "$result_variable" '%s' "$value"
}

verify_systemd_state() {
  local operation="$1"
  local unit="$2"
  local expected="$3"
  local actual=""
  query_systemd_state "$operation" "$unit" actual || return 1
  [[ "$actual" == "$expected" ]] || {
    echo "Systemd state verification failed for $unit: expected $expected, found $actual" >&2
    return 1
  }
}

verify_systemd_disabled_or_static() {
  local unit="$1"
  local actual=""
  query_systemd_state is-enabled "$unit" actual || return 1
  case "$actual" in
    disabled|static) ;;
    *)
      echo "Systemd enablement verification failed for $unit: found $actual" >&2
      return 1
      ;;
  esac
}

strict_shutdown() {
  local socket_enabled=""
  local socket_active=""
  local service_enabled=""
  local service_active=""
  query_systemd_state is-enabled dgx-spark-ops-agent.socket socket_enabled
  query_systemd_state is-active dgx-spark-ops-agent.socket socket_active
  query_systemd_state is-enabled dgx-spark-ops-agent.service service_enabled
  query_systemd_state is-active dgx-spark-ops-agent.service service_active

  if [[ "$service_active" == active ]]; then
    systemctl stop dgx-spark-ops-agent.service
    verify_systemd_state is-active dgx-spark-ops-agent.service inactive
  fi
  if [[ "$socket_active" == active ]]; then
    systemctl stop dgx-spark-ops-agent.socket
    verify_systemd_state is-active dgx-spark-ops-agent.socket inactive
  fi
  case "$socket_enabled" in
    enabled)
      systemctl disable dgx-spark-ops-agent.socket
      verify_systemd_state is-enabled dgx-spark-ops-agent.socket disabled
      ;;
    enabled-runtime)
      systemctl disable --runtime dgx-spark-ops-agent.socket
      verify_systemd_state is-enabled dgx-spark-ops-agent.socket disabled
      ;;
    disabled|not-found) ;;
  esac
  case "$service_enabled" in
    enabled)
      systemctl disable dgx-spark-ops-agent.service
      verify_systemd_disabled_or_static dgx-spark-ops-agent.service
      ;;
    enabled-runtime)
      systemctl disable --runtime dgx-spark-ops-agent.service
      verify_systemd_disabled_or_static dgx-spark-ops-agent.service
      ;;
    disabled|static|not-found) ;;
  esac
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
  command -v find >/dev/null 2>&1 || fail "find is required"
  command -v systemctl >/dev/null 2>&1 || fail "systemctl is required"

  if $purge_key; then
    confirm_exact "PURGE OPS AGENT KEY"
  fi
  if $purge_jobs; then
    confirm_exact "PURGE OPS AGENT JOBS"
  fi

  validate_all_removal_targets
  strict_shutdown
  assert_trusted_path_chain "$PACKAGE_DIR" /usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent directory
  rm -rf --one-file-system -- "$PACKAGE_DIR"
  assert_trusted_path_chain "$SERVICE_UNIT" /etc/systemd/system/dgx-spark-ops-agent.service file
  rm -f -- "$SERVICE_UNIT"
  assert_trusted_path_chain "$SOCKET_UNIT" /etc/systemd/system/dgx-spark-ops-agent.socket file
  rm -f -- "$SOCKET_UNIT"
  assert_trusted_path_chain "$SOCKET_PATH" /run/dgx-spark-manager/ops-agent.sock socket
  if [[ -e "$SOCKET_PATH" ]]; then
    rm -f -- "$SOCKET_PATH"
  fi
  assert_trusted_path_chain "$SOCKET_DIR" /run/dgx-spark-manager directory
  if [[ -d "$SOCKET_DIR" && -z "$(find "$SOCKET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rmdir -- "$SOCKET_DIR"
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
