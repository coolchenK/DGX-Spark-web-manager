#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_PACKAGE="$ROOT_DIR/host_agent/dgx_ops_agent"
SOURCE_UNITS="$ROOT_DIR/deploy/systemd"
GROUP_NAME=dgx-spark-ops

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
fi

PACKAGE_PARENT="$INSTALL_ROOT/usr/local/lib/dgx-spark-ops-agent"
PACKAGE_DIR="$PACKAGE_PARENT/dgx_ops_agent"
KEY_DIR="$INSTALL_ROOT/etc/dgx-spark-manager"
KEY_FILE="$KEY_DIR/ops-agent.key"
UNIT_DIR="$INSTALL_ROOT/etc/systemd/system"
JOBS_DIR="$INSTALL_ROOT/var/lib/dgx-spark-ops-agent/jobs"
SOCKET_PATH="$INSTALL_ROOT/run/dgx-spark-manager/ops-agent.sock"
PACKAGE_BACKUP=""
PACKAGE_STAGING=""
PACKAGE_UPDATE_ACTIVE=false
PACKAGE_WAS_PRESENT=false
TRANSACTION_DIR=""
SOCKET_UNIT_EXISTED=false
SERVICE_UNIT_EXISTED=false
SOCKET_ENABLED_STATE=""
SOCKET_ACTIVE_STATE=""
SERVICE_ENABLED_STATE=""
SERVICE_ACTIVE_STATE=""
ROLLBACK_IN_PROGRESS=false
TRANSACTION_ACTIVE=false
SNAPSHOTS_READY=false
KEY_TEMP=""
KEY_CREATED_THIS_RUN=false

usage() {
  cat <<'EOF'
Usage: ./scripts/install-ops-agent.sh [--apply]

Without --apply, this command only previews the installation. Applying the
plan requires effective root; run:
  sudo ./scripts/install-ops-agent.sh --apply
EOF
}

fail() {
  echo "$*" >&2
  return 1
}

assert_exact_path() {
  local actual="$1"
  local expected="$2"
  [[ "$actual" == "$INSTALL_ROOT$expected" ]] || fail "Refusing unsafe path: $actual"
}

assert_no_symlink_chain() {
  local target="$1"
  local current=""
  local component
  IFS='/' read -r -a components <<< "${target#/}"
  for component in "${components[@]}"; do
    [[ -n "$component" ]] || continue
    current="$current/$component"
    [[ ! -L "$current" ]] || fail "Refusing symlinked install path: $current"
  done
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required"
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
    is-active:socket:inactive:4|\
    is-active:socket:not-found:3|is-active:socket:not-found:4|\
    is-active:service:active:0|is-active:service:inactive:3|\
    is-active:service:inactive:4|\
    is-active:service:not-found:3|is-active:service:not-found:4)
      return 0
      ;;
    *)
      echo "Unsafe systemd state response for $operation $unit_type: value='$value' status=$status" >&2
      return 1
      ;;
  esac
}

query_systemd_state() {
  local operation="$1"
  local unit="$2"
  local unit_type="$3"
  local result_variable="$4"
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
  if [[ "$status" == 4 ]]; then
    local unit_path="$UNIT_DIR/$unit"
    if [[ -e "$unit_path" || -L "$unit_path" ]]; then
      echo "Systemd reported an existing unit as unknown: $unit" >&2
      return 1
    fi
    value=not-found
  fi
  printf -v "$result_variable" '%s' "$value"
}

verify_systemd_state() {
  local operation="$1"
  local unit="$2"
  local unit_type="$3"
  local expected="$4"
  local actual=""
  query_systemd_state "$operation" "$unit" "$unit_type" actual || return 1
  [[ "$actual" == "$expected" ]] || {
    echo "Systemd state verification failed for $unit: expected $expected, found $actual" >&2
    return 1
  }
}

capture_systemd_snapshot() {
  query_systemd_state is-enabled dgx-spark-ops-agent.socket socket SOCKET_ENABLED_STATE
  query_systemd_state is-active dgx-spark-ops-agent.socket socket SOCKET_ACTIVE_STATE
  query_systemd_state is-enabled dgx-spark-ops-agent.service service SERVICE_ENABLED_STATE
  query_systemd_state is-active dgx-spark-ops-agent.service service SERVICE_ACTIVE_STATE
}

effective_uid() {
  if [[ "${DGX_OPS_AGENT_TESTING:-}" == "1" && -n "${DGX_OPS_AGENT_TEST_EUID:-}" ]]; then
    printf '%s\n' "$DGX_OPS_AGENT_TEST_EUID"
  else
    id -u
  fi
}

machine_architecture() {
  if [[ "${DGX_OPS_AGENT_TESTING:-}" == "1" && -n "${DGX_OPS_AGENT_TEST_ARCH:-}" ]]; then
    printf '%s\n' "$DGX_OPS_AGENT_TEST_ARCH"
  else
    uname -m
  fi
}

validate_key_content() {
  local path="$1"
  python3 - "$path" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("Agent key is not a regular file")
with open(path, "rb", buffering=0) as handle:
    value = handle.read(66)
if len(value) == 32:
    raise SystemExit(0)
if value.endswith(b"\n"):
    value = value[:-1]
if len(value) != 64 or any(byte not in b"0123456789abcdefABCDEF" for byte in value):
    raise SystemExit("Agent key must contain 32 raw bytes or 64 hexadecimal characters")
PY
}

validate_key_metadata() {
  local path="$1"
  [[ ! -L "$path" ]] || fail "Agent key must not be a symlink"
  local metadata
  metadata="$(stat -c '%U:%G:%a' -- "$path")"
  [[ "$metadata" == "root:dgx-spark-ops:640" ]] || {
    fail "Existing Agent key must be root:dgx-spark-ops:640 (found $metadata)"
  }
}

validate_existing_key() {
  validate_key_metadata "$KEY_FILE"
  validate_key_content "$KEY_FILE"
}

create_key_if_missing() {
  assert_exact_path "$KEY_FILE" /etc/dgx-spark-manager/ops-agent.key
  assert_no_symlink_chain "$KEY_DIR"
  install -d -o root -g root -m 0750 -- "$KEY_DIR"
  if [[ -e "$KEY_FILE" || -L "$KEY_FILE" ]]; then
    validate_existing_key
    echo "Preserving validated Agent key at /etc/dgx-spark-manager/ops-agent.key"
    return
  fi

  KEY_TEMP="$(mktemp "$KEY_DIR/.ops-agent.key.XXXXXX")"
  openssl rand -hex 32 > "$KEY_TEMP"
  validate_key_content "$KEY_TEMP"
  chown root:"$GROUP_NAME" -- "$KEY_TEMP"
  chmod 0640 -- "$KEY_TEMP"
  validate_key_metadata "$KEY_TEMP"
  [[ ! -e "$KEY_FILE" && ! -L "$KEY_FILE" ]] || {
    fail "Agent key appeared during installation; rerun after validating it"
  }
  mv -- "$KEY_TEMP" "$KEY_FILE"
  KEY_TEMP=""
  KEY_CREATED_THIS_RUN=true
  validate_existing_key
  echo "Created Agent key at /etc/dgx-spark-manager/ops-agent.key"
}

install_package() {
  assert_exact_path "$PACKAGE_DIR" /usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent
  assert_no_symlink_chain "$PACKAGE_PARENT"
  install -d -o root -g root -m 0755 -- "$PACKAGE_PARENT"

  local STAGED_PACKAGE
  PACKAGE_STAGING="$(mktemp -d "$PACKAGE_PARENT/.install.XXXXXX")"
  STAGED_PACKAGE="$PACKAGE_STAGING/dgx_ops_agent"
  install -d -o root -g root -m 0755 -- "$STAGED_PACKAGE"
  local source_file
  for source_file in "$SOURCE_PACKAGE"/*.py; do
    install -o root -g root -m 0644 -- "$source_file" "$STAGED_PACKAGE/$(basename "$source_file")"
  done
  python3 -m compileall -q "$STAGED_PACKAGE"

  if [[ -e "$PACKAGE_DIR" || -L "$PACKAGE_DIR" ]]; then
    [[ -d "$PACKAGE_DIR" && ! -L "$PACKAGE_DIR" ]] || fail "Existing Agent package path is unsafe"
    PACKAGE_BACKUP="$PACKAGE_PARENT/.previous.$$"
    [[ ! -e "$PACKAGE_BACKUP" && ! -L "$PACKAGE_BACKUP" ]] || fail "Agent package backup path already exists"
    mv -- "$PACKAGE_DIR" "$PACKAGE_BACKUP"
  fi

  if ! mv -- "$STAGED_PACKAGE" "$PACKAGE_DIR"; then
    if [[ -n "$PACKAGE_BACKUP" && ! -e "$PACKAGE_DIR" ]]; then
      mv -- "$PACKAGE_BACKUP" "$PACKAGE_DIR"
    fi
    fail "Package update failed; the previous package was restored where possible"
  fi
  PACKAGE_UPDATE_ACTIVE=true
  rmdir -- "$PACKAGE_STAGING"
  PACKAGE_STAGING=""
}

snapshot_unit() {
  local path="$1"
  local name="$2"
  local state_variable="$3"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -f "$path" && ! -L "$path" ]] || fail "Existing systemd unit path is unsafe: $path"
    cp -a -- "$path" "$TRANSACTION_DIR/$name"
    printf -v "$state_variable" '%s' true
  else
    printf -v "$state_variable" '%s' false
  fi
}

snapshot_install_transaction() {
  assert_exact_path "$PACKAGE_DIR" /usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent
  assert_no_symlink_chain "$PACKAGE_PARENT"
  assert_no_symlink_chain "$UNIT_DIR"
  install -d -o root -g root -m 0755 -- "$PACKAGE_PARENT"
  TRANSACTION_DIR="$(mktemp -d "$PACKAGE_PARENT/.transaction.XXXXXX")"

  if [[ -e "$PACKAGE_DIR" || -L "$PACKAGE_DIR" ]]; then
    [[ -d "$PACKAGE_DIR" && ! -L "$PACKAGE_DIR" ]] || fail "Existing Agent package path is unsafe"
    PACKAGE_WAS_PRESENT=true
  else
    PACKAGE_WAS_PRESENT=false
  fi
  snapshot_unit "$UNIT_DIR/dgx-spark-ops-agent.socket" socket.unit SOCKET_UNIT_EXISTED
  snapshot_unit "$UNIT_DIR/dgx-spark-ops-agent.service" service.unit SERVICE_UNIT_EXISTED
  SNAPSHOTS_READY=true
}

install_unit() {
  local name="$1"
  local destination="$UNIT_DIR/$name"
  assert_exact_path "$destination" "/etc/systemd/system/$name"
  assert_no_symlink_chain "$UNIT_DIR"
  install -d -o root -g root -m 0755 -- "$UNIT_DIR"
  local temporary
  temporary="$(mktemp "$UNIT_DIR/.$name.XXXXXX")"
  install -o root -g root -m 0644 -- "$SOURCE_UNITS/$name" "$temporary"
  mv -f -- "$temporary" "$destination"
}

probe_health() {
  PYTHONPATH="$PACKAGE_PARENT" python3 - "$SOCKET_PATH" "$KEY_FILE" <<'PY'
import hashlib
import hmac
import socket
import sys
import time

from dgx_ops_agent.protocol import (
    PROTOCOL_VERSION,
    canonical_bytes,
    new_request,
    read_frame,
    sign_message,
    write_frame,
)

_RESPONSE_FIELDS = {
    "protocol_version",
    "request_id",
    "ok",
    "result",
    "error",
    "timestamp",
    "signature",
}

socket_path, key_path = sys.argv[1:]
with open(key_path, "rb", buffering=0) as handle:
    encoded = handle.read(66)
if len(encoded) == 32:
    secret = encoded
else:
    secret = bytes.fromhex(encoded.strip().decode("ascii"))
request = sign_message(new_request("agent.health", {}), secret)
connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
connection.settimeout(10)
try:
    connection.connect(socket_path)
    write_frame(connection, request)
    response = read_frame(connection)
finally:
    connection.close()
signature = response.get("signature")
if set(response) != _RESPONSE_FIELDS:
    raise SystemExit("Agent health response schema was invalid")
if not isinstance(signature, str):
    raise SystemExit("Agent health response was unsigned")
expected = hmac.new(secret, canonical_bytes(response), hashlib.sha256).hexdigest()
if not hmac.compare_digest(signature, expected):
    raise SystemExit("Agent health response signature was invalid")
if response.get("request_id") != request["request_id"]:
    raise SystemExit("Agent health response request_id did not match")
timestamp = response.get("timestamp")
if type(timestamp) is not int or abs(int(time.time()) - timestamp) > 30:
    raise SystemExit("Agent health response timestamp was invalid")
result = response.get("result")
if (
    response.get("protocol_version") != PROTOCOL_VERSION
    or response.get("ok") is not True
    or response.get("error") is not None
    or not isinstance(result, dict)
    or result.get("status") != "ok"
    or result.get("protocol_version") != PROTOCOL_VERSION
):
    raise SystemExit("Agent health check failed")
print("Agent health check passed")
PY
}

restore_package_snapshot() {
  local restored=true
  if [[ -n "$PACKAGE_BACKUP" && -d "$PACKAGE_BACKUP" && ! -L "$PACKAGE_BACKUP" ]]; then
    if [[ -e "$PACKAGE_DIR" || -L "$PACKAGE_DIR" ]]; then
      rm -rf --one-file-system -- "$PACKAGE_DIR" || restored=false
    fi
    if $restored; then
      mv -- "$PACKAGE_BACKUP" "$PACKAGE_DIR" || restored=false
    fi
  elif $PACKAGE_WAS_PRESENT; then
    [[ -d "$PACKAGE_DIR" && ! -L "$PACKAGE_DIR" ]] || restored=false
  elif [[ -e "$PACKAGE_DIR" || -L "$PACKAGE_DIR" ]]; then
    rm -rf --one-file-system -- "$PACKAGE_DIR" || restored=false
  fi
  PACKAGE_UPDATE_ACTIVE=false
  $restored
}

restore_unit_snapshot() {
  local destination="$1"
  local name="$2"
  local existed="$3"
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ ! -d "$destination" || -L "$destination" ]] || return 1
    rm -f -- "$destination" || return 1
  fi
  if $existed; then
    cp -a -- "$TRANSACTION_DIR/$name" "$destination" || return 1
  fi
}

restore_systemd_state() {
  local restored=true
  case "$SOCKET_ENABLED_STATE" in
    enabled) systemctl enable dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false ;;
    enabled-runtime) systemctl enable --runtime dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false ;;
    disabled) systemctl disable dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false ;;
    not-found) ;;
    *) restored=false ;;
  esac
  case "$SERVICE_ENABLED_STATE" in
    enabled) systemctl enable dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false ;;
    enabled-runtime) systemctl enable --runtime dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false ;;
    disabled) systemctl disable dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false ;;
    static|not-found) ;;
    *) restored=false ;;
  esac

  case "$SOCKET_ACTIVE_STATE" in
    active) systemctl start dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false ;;
    inactive) systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false ;;
    not-found) ;;
    *) restored=false ;;
  esac
  case "$SERVICE_ACTIVE_STATE" in
    active) systemctl start dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false ;;
    inactive) systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false ;;
    not-found) ;;
    *) restored=false ;;
  esac

  if [[ "$SOCKET_ACTIVE_STATE" == active || "$SERVICE_ACTIVE_STATE" == active ]]; then
    probe_health || restored=false
    if [[ "$SERVICE_ACTIVE_STATE" == inactive ]]; then
      systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false
    fi
    if [[ "$SOCKET_ACTIVE_STATE" == inactive ]]; then
      systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false
    fi
  fi
  verify_systemd_state is-enabled dgx-spark-ops-agent.socket socket "$SOCKET_ENABLED_STATE" || restored=false
  verify_systemd_state is-active dgx-spark-ops-agent.socket socket "$SOCKET_ACTIVE_STATE" || restored=false
  verify_systemd_state is-enabled dgx-spark-ops-agent.service service "$SERVICE_ENABLED_STATE" || restored=false
  verify_systemd_state is-active dgx-spark-ops-agent.service service "$SERVICE_ACTIVE_STATE" || restored=false
  $restored
}

rollback_install_transaction() {
  local restored=true
  set +e
  ROLLBACK_IN_PROGRESS=true
  if $SNAPSHOTS_READY; then
    if [[ -e "$UNIT_DIR/dgx-spark-ops-agent.service" ]]; then
      systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false
      systemctl disable dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false
    fi
    if [[ -e "$UNIT_DIR/dgx-spark-ops-agent.socket" ]]; then
      systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false
      systemctl disable dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false
    fi
    restore_package_snapshot || restored=false
    restore_unit_snapshot "$UNIT_DIR/dgx-spark-ops-agent.socket" socket.unit "$SOCKET_UNIT_EXISTED" || restored=false
    restore_unit_snapshot "$UNIT_DIR/dgx-spark-ops-agent.service" service.unit "$SERVICE_UNIT_EXISTED" || restored=false
    systemctl daemon-reload || restored=false
  fi
  restore_systemd_state || restored=false
  if [[ -n "$PACKAGE_STAGING" && "$PACKAGE_STAGING" == "$PACKAGE_PARENT"/.install.* ]]; then
    rm -rf --one-file-system -- "$PACKAGE_STAGING" || restored=false
  fi
  if [[ -n "$KEY_TEMP" && "$KEY_TEMP" == "$KEY_DIR"/.ops-agent.key.* ]]; then
    rm -f -- "$KEY_TEMP" || restored=false
  fi
  if [[ -n "$PACKAGE_BACKUP" && -e "$PACKAGE_BACKUP" ]]; then
    restored=false
  else
    PACKAGE_BACKUP=""
  fi
  if [[ -n "$TRANSACTION_DIR" && "$TRANSACTION_DIR" == "$PACKAGE_PARENT"/.transaction.* ]]; then
    rm -rf --one-file-system -- "$TRANSACTION_DIR" || restored=false
  fi
  TRANSACTION_DIR=""
  PACKAGE_STAGING=""
  KEY_TEMP=""
  PACKAGE_UPDATE_ACTIVE=false
  SNAPSHOTS_READY=false
  ROLLBACK_IN_PROGRESS=false
  if $restored; then
    return 0
  fi
  return 1
}

clear_transaction_traps() {
  trap - EXIT HUP INT TERM
}

transaction_exit() {
  local original_status="$?"
  if $ROLLBACK_IN_PROGRESS; then
    clear_transaction_traps
    exit "$original_status"
  fi
  clear_transaction_traps
  if $TRANSACTION_ACTIVE; then
    if rollback_install_transaction; then
      echo "Agent installation failed; the previous installation was restored." >&2
    else
      echo "Agent installation failed and automatic rollback was incomplete." >&2
      echo "Inspect systemctl status and the protected Agent install paths before retrying." >&2
    fi
  fi
  exit "$original_status"
}

begin_install_transaction() {
  TRANSACTION_ACTIVE=true
  trap transaction_exit EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

commit_install_transaction() {
  TRANSACTION_ACTIVE=false
  clear_transaction_traps
  if [[ -n "$PACKAGE_BACKUP" ]]; then
    assert_exact_path "$PACKAGE_BACKUP" "/usr/local/lib/dgx-spark-ops-agent/.previous.$$"
    rm -rf --one-file-system -- "$PACKAGE_BACKUP" || {
      echo "Warning: could not remove previous Agent package backup: $PACKAGE_BACKUP" >&2
    }
  fi
  if [[ -n "$TRANSACTION_DIR" && "$TRANSACTION_DIR" == "$PACKAGE_PARENT"/.transaction.* ]]; then
    rm -rf --one-file-system -- "$TRANSACTION_DIR" || {
      echo "Warning: could not remove Agent transaction snapshot: $TRANSACTION_DIR" >&2
    }
  fi
  PACKAGE_BACKUP=""
  TRANSACTION_DIR=""
  PACKAGE_UPDATE_ACTIVE=false
  SNAPSHOTS_READY=false
}

quiesce_existing_agent() {
  if [[ "$SERVICE_ACTIVE_STATE" == active ]]; then
    systemctl stop dgx-spark-ops-agent.service
    verify_systemd_state is-active dgx-spark-ops-agent.service service inactive
  fi
  if [[ "$SOCKET_ACTIVE_STATE" == active ]]; then
    systemctl stop dgx-spark-ops-agent.socket
    verify_systemd_state is-active dgx-spark-ops-agent.socket socket inactive
  fi
}

activate_new_agent() {
  systemctl daemon-reload
  systemctl disable dgx-spark-ops-agent.service
  verify_systemd_state is-enabled dgx-spark-ops-agent.service service static
  systemctl enable dgx-spark-ops-agent.socket
  verify_systemd_state is-enabled dgx-spark-ops-agent.socket socket enabled
  systemctl start dgx-spark-ops-agent.socket
  verify_systemd_state is-active dgx-spark-ops-agent.socket socket active
  systemctl restart dgx-spark-ops-agent.service
  verify_systemd_state is-active dgx-spark-ops-agent.service service active
  probe_health
}

apply_installation() {
  local arch="$1"
  [[ "$arch" == "aarch64" || "$arch" == "arm64" ]] || fail "Unsupported architecture: $arch"
  [[ "$(effective_uid)" == "0" ]] || {
    fail "Installation requires effective root. Run: sudo ./scripts/install-ops-agent.sh --apply"
  }

  local dependency
  for dependency in basename chown chmod cp dirname getent groupadd id install ln mktemp mv openssl python3 rm rmdir stat systemctl uname; do
    require_command "$dependency"
  done
  [[ -d "$SOURCE_PACKAGE" ]] || fail "Agent package source is missing"
  [[ -f "$SOURCE_UNITS/dgx-spark-ops-agent.service" ]] || fail "Agent service unit is missing"
  [[ -f "$SOURCE_UNITS/dgx-spark-ops-agent.socket" ]] || fail "Agent socket unit is missing"

  capture_systemd_snapshot
  begin_install_transaction
  quiesce_existing_agent
  snapshot_install_transaction

  if ! getent group "$GROUP_NAME" >/dev/null; then
    groupadd --system "$GROUP_NAME"
  fi
  getent group "$GROUP_NAME" >/dev/null || fail "Could not resolve $GROUP_NAME group"
  assert_exact_path "$JOBS_DIR" /var/lib/dgx-spark-ops-agent/jobs
  assert_no_symlink_chain "$JOBS_DIR"
  install -d -o root -g root -m 0750 -- "$(dirname "$JOBS_DIR")"
  install -d -o root -g root -m 0750 -- "$JOBS_DIR"
  create_key_if_missing
  install_package
  install_unit dgx-spark-ops-agent.service
  install_unit dgx-spark-ops-agent.socket
  activate_new_agent
  commit_install_transaction
  echo "DGX Spark host operations agent installed"
}

main() {
  local apply=false
  case "$#:${1:-}" in
    0:) ;;
    1:--apply) apply=true ;;
    *)
      echo "Unknown argument: ${1:-}" >&2
      usage >&2
      return 2
      ;;
  esac

  local arch
  arch="$(machine_architecture)"
  cat <<EOF
DGX Spark host operations agent installation plan
  Architecture: $arch
  Package:      /usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent
  Key:          /etc/dgx-spark-manager/ops-agent.key (preserved on updates)
  Socket:       /run/dgx-spark-manager/ops-agent.sock
  Job logs:     /var/lib/dgx-spark-ops-agent/jobs (preserved on updates)
  Rollback:     ./scripts/uninstall-ops-agent.sh --apply
EOF
  if [[ "$apply" != true ]]; then
    echo
    echo "Preview only. Run sudo ./scripts/install-ops-agent.sh --apply to continue."
    return 0
  fi
  apply_installation "$arch"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
