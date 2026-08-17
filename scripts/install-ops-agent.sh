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
SOCKET_WAS_ENABLED=false
SOCKET_WAS_ACTIVE=false
SERVICE_WAS_ENABLED=false
SERVICE_WAS_ACTIVE=false
ROLLBACK_IN_PROGRESS=false
TRANSACTION_READY=false

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

validate_existing_key() {
  [[ ! -L "$KEY_FILE" ]] || fail "Existing Agent key must not be a symlink"
  local metadata
  metadata="$(stat -c '%U:%G:%a' -- "$KEY_FILE")"
  [[ "$metadata" == "root:dgx-spark-ops:640" ]] || {
    fail "Existing Agent key must be root:dgx-spark-ops:640 (found $metadata)"
  }
  python3 - "$KEY_FILE" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("Existing Agent key is not a regular file")
with open(path, "rb", buffering=0) as handle:
    value = handle.read(66)
if len(value) == 32:
    raise SystemExit(0)
if value.endswith(b"\n"):
    value = value[:-1]
if len(value) != 64 or any(byte not in b"0123456789abcdefABCDEF" for byte in value):
    raise SystemExit("Existing Agent key must contain 32 raw bytes or 64 hexadecimal characters")
PY
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

  local temporary_key
  temporary_key="$(mktemp "$KEY_DIR/.ops-agent.key.XXXXXX")"
  trap 'rm -f -- "$temporary_key"' RETURN
  openssl rand -hex 32 > "$temporary_key"
  chown root:"$GROUP_NAME" -- "$temporary_key"
  chmod 0640 -- "$temporary_key"
  ln -- "$temporary_key" "$KEY_FILE" || fail "Agent key appeared during installation; rerun after validating it"
  rm -f -- "$temporary_key"
  trap - RETURN
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

  if systemctl is-enabled --quiet dgx-spark-ops-agent.socket; then
    SOCKET_WAS_ENABLED=true
  else
    SOCKET_WAS_ENABLED=false
  fi
  if systemctl is-active --quiet dgx-spark-ops-agent.socket; then
    SOCKET_WAS_ACTIVE=true
  else
    SOCKET_WAS_ACTIVE=false
  fi
  if systemctl is-enabled --quiet dgx-spark-ops-agent.service; then
    SERVICE_WAS_ENABLED=true
  else
    SERVICE_WAS_ENABLED=false
  fi
  if systemctl is-active --quiet dgx-spark-ops-agent.service; then
    SERVICE_WAS_ACTIVE=true
  else
    SERVICE_WAS_ACTIVE=false
  fi
  TRANSACTION_READY=true
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
  if $SOCKET_WAS_ENABLED; then
    systemctl enable dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false
  else
    systemctl disable dgx-spark-ops-agent.socket >/dev/null 2>&1 || true
  fi
  if $SERVICE_WAS_ENABLED; then
    systemctl enable dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false
  else
    systemctl disable dgx-spark-ops-agent.service >/dev/null 2>&1 || true
  fi

  if $SOCKET_WAS_ACTIVE; then
    systemctl start dgx-spark-ops-agent.socket >/dev/null 2>&1 || restored=false
  else
    systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1 || true
  fi
  if $SERVICE_WAS_ACTIVE; then
    systemctl start dgx-spark-ops-agent.service >/dev/null 2>&1 || restored=false
  else
    systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || true
  fi

  if $SOCKET_WAS_ACTIVE || $SERVICE_WAS_ACTIVE; then
    probe_health || restored=false
    if ! $SERVICE_WAS_ACTIVE; then
      systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || true
    fi
    if ! $SOCKET_WAS_ACTIVE; then
      systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1 || true
    fi
  fi
  $restored
}

rollback_install_transaction() {
  local prior_errexit=false
  local restored=true
  [[ "$-" == *e* ]] && prior_errexit=true
  set +e
  ROLLBACK_IN_PROGRESS=true
  if $TRANSACTION_READY; then
    systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1
    systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1
    restore_package_snapshot || restored=false
    restore_unit_snapshot "$UNIT_DIR/dgx-spark-ops-agent.socket" socket.unit "$SOCKET_UNIT_EXISTED" || restored=false
    restore_unit_snapshot "$UNIT_DIR/dgx-spark-ops-agent.service" service.unit "$SERVICE_UNIT_EXISTED" || restored=false
    systemctl daemon-reload || restored=false
    restore_systemd_state || restored=false
  fi
  if [[ -n "$PACKAGE_STAGING" && "$PACKAGE_STAGING" == "$PACKAGE_PARENT"/.install.* ]]; then
    rm -rf --one-file-system -- "$PACKAGE_STAGING"
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
  PACKAGE_UPDATE_ACTIVE=false
  TRANSACTION_READY=false
  ROLLBACK_IN_PROGRESS=false
  if $prior_errexit; then
    set -e
  fi
  if $restored; then
    return 0
  fi
  return 1
}

rollback_failed_install() {
  local original_status="$?"
  if $ROLLBACK_IN_PROGRESS; then
    trap - ERR
    exit "$original_status"
  fi
  trap - ERR
  if rollback_install_transaction; then
    echo "Agent installation failed; the previous installation was restored." >&2
  else
    echo "Agent installation failed and automatic rollback was incomplete." >&2
    echo "Inspect systemctl status and the protected Agent install paths before retrying." >&2
  fi
  exit "$original_status"
}

commit_install_transaction() {
  trap - ERR
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
  TRANSACTION_READY=false
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

  if ! getent group "$GROUP_NAME" >/dev/null; then
    groupadd --system "$GROUP_NAME"
  fi
  getent group "$GROUP_NAME" >/dev/null || fail "Could not resolve $GROUP_NAME group"

  assert_exact_path "$JOBS_DIR" /var/lib/dgx-spark-ops-agent/jobs
  assert_no_symlink_chain "$JOBS_DIR"
  install -d -o root -g root -m 0750 -- "$(dirname "$JOBS_DIR")"
  install -d -o root -g root -m 0750 -- "$JOBS_DIR"

  create_key_if_missing
  trap rollback_failed_install ERR
  snapshot_install_transaction
  systemctl stop dgx-spark-ops-agent.socket >/dev/null 2>&1 || true
  systemctl stop dgx-spark-ops-agent.service >/dev/null 2>&1 || true
  install_package
  install_unit dgx-spark-ops-agent.service
  install_unit dgx-spark-ops-agent.socket
  systemctl daemon-reload
  systemctl enable --now dgx-spark-ops-agent.socket
  systemctl try-restart dgx-spark-ops-agent.service >/dev/null 2>&1 || true
  probe_health
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
