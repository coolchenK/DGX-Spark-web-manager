"""Minimal POSIX exec launcher gated by a parent-owned pipe."""

from __future__ import annotations

import json
import os
import struct
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

EXIT_GATE_CLOSED = 75
EXIT_INVALID_REQUEST = 76
EXIT_EXEC_FAILED = 126
GO = b"G"
MAX_REQUEST_SIZE = 128 * 1024
MAX_ARGUMENTS = 256
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_ENVIRONMENT_VARIABLES = 18
MAX_ENVIRONMENT_VALUE_LENGTH = 256
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ExecFunction = Callable[[str, list[str], dict[str, str]], Any]


def write_request(
    descriptor: int,
    *,
    argv: Sequence[str],
    cwd: str | None,
    environment: Mapping[str, str],
) -> None:
    """Write one bounded request and close the pipe to mark it complete."""

    request = _validate_request(
        {"argv": list(argv), "cwd": cwd, "environment": dict(environment)}
    )
    payload = json.dumps(
        request,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    if not payload or len(payload) > MAX_REQUEST_SIZE:
        raise ValueError("invalid launcher request")
    try:
        _write_all(descriptor, struct.pack(">I", len(payload)) + payload)
    finally:
        os.close(descriptor)


def run_launcher(
    request_descriptor: int,
    gate_descriptor: int,
    *,
    exec_function: ExecFunction = os.execvpe,
) -> int:
    """Read a request, wait for GO, then replace this process with the command."""

    try:
        header = _read_exact(request_descriptor, 4)
        if len(header) != 4:
            return EXIT_INVALID_REQUEST
        length = struct.unpack(">I", header)[0]
        if not 0 < length <= MAX_REQUEST_SIZE:
            return EXIT_INVALID_REQUEST
        payload = _read_exact(request_descriptor, length)
        if len(payload) != length or os.read(request_descriptor, 1):
            return EXIT_INVALID_REQUEST
        try:
            request = json.loads(
                payload.decode("ascii"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda _value: _raise_invalid_request(),
            )
            request = _validate_request(request)
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            return EXIT_INVALID_REQUEST

        if os.read(gate_descriptor, 1) != GO:
            return EXIT_GATE_CLOSED
        os.close(request_descriptor)
        request_descriptor = -1
        os.close(gate_descriptor)
        gate_descriptor = -1
        if request["cwd"] is not None:
            os.chdir(request["cwd"])
        argv = request["argv"]
        exec_function(argv[0], argv, request["environment"])
        return 0
    except (OSError, ValueError):
        return EXIT_EXEC_FAILED
    finally:
        for descriptor in (request_descriptor, gate_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _validate_request(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"argv", "cwd", "environment"}:
        raise ValueError("invalid launcher request")
    argv = value["argv"]
    cwd = value["cwd"]
    environment = value["environment"]
    if (
        type(argv) is not list
        or not argv
        or len(argv) > MAX_ARGUMENTS
        or any(not isinstance(item, str) or not item or "\0" in item for item in argv)
        or sum(len(item.encode("utf-8")) for item in argv) > MAX_ARGUMENT_BYTES
    ):
        raise ValueError("invalid launcher request")
    if cwd is not None and (
        not isinstance(cwd, str) or not cwd or "\0" in cwd
    ):
        raise ValueError("invalid launcher request")
    if type(environment) is not dict or not 1 <= len(environment) <= MAX_ENVIRONMENT_VARIABLES:
        raise ValueError("invalid launcher request")
    for key, item in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "\0" in key
            or not isinstance(item, str)
            or len(item) > MAX_ENVIRONMENT_VALUE_LENGTH
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in item)
        ):
            raise ValueError("invalid launcher request")
    if environment.get("PATH") != SAFE_PATH:
        raise ValueError("invalid launcher request")
    return {"argv": argv, "cwd": cwd, "environment": environment}


def _read_exact(descriptor: int, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = os.read(descriptor, length - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("launcher request write failed")
        view = view[written:]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid launcher request")
        result[key] = value
    return result


def _raise_invalid_request() -> None:
    raise ValueError("invalid launcher request")


def main() -> int:
    if len(sys.argv) != 3:
        return EXIT_INVALID_REQUEST
    try:
        request_descriptor = int(sys.argv[1])
        gate_descriptor = int(sys.argv[2])
    except ValueError:
        return EXIT_INVALID_REQUEST
    if request_descriptor < 0 or gate_descriptor < 0:
        return EXIT_INVALID_REQUEST
    return run_launcher(request_descriptor, gate_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXIT_GATE_CLOSED", "run_launcher", "write_request"]
