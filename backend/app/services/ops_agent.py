"""Authenticated synchronous client for the privileged host operations agent."""

from __future__ import annotations

import hashlib
import hmac
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from host_agent.dgx_ops_agent.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    canonical_bytes,
    encode_frame,
    new_request,
    read_frame,
    sign_message,
)

_RESPONSE_FIELDS = frozenset(
    {
        "protocol_version",
        "request_id",
        "ok",
        "result",
        "error",
        "timestamp",
        "signature",
    }
)
_SIGNATURE_LENGTH = hashlib.sha256().digest_size * 2
_MAX_RESPONSE_AGE_SECONDS = 30
_MAX_ERROR_CODE_LENGTH = 128
_MAX_ERROR_MESSAGE_LENGTH = 1_024
_SAFE_REMOTE_ERROR_MESSAGE = "Host operations agent returned an error"
_REMOTE_ERROR_CODES = frozenset(
    {
        "approval_required",
        "authentication_failed",
        "expired_request",
        "frame_too_large",
        "invalid_frame",
        "invalid_parameters",
        "invalid_request",
        "job_not_found",
        "operation_failed",
        "operation_timeout",
        "policy_rejected",
        "read_timeout",
        "replay_detected",
        "unknown_action",
    }
)
_HEX_BYTES = frozenset(b"0123456789abcdefABCDEF")
_MAX_KEY_FILE_SIZE = 65


def _is_remote_error_code(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= _MAX_ERROR_CODE_LENGTH
        and value.isascii()
        and value in _REMOTE_ERROR_CODES
    )


class OpsAgentError(RuntimeError):
    """Base class for manager-side Agent failures."""


class OpsAgentUnavailable(OpsAgentError):
    """The Agent transport or its key is unavailable."""

    def __init__(self, _detail: str | None = None) -> None:
        super().__init__("Host operations agent is unavailable")


class OpsAgentProtocolError(OpsAgentError):
    """The Agent returned an unauthenticated or malformed response."""

    def __init__(self, _detail: str | None = None) -> None:
        super().__init__("Host operations agent returned an invalid response")


class OpsAgentRemoteError(OpsAgentError):
    """The Agent authenticated a stable application error."""

    def __init__(self, code: str) -> None:
        if not _is_remote_error_code(code):
            raise OpsAgentProtocolError()
        self.code = code
        self.message = _SAFE_REMOTE_ERROR_MESSAGE
        super().__init__(_SAFE_REMOTE_ERROR_MESSAGE)


@dataclass(frozen=True, slots=True)
class OpsAgentHealth:
    status: str
    protocol_version: int


class _DeadlineReader:
    def __init__(
        self,
        connection: socket.socket,
        deadline: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._connection = connection
        self._deadline = deadline
        self._monotonic = monotonic

    def recv(self, size: int) -> bytes:
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise TimeoutError("Agent response deadline exceeded")
        self._connection.settimeout(remaining)
        return self._connection.recv(size)


class OpsAgentClient:
    """Open one authenticated Unix stream connection for each Agent call."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        key_path: str | os.PathLike[str],
        *,
        connect_timeout_seconds: float = 3,
        read_timeout_seconds: float = 30,
        output_limit_bytes: int = 1_000_000,
        _connection_factory: Callable[[], socket.socket] | None = None,
        _monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.key_path = Path(key_path)
        self._connect_timeout = connect_timeout_seconds
        self._read_timeout = read_timeout_seconds
        self._output_limit = output_limit_bytes
        self._connection_factory = _connection_factory
        self._monotonic = _monotonic
        self._key_lock = threading.Lock()
        self._key_loaded = False
        self._key_unavailable = False
        self._secret: bytes | None = None

    def call(
        self,
        action: str,
        parameters: dict[str, Any],
        *,
        approval: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        timeout = self._read_timeout if timeout_seconds is None else timeout_seconds
        deadline = self._monotonic() + max(0, timeout)
        secret = self._get_secret()
        try:
            unsigned = new_request(action, parameters, approval=approval)
            request = sign_message(unsigned, secret)
            frame = encode_frame(request)
        except ProtocolError:
            raise OpsAgentProtocolError() from None

        connection = self._open_connection(deadline)
        try:
            self._write(connection, frame, deadline)
            response = self._read(connection, deadline)
        finally:
            connection.close()

        return self._validate_response(
            response,
            secret=secret,
            request_id=request["request_id"],
        )

    def _open_connection(self, deadline: float) -> socket.socket:
        if self._connection_factory is not None:
            try:
                return self._connection_factory()
            except OSError:
                raise OpsAgentUnavailable() from None

        unix_family = getattr(socket, "AF_UNIX", None)
        if unix_family is None:
            raise OpsAgentUnavailable()
        try:
            connection = socket.socket(unix_family, socket.SOCK_STREAM)
        except OSError:
            raise OpsAgentUnavailable() from None
        try:
            self._connect(connection, deadline)
        except Exception:
            connection.close()
            raise
        return connection

    def health(self) -> OpsAgentHealth:
        result = self.call("agent.health", {})
        if frozenset(result) != {"status", "protocol_version"}:
            raise OpsAgentProtocolError()
        status = result.get("status")
        version = result.get("protocol_version")
        if status != "ok" or type(version) is not int or version != PROTOCOL_VERSION:
            raise OpsAgentProtocolError()
        return OpsAgentHealth(status=status, protocol_version=version)

    def _get_secret(self) -> bytes:
        if not self._key_loaded:
            with self._key_lock:
                if not self._key_loaded:
                    try:
                        self._secret = self._read_key_file()
                    except (OSError, ValueError):
                        self._key_unavailable = True
                    self._key_loaded = True
        if self._key_unavailable or self._secret is None:
            raise OpsAgentUnavailable()
        return self._secret

    def _read_key_file(self) -> bytes:
        with self.key_path.open("rb") as key_file:
            value = key_file.read(_MAX_KEY_FILE_SIZE + 1)
        if len(value) == 32:
            return value
        if len(value) == 65 and value.endswith(b"\n"):
            value = value[:-1]
        if len(value) != 64 or any(byte not in _HEX_BYTES for byte in value):
            raise ValueError("invalid Agent key")
        return bytes.fromhex(value.decode("ascii"))

    def _connect(self, connection: socket.socket, deadline: float) -> None:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise OpsAgentUnavailable()
        connection.settimeout(min(self._connect_timeout, remaining))
        try:
            connection.connect(os.fspath(self.socket_path))
        except (OSError, TimeoutError):
            raise OpsAgentUnavailable() from None

    def _write(self, connection: socket.socket, frame: bytes, deadline: float) -> None:
        view = memoryview(frame)
        sent = 0
        try:
            while sent < len(view):
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise TimeoutError
                connection.settimeout(remaining)
                count = connection.send(view[sent:])
                if count <= 0:
                    raise OSError("Agent connection closed")
                sent += count
        except (OSError, TimeoutError):
            raise OpsAgentUnavailable() from None

    def _read(self, connection: socket.socket, deadline: float) -> dict[str, Any]:
        try:
            if deadline - self._monotonic() <= 0:
                raise TimeoutError
            return read_frame(
                _DeadlineReader(
                    connection,
                    deadline,
                    monotonic=self._monotonic,
                )
            )
        except ProtocolError as exc:
            if isinstance(exc.__cause__, (OSError, TimeoutError)):
                raise OpsAgentUnavailable() from None
            raise OpsAgentProtocolError() from None

    def _validate_response(
        self,
        response: dict[str, Any],
        *,
        secret: bytes,
        request_id: str,
    ) -> dict[str, Any]:
        if frozenset(response) != _RESPONSE_FIELDS:
            raise OpsAgentProtocolError()
        version = response.get("protocol_version")
        if type(version) is not int or version != PROTOCOL_VERSION:
            raise OpsAgentProtocolError()
        if response.get("request_id") != request_id:
            raise OpsAgentProtocolError()

        timestamp = response.get("timestamp")
        if (
            type(timestamp) is not int
            or abs(int(time.time()) - timestamp) > _MAX_RESPONSE_AGE_SECONDS
        ):
            raise OpsAgentProtocolError()

        signature = response.get("signature")
        if not isinstance(signature, str) or len(signature) != _SIGNATURE_LENGTH:
            raise OpsAgentProtocolError()
        try:
            expected = hmac.new(secret, canonical_bytes(response), hashlib.sha256).hexdigest()
        except ProtocolError:
            raise OpsAgentProtocolError() from None
        if not hmac.compare_digest(signature, expected):
            raise OpsAgentProtocolError()

        ok = response.get("ok")
        result = response.get("result")
        error = response.get("error")
        if type(ok) is not bool or (result is not None and not isinstance(result, dict)):
            raise OpsAgentProtocolError()
        self._validate_output_limit(result)
        if ok:
            if not isinstance(result, dict) or error is not None:
                raise OpsAgentProtocolError()
            return result

        if not isinstance(error, dict) or frozenset(error) != {"code", "message"}:
            raise OpsAgentProtocolError()
        code = error.get("code")
        message = error.get("message")
        if (
            not _is_remote_error_code(code)
            or not isinstance(message, str)
            or not message
            or len(message) > _MAX_ERROR_MESSAGE_LENGTH
        ):
            raise OpsAgentProtocolError()
        error["message"] = None
        message = None
        raise OpsAgentRemoteError(code) from None

    def _validate_output_limit(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        total = 0
        stack: list[Any] = [result]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    if key == "output":
                        if not isinstance(value, str):
                            raise OpsAgentProtocolError()
                        try:
                            encoded_length = len(value.encode("utf-8"))
                        except UnicodeEncodeError:
                            raise OpsAgentProtocolError() from None
                        total += encoded_length
                        if total > self._output_limit:
                            raise OpsAgentProtocolError()
                    else:
                        stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)


__all__ = [
    "OpsAgentClient",
    "OpsAgentError",
    "OpsAgentHealth",
    "OpsAgentProtocolError",
    "OpsAgentRemoteError",
    "OpsAgentUnavailable",
]
