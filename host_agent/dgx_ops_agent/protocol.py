"""Signed JSON protocol and bounded stream framing for the host operations agent."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import struct
import threading
import time
import uuid
from typing import Any, BinaryIO

PROTOCOL_VERSION = 1
MAX_FRAME_SIZE = 1024 * 1024

_MIN_SECRET_SIZE = 32
_MAX_ACTION_LENGTH = 128
_MAX_REQUEST_ID_LENGTH = 128
_MAX_NONCE_LENGTH = 256
_MAX_TIMESTAMP = (1 << 63) - 1
_MAX_AGE = 24 * 60 * 60
_DEFAULT_NONCE_CAPACITY = 4096
_REQUIRED_REQUEST_FIELDS = frozenset(
    {
        "protocol_version",
        "request_id",
        "action",
        "parameters",
        "timestamp",
        "nonce",
        "signature",
    }
)
_OPTIONAL_REQUEST_FIELDS = frozenset({"approval"})


class ProtocolError(ValueError):
    """A message violates the authenticated wire protocol."""


def _validate_secret(secret: bytes) -> None:
    if not isinstance(secret, bytes) or len(secret) < _MIN_SECRET_SIZE:
        raise ProtocolError("secret must be at least 32 bytes")


def _validate_bounded_string(value: Any, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ProtocolError(f"invalid {name}")


def _validate_timestamp(value: Any, name: str = "timestamp") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_TIMESTAMP:
        raise ProtocolError(f"invalid {name}")


def _validate_max_age(max_age: Any) -> None:
    if (
        isinstance(max_age, bool)
        or not isinstance(max_age, int)
        or not 0 <= max_age <= _MAX_AGE
    ):
        raise ProtocolError("invalid max_age")


def canonical_bytes(message: dict[str, Any]) -> bytes:
    """Serialize a JSON object deterministically, excluding its top-level signature."""
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    unsigned = {key: value for key, value in message.items() if key != "signature"}
    try:
        encoded = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("message is not valid JSON") from exc
    return encoded.encode("ascii")


def new_request(
    action: str,
    parameters: dict[str, Any],
    *,
    approval: dict[str, Any] | None = None,
    now: int | None = None,
    nonce: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build an unsigned request with generated identifiers when they are omitted."""
    timestamp = int(time.time()) if now is None else now
    request_nonce = secrets.token_hex(16) if nonce is None else nonce
    identifier = str(uuid.uuid4()) if request_id is None else request_id

    _validate_bounded_string(action, "action", _MAX_ACTION_LENGTH)
    if not isinstance(parameters, dict):
        raise ProtocolError("invalid parameters")
    _validate_timestamp(timestamp)
    _validate_bounded_string(request_nonce, "nonce", _MAX_NONCE_LENGTH)
    _validate_bounded_string(identifier, "request_id", _MAX_REQUEST_ID_LENGTH)
    if approval is not None and not isinstance(approval, dict):
        raise ProtocolError("invalid approval")

    request: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": identifier,
        "action": action,
        "parameters": dict(parameters),
        "timestamp": timestamp,
        "nonce": request_nonce,
    }
    if approval is not None:
        request["approval"] = dict(approval)
    return request


def sign_message(message: dict[str, Any], secret: bytes) -> dict[str, Any]:
    """Return a signed shallow copy of a JSON object."""
    _validate_secret(secret)
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    result = dict(message)
    result["signature"] = hmac.new(secret, canonical_bytes(result), hashlib.sha256).hexdigest()
    return result


def _validate_request_schema(message: dict[str, Any]) -> None:
    version = message.get("protocol_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")

    fields = frozenset(message)
    if not _REQUIRED_REQUEST_FIELDS <= fields or fields - (
        _REQUIRED_REQUEST_FIELDS | _OPTIONAL_REQUEST_FIELDS
    ):
        raise ProtocolError("invalid request schema")

    try:
        _validate_bounded_string(message["request_id"], "request_id", _MAX_REQUEST_ID_LENGTH)
        _validate_bounded_string(message["action"], "action", _MAX_ACTION_LENGTH)
        if not isinstance(message["parameters"], dict):
            raise ProtocolError("invalid parameters")
        _validate_timestamp(message["timestamp"])
        _validate_bounded_string(message["nonce"], "nonce", _MAX_NONCE_LENGTH)
        if "approval" in message and not isinstance(message["approval"], dict):
            raise ProtocolError("invalid approval")
    except ProtocolError as exc:
        raise ProtocolError("invalid request schema") from exc


class NonceCache:
    """Thread-safe, bounded cache of nonces that remain inside the replay window."""

    def __init__(self, max_entries: int = _DEFAULT_NONCE_CAPACITY) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ProtocolError("invalid nonce cache capacity")
        self._max_entries = max_entries
        self._entries: dict[str, int] = {}
        self._lock = threading.Lock()

    def consume(self, nonce: str, timestamp: int, *, now: int, max_age: int = 30) -> None:
        """Atomically record a nonce or reject it if it is live or capacity is exhausted."""
        _validate_bounded_string(nonce, "nonce", _MAX_NONCE_LENGTH)
        _validate_timestamp(timestamp)
        _validate_timestamp(now, "current time")
        _validate_max_age(max_age)

        with self._lock:
            expired = [key for key, expires_at in self._entries.items() if expires_at < now]
            for key in expired:
                del self._entries[key]

            if nonce in self._entries:
                raise ProtocolError("replayed nonce")
            if len(self._entries) >= self._max_entries:
                raise ProtocolError("nonce cache capacity exceeded")
            self._entries[nonce] = timestamp + max_age


def verify_message(
    message: dict[str, Any],
    secret: bytes,
    *,
    now: int,
    nonces: NonceCache,
    max_age: int = 30,
) -> dict[str, Any]:
    """Authenticate and validate a request before atomically consuming its nonce."""
    _validate_secret(secret)
    if not isinstance(message, dict):
        raise ProtocolError("invalid signature")

    signature = message.get("signature")
    try:
        expected = hmac.new(secret, canonical_bytes(message), hashlib.sha256).hexdigest()
    except ProtocolError as exc:
        raise ProtocolError("invalid signature") from exc
    if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
        raise ProtocolError("invalid signature")

    _validate_request_schema(message)
    _validate_timestamp(now, "current time")
    _validate_max_age(max_age)

    timestamp = message["timestamp"]
    if timestamp < now - max_age:
        raise ProtocolError("expired request")
    if timestamp > now + max_age:
        raise ProtocolError("request timestamp is too far in the future")

    if not isinstance(nonces, NonceCache):
        raise ProtocolError("invalid nonce cache")
    nonces.consume(message["nonce"], timestamp, now=now, max_age=max_age)
    return {key: value for key, value in message.items() if key != "signature"}


def _json_payload(message: dict[str, Any]) -> bytes:
    if not isinstance(message, dict):
        raise ProtocolError("frame must contain a JSON object")
    try:
        payload = json.dumps(
            message,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid frame JSON") from exc
    if len(payload) > MAX_FRAME_SIZE:
        raise ProtocolError("frame too large")
    return payload


def encode_frame(message: dict[str, Any]) -> bytes:
    """Encode one bounded JSON object with a four-byte big-endian length prefix."""
    payload = _json_payload(message)
    return struct.pack(">I", len(payload)) + payload


def _read_chunk(reader: BinaryIO, size: int) -> bytes:
    operation = getattr(reader, "recv", None) or getattr(reader, "read", None)
    if operation is None:
        raise ProtocolError("invalid frame reader")
    try:
        chunk = operation(size)
    except (OSError, ValueError) as exc:
        raise ProtocolError("frame read failed") from exc
    if not isinstance(chunk, bytes):
        raise ProtocolError("frame read failed")
    return chunk


def _read_exact(reader: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = _read_chunk(reader, remaining)
        if not chunk:
            raise ProtocolError("unexpected EOF while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(reader: BinaryIO) -> dict[str, Any]:
    """Read exactly one bounded frame from a socket or binary stream."""
    header = _read_exact(reader, 4)
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_SIZE:
        raise ProtocolError("frame too large")

    payload = _read_exact(reader, length)
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid frame JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("frame must contain a JSON object")
    return message


def write_frame(writer: BinaryIO, message: dict[str, Any]) -> None:
    """Write one complete frame to a socket or binary stream."""
    frame = encode_frame(message)
    sendall = getattr(writer, "sendall", None)
    if sendall is not None:
        try:
            sendall(frame)
        except OSError as exc:
            raise ProtocolError("frame write failed") from exc
        return

    write = getattr(writer, "write", None)
    if write is None:
        raise ProtocolError("invalid frame writer")
    view = memoryview(frame)
    try:
        while view:
            written = write(view)
            if not isinstance(written, int) or written <= 0:
                raise ProtocolError("frame write failed")
            view = view[written:]
        flush = getattr(writer, "flush", None)
        if flush is not None:
            flush()
    except (OSError, ValueError) as exc:
        raise ProtocolError("frame write failed") from exc
