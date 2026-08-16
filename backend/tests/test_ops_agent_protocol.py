import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from host_agent.dgx_ops_agent.protocol import (
    MAX_FRAME_SIZE,
    PROTOCOL_VERSION,
    NonceCache,
    ProtocolError,
    canonical_bytes,
    encode_frame,
    new_request,
    read_frame,
    sign_message,
    verify_message,
    write_frame,
)

SECRET = b"x" * 32


def _signed_request(**changes):
    request = new_request(
        "host.memory",
        {},
        now=1000,
        nonce="nonce-1",
        request_id="request-1",
    )
    request.update(changes)
    return sign_message(request, SECRET)


def test_canonical_json_is_stable_ascii_and_excludes_signature():
    message = {"z": 1, "signature": "must-not-be-signed", "a": "\u8bb0"}

    assert canonical_bytes(message) == b'{"a":"\\u8bb0","z":1}'


def test_signed_request_round_trip_without_mutating_input():
    request = new_request(
        "host.memory",
        {},
        now=1000,
        nonce="nonce-1",
        request_id="request-1",
    )

    signed = sign_message(request, SECRET)
    verified = verify_message(signed, SECRET, now=1005, nonces=NonceCache())

    assert "signature" not in request
    assert signed["signature"] != SECRET.hex()
    assert verified == request
    assert verified["action"] == "host.memory"


def test_new_request_has_fixed_schema_and_optional_approval():
    request = new_request(
        "shell.execute",
        {"command": "true"},
        approval={"approval_id": "approved-1"},
        now=1000,
        nonce="nonce-1",
        request_id="request-1",
    )

    assert request == {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "request-1",
        "action": "shell.execute",
        "parameters": {"command": "true"},
        "timestamp": 1000,
        "nonce": "nonce-1",
        "approval": {"approval_id": "approved-1"},
    }
    assert "approval" not in new_request(
        "host.memory",
        {},
        now=1000,
        nonce="nonce-2",
        request_id="request-2",
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"action": ""}, "action"),
        ({"action": 1}, "action"),
        ({"parameters": []}, "parameters"),
        ({"now": True}, "timestamp"),
        ({"now": -1}, "timestamp"),
        ({"nonce": ""}, "nonce"),
        ({"nonce": 1}, "nonce"),
        ({"request_id": ""}, "request_id"),
        ({"approval": []}, "approval"),
    ],
)
def test_new_request_rejects_invalid_types_and_empty_identifiers(kwargs, error):
    values = {
        "action": "host.memory",
        "parameters": {},
        "now": 1000,
        "nonce": "nonce-1",
        "request_id": "request-1",
    }
    values.update(kwargs)

    with pytest.raises(ProtocolError, match=error):
        new_request(**values)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"protocol_version": True}, "version"),
        ({"protocol_version": 2}, "version"),
        ({"request_id": 1}, "schema"),
        ({"request_id": ""}, "schema"),
        ({"action": 1}, "schema"),
        ({"action": ""}, "schema"),
        ({"parameters": []}, "schema"),
        ({"timestamp": True}, "schema"),
        ({"timestamp": "1000"}, "schema"),
        ({"nonce": 1}, "schema"),
        ({"nonce": ""}, "schema"),
        ({"approval": None}, "schema"),
        ({"unexpected": "field"}, "schema"),
    ],
)
def test_verify_rejects_signed_wrong_version_or_schema(change, error):
    with pytest.raises(ProtocolError, match=error):
        verify_message(
            _signed_request(**change),
            SECRET,
            now=1001,
            nonces=NonceCache(),
        )


def test_verify_checks_signature_before_disclosing_schema_errors():
    malformed = new_request(
        "host.memory",
        {},
        now=1000,
        nonce="nonce-1",
        request_id="request-1",
    )
    malformed["unexpected"] = "field"
    malformed["signature"] = "0" * 64

    with pytest.raises(ProtocolError, match="signature") as exc_info:
        verify_message(malformed, SECRET, now=1001, nonces=NonceCache())

    assert SECRET.hex() not in str(exc_info.value)
    assert malformed["signature"] not in str(exc_info.value)


def test_rejects_tampering_expiration_future_timestamp_and_replay():
    cache = NonceCache()
    signed = _signed_request()
    verify_message(signed, SECRET, now=1001, nonces=cache)

    with pytest.raises(ProtocolError, match="replayed"):
        verify_message(signed, SECRET, now=1002, nonces=cache)

    with pytest.raises(ProtocolError, match="expired"):
        verify_message(_signed_request(timestamp=1), SECRET, now=1000, nonces=NonceCache())

    with pytest.raises(ProtocolError, match="future"):
        verify_message(
            _signed_request(timestamp=1032),
            SECRET,
            now=1000,
            nonces=NonceCache(),
        )

    tampered = _signed_request()
    tampered["parameters"] = {"changed": True}
    with pytest.raises(ProtocolError, match="signature"):
        verify_message(tampered, SECRET, now=1001, nonces=NonceCache())


def test_invalid_signature_does_not_consume_nonce():
    valid = _signed_request()
    invalid = dict(valid, signature="0" * 64)
    cache = NonceCache()

    with pytest.raises(ProtocolError, match="signature"):
        verify_message(invalid, SECRET, now=1001, nonces=cache)

    assert verify_message(valid, SECRET, now=1001, nonces=cache)["nonce"] == "nonce-1"


def test_invalid_timestamp_cannot_bypass_validation_or_consume_nonce():
    cache = NonceCache()
    invalid = _signed_request(timestamp=True)

    with pytest.raises(ProtocolError, match="schema"):
        verify_message(invalid, SECRET, now=1001, nonces=cache)

    assert verify_message(_signed_request(), SECRET, now=1001, nonces=cache)


def test_timestamp_window_rejects_bool_configuration_values():
    signed = _signed_request()

    with pytest.raises(ProtocolError, match="current time"):
        verify_message(signed, SECRET, now=True, nonces=NonceCache())
    with pytest.raises(ProtocolError, match="max_age"):
        verify_message(signed, SECRET, now=1001, nonces=NonceCache(), max_age=True)


def test_nonce_cache_allows_only_one_concurrent_consumer():
    cache = NonceCache()

    def consume_once():
        try:
            verify_message(_signed_request(), SECRET, now=1001, nonces=cache)
        except ProtocolError as exc:
            return str(exc)
        return "accepted"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: consume_once(), range(16)))

    assert results.count("accepted") == 1
    assert results.count("replayed nonce") == 15


def test_nonce_cache_is_bounded_and_cleans_expired_entries():
    cache = NonceCache(max_entries=1)
    cache.consume("nonce-1", timestamp=100, now=100, max_age=10)

    with pytest.raises(ProtocolError, match="capacity"):
        cache.consume("nonce-2", timestamp=100, now=100, max_age=10)

    cache.consume("nonce-2", timestamp=111, now=111, max_age=10)
    with pytest.raises(ProtocolError, match="replayed"):
        cache.consume("nonce-2", timestamp=111, now=111, max_age=10)


def test_frame_round_trip_uses_four_byte_big_endian_length():
    message = {"action": "host.memory", "parameters": {}}

    frame = encode_frame(message)

    assert frame[:4] == struct.pack(">I", len(frame) - 4)
    assert read_frame(io.BytesIO(frame)) == message


def test_write_frame_encodes_an_object():
    output = io.BytesIO()

    write_frame(output, {"status": "ok"})

    assert read_frame(io.BytesIO(output.getvalue())) == {"status": "ok"}


@pytest.mark.parametrize("message", [[], "text", 1, None])
def test_frame_encoding_rejects_non_objects(message):
    with pytest.raises(ProtocolError, match="object"):
        encode_frame(message)


def test_frame_encoding_rejects_oversized_payload():
    with pytest.raises(ProtocolError, match="too large"):
        encode_frame({"payload": "x" * MAX_FRAME_SIZE})


class _GuardedReader:
    def __init__(self, header):
        self.header = header
        self.calls = 0

    def read(self, _size):
        self.calls += 1
        if self.calls == 1:
            return self.header
        raise AssertionError("oversized frame body was read")


def test_frame_reader_rejects_oversized_length_before_reading_body():
    reader = _GuardedReader(struct.pack(">I", MAX_FRAME_SIZE + 1))

    with pytest.raises(ProtocolError, match="too large"):
        read_frame(reader)

    assert reader.calls == 1


class _ChunkedReader:
    def __init__(self, data, chunk_size=1):
        self.data = data
        self.chunk_size = chunk_size

    def read(self, size):
        chunk = self.data[: min(size, self.chunk_size)]
        self.data = self.data[len(chunk) :]
        return chunk


def test_frame_reader_handles_short_reads():
    frame = encode_frame({"status": "ok"})

    assert read_frame(_ChunkedReader(frame)) == {"status": "ok"}


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00\x00",
        struct.pack(">I", 5) + b"{}",
    ],
)
def test_frame_reader_rejects_eof(data):
    with pytest.raises(ProtocolError, match="EOF"):
        read_frame(io.BytesIO(data))


@pytest.mark.parametrize(
    "body",
    [
        json.dumps([1, 2]).encode(),
        b'{"ok":true} trailing',
        b"\xff",
    ],
)
def test_frame_reader_rejects_non_object_or_invalid_json(body):
    frame = struct.pack(">I", len(body)) + body

    with pytest.raises(ProtocolError, match="frame"):
        read_frame(io.BytesIO(frame))
