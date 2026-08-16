import hashlib
import hmac
import io
import json
import math
import struct
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from host_agent.dgx_ops_agent import protocol as protocol_module
from host_agent.dgx_ops_agent.policy import (
    READ_ONLY_ACTIONS,
    PolicyError,
    validate_action,
)
from host_agent.dgx_ops_agent.protocol import (
    MAX_FRAME_SIZE,
    PROTOCOL_VERSION,
    NonceCache,
    ProtocolError,
    _read_exact,
    canonical_bytes,
    encode_frame,
    new_request,
    read_frame,
    sign_message,
    verify_message,
    write_frame,
)

SECRET = b"x" * 32
REPOSITORY_ROOT = Path(__file__).parents[2]


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


VALID_APPROVAL = {
    "plan_id": "plan-1",
    "step_id": "step-1",
    "approved_by": "admin",
    "approved_at": "2026-08-17T00:00:00Z",
}

READ_ACTION_CASES = {
    "host.memory": (
        {},
        ("/usr/bin/free", "--bytes"),
        10,
    ),
    "host.disk": (
        {},
        (
            "/usr/bin/df",
            "--block-size=1",
            "--output=source,target,size,used,avail,pcent",
        ),
        10,
    ),
    "host.gpu": (
        {},
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        15,
    ),
    "host.ports": ({}, ("/usr/bin/ss", "-lntupH"), 10),
    "host.processes": (
        {},
        (
            "/usr/bin/ps",
            "-eo",
            "pid,ppid,user,stat,%cpu,%mem,etimes,comm",
            "--sort=-%cpu",
        ),
        10,
    ),
    "docker.list": (
        {},
        ("/usr/bin/docker", "container", "list", "--all", "--no-trunc"),
        15,
    ),
    "docker.inspect": (
        {"container": "web-1"},
        ("/usr/bin/docker", "container", "inspect", "web-1"),
        15,
    ),
    "docker.logs": (
        {"container": "a" * 64, "tail": 250},
        ("/usr/bin/docker", "logs", "--tail", "250", "a" * 64),
        15,
    ),
    "docker.stats": (
        {"container": "web_1"},
        ("/usr/bin/docker", "stats", "--no-stream", "web_1"),
        15,
    ),
    "systemd.status": (
        {"service": "docker.service"},
        ("/usr/bin/systemctl", "--no-pager", "--full", "status", "docker.service"),
        15,
    ),
    "systemd.journal": (
        {"service": "docker.service", "tail": 500},
        (
            "/usr/bin/journalctl",
            "--no-pager",
            "--output=short-iso",
            "--unit",
            "docker.service",
            "--lines",
            "500",
        ),
        15,
    ),
}


@pytest.mark.parametrize("action", READ_ACTION_CASES)
def test_policy_read_only_actions_bind_fixed_argv_without_approval(action):
    parameters, expected_argv, expected_timeout = READ_ACTION_CASES[action]

    validated = validate_action(action, parameters, approval=None)

    assert READ_ONLY_ACTIONS == frozenset(READ_ACTION_CASES)
    assert validated.action == action
    assert validated.argv == expected_argv
    assert validated.cwd is None
    assert validated.timeout == expected_timeout
    assert validated.read_only is True
    assert validated.approval is None
    assert validated.environment == ()
    assert validated.argv[0].startswith("/")


@pytest.mark.parametrize("action", READ_ACTION_CASES)
def test_policy_read_only_actions_reject_unknown_parameter_keys(action):
    parameters = dict(READ_ACTION_CASES[action][0], unexpected="value")

    with pytest.raises(PolicyError, match="parameters"):
        validate_action(action, parameters, approval=None)


@pytest.mark.parametrize(
    ("action", "parameters"),
    [
        ("docker.inspect", {"container": "-x"}),
        ("docker.inspect", {"container": "../container"}),
        ("docker.inspect", {"container": "web;id"}),
        ("docker.inspect", {"container": "web name"}),
        ("docker.inspect", {"container": "a" * 129}),
        ("docker.logs", {"container": "web", "tail": 0}),
        ("docker.logs", {"container": "web", "tail": 5001}),
        ("docker.logs", {"container": "web", "tail": True}),
        ("docker.logs", {"container": "web", "tail": "10"}),
        ("systemd.status", {"service": "-user.service"}),
        ("systemd.status", {"service": "../user.service"}),
        ("systemd.status", {"service": "user/service"}),
        ("systemd.status", {"service": "user.service;id"}),
        ("systemd.status", {"service": "a" * 257}),
        ("systemd.journal", {"service": "docker.service", "tail": False}),
    ],
)
def test_policy_rejects_unsafe_selectors_and_invalid_tail(action, parameters):
    with pytest.raises(PolicyError, match="parameters"):
        validate_action(action, parameters, approval=None)


@pytest.mark.parametrize(
    ("action", "parameters"),
    [
        ("docker.inspect", {}),
        ("docker.inspect", {"container": 123}),
        ("docker.logs", {"container": "web"}),
        ("docker.stats", {"container": None}),
        ("systemd.status", {}),
        ("systemd.journal", {"service": "docker.service"}),
    ],
)
def test_policy_read_tools_enforce_exact_required_types(action, parameters):
    with pytest.raises(PolicyError, match="parameters"):
        validate_action(action, parameters, approval=None)


def test_policy_shell_requires_complete_approval_and_binds_bash_argv():
    parameters = {"command": "printf ok", "cwd": "/var/tmp", "timeout": 30}

    with pytest.raises(PolicyError, match="approval"):
        validate_action("shell.execute", parameters, approval=None)

    validated = validate_action("shell.execute", parameters, approval=VALID_APPROVAL)

    assert validated.action == "shell.execute"
    assert validated.argv == ("/bin/bash", "-lc", "printf ok")
    assert validated.cwd == "/var/tmp"
    assert validated.timeout == 30
    assert validated.read_only is False
    assert validated.approval is not None
    assert validated.approval.plan_id == "plan-1"
    assert validated.approval.step_id == "step-1"
    assert validated.approval.approved_by == "admin"
    assert validated.approval.approved_at == "2026-08-17T00:00:00Z"


def test_policy_shell_defaults_timeout_but_still_requires_approval():
    parameters = {"command": "id", "cwd": "/"}

    with pytest.raises(PolicyError, match="approval"):
        validate_action("shell.execute", parameters, approval=None)

    validated = validate_action("shell.execute", parameters, approval=VALID_APPROVAL)

    assert validated.timeout == 30
    assert validated.environment == ()


def test_policy_shell_accepts_only_bounded_non_sensitive_environment():
    environment = {
        "TZ": "UTC",
        "TERM": "xterm-256color",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "NO_COLOR": "1",
        "LC_CTYPE": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LANG": "en_US.UTF-8",
        "DEBIAN_FRONTEND": "noninteractive",
        "COLORTERM": "truecolor",
    }

    validated = validate_action(
        "shell.execute",
        {"command": "id", "cwd": "/", "env": environment},
        approval=VALID_APPROVAL,
    )

    assert validated.environment == tuple(sorted(environment.items()))
    assert dict(validated.environment) == environment


def test_policy_shell_accepts_empty_environment():
    validated = validate_action(
        "shell.execute",
        {"command": "id", "cwd": "/", "env": {}},
        approval=VALID_APPROVAL,
    )

    assert validated.environment == ()


@pytest.mark.parametrize(
    "environment",
    [
        True,
        [],
        {"UNKNOWN": "value"},
        {"PATH": "/tmp/bin"},
        {"LD_PRELOAD": "/tmp/library.so"},
        {"PYTHONPATH": "/tmp/python"},
        {"PYTHONHOME": "/tmp/python"},
        {"DGX_OPS_AGENT_SECRET": "secret"},
        {1: "value"},
        {"LANG": 1},
        {"LANG": True},
        {"LANG": "line\nvalue"},
        {"LANG": "value\x00suffix"},
        {"LANG": "x" * 257},
        {f"EXTRA_{index}": "value" for index in range(17)},
    ],
)
def test_policy_shell_rejects_unsafe_environment_without_echoing_it(environment):
    secret = "secret"

    with pytest.raises(PolicyError, match="^invalid parameters$") as exc_info:
        validate_action(
            "shell.execute",
            {"command": "id", "cwd": "/", "env": environment},
            approval=VALID_APPROVAL,
        )

    assert secret not in str(exc_info.value)


def test_policy_shell_environment_is_deterministic_and_immutable():
    first = validate_action(
        "shell.execute",
        {"command": "id", "cwd": "/", "env": {"TZ": "UTC", "LANG": "C"}},
        approval=VALID_APPROVAL,
    )
    second = validate_action(
        "shell.execute",
        {"command": "id", "cwd": "/", "env": {"LANG": "C", "TZ": "UTC"}},
        approval=VALID_APPROVAL,
    )

    assert first.environment == second.environment == (("LANG", "C"), ("TZ", "UTC"))
    with pytest.raises(TypeError):
        first.environment[0] = ("PATH", "/tmp/bin")


@pytest.mark.parametrize(
    "parameters",
    [
        {"command": "", "cwd": "/", "timeout": 30},
        {"command": 1, "cwd": "/", "timeout": 30},
        {"command": "x\x00id", "cwd": "/", "timeout": 30},
        {"command": "x" * 16_385, "cwd": "/", "timeout": 30},
        {"command": "id", "cwd": "relative", "timeout": 30},
        {"command": "id", "cwd": "/tmp/../var", "timeout": 30},
        {"command": "id", "cwd": "/tmp\x00/var", "timeout": 30},
        {"command": "id", "cwd": "/", "timeout": True},
        {"command": "id", "cwd": "/", "timeout": 0},
        {"command": "id", "cwd": "/", "timeout": 3601},
        {"command": "id", "cwd": "/", "timeout": 30, "unexpected": True},
    ],
)
def test_policy_shell_parameters_have_exact_bounded_schema(parameters):
    with pytest.raises(PolicyError, match="parameters") as exc_info:
        validate_action("shell.execute", parameters, approval=VALID_APPROVAL)

    assert "x" * 100 not in str(exc_info.value)


@pytest.mark.parametrize(
    "approval",
    [
        {},
        {**VALID_APPROVAL, "unexpected": "value"},
        {key: value for key, value in VALID_APPROVAL.items() if key != "step_id"},
        {**VALID_APPROVAL, "plan_id": ""},
        {**VALID_APPROVAL, "step_id": 1},
        {**VALID_APPROVAL, "approved_by": "a" * 129},
        {**VALID_APPROVAL, "approved_at": "2026-08-17T00:00:00+00:00"},
        {**VALID_APPROVAL, "approved_at": "2026-08-17T08:00:00+08:00"},
        {**VALID_APPROVAL, "approved_at": "not-a-date"},
    ],
)
def test_policy_shell_approval_has_exact_bounded_utc_schema(approval):
    with pytest.raises(PolicyError, match="approval"):
        validate_action(
            "shell.execute",
            {"command": "id", "cwd": "/", "timeout": 30},
            approval=approval,
        )


def test_policy_fails_closed_without_echoing_action_or_command_secrets():
    secret = "TOKEN=very-secret-value"

    with pytest.raises(PolicyError, match="^unknown action$") as action_error:
        validate_action(f"unknown.{secret}", {}, approval=None)
    with pytest.raises(PolicyError, match="parameters") as parameter_error:
        validate_action(
            "shell.execute",
            {"command": secret + "\x00", "cwd": "/", "timeout": 30},
            approval=VALID_APPROVAL,
        )

    assert secret not in str(action_error.value)
    assert secret not in str(parameter_error.value)


def test_policy_validated_action_and_approval_are_immutable():
    validated = validate_action(
        "shell.execute",
        {"command": "id", "cwd": "/", "timeout": 30},
        approval=VALID_APPROVAL,
    )

    with pytest.raises(FrozenInstanceError):
        validated.timeout = 60
    with pytest.raises(FrozenInstanceError):
        validated.approval.approved_by = "other"
    with pytest.raises(TypeError):
        validated.argv[0] = "/tmp/bash"


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


def test_sign_message_uses_hmac_sha256_over_canonical_bytes():
    request = new_request(
        "host.memory",
        {"unicode": "\u8bb0"},
        now=1000,
        nonce="nonce-1",
        request_id="request-1",
    )

    signed = sign_message(request, SECRET)
    expected = hmac.new(SECRET, canonical_bytes(request), hashlib.sha256).hexdigest()

    assert signed["signature"] == expected


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


@pytest.mark.parametrize("signature", ["\u8bb0", "\ud800", "0" * 63, "g" * 64])
def test_malformed_signature_is_rejected_without_native_compare_errors(signature):
    message = _signed_request()
    message["signature"] = signature

    with pytest.raises(ProtocolError, match="^invalid signature$"):
        verify_message(message, SECRET, now=1001, nonces=NonceCache())


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


def test_frame_reader_rejects_zero_length_before_reading_body():
    with pytest.raises(ProtocolError, match="frame length"):
        read_frame(io.BytesIO(struct.pack(">I", 0)))


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


class _SingleByteReader:
    def __init__(self, length):
        self.length = length
        self.calls = 0
        self.first_requested = None
        self.last_requested = None

    def read(self, size):
        self.calls += 1
        if self.first_requested is None:
            self.first_requested = size
        self.last_requested = size
        if self.calls <= self.length:
            return b"x"
        return b""


def test_read_exact_bounds_memory_for_one_byte_fragments():
    reader = _SingleByteReader(MAX_FRAME_SIZE)

    tracemalloc.start()
    try:
        result = _read_exact(reader, MAX_FRAME_SIZE)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result == b"x" * MAX_FRAME_SIZE
    assert reader.calls == MAX_FRAME_SIZE
    assert reader.first_requested == MAX_FRAME_SIZE
    assert reader.last_requested == 1
    assert peak < MAX_FRAME_SIZE * 8


class _OverReturningReader:
    def __init__(self):
        self.calls = 0

    def read(self, size):
        self.calls += 1
        if self.calls == 1:
            return b"x" * (size + 1)
        return b""


def test_read_exact_rejects_reader_returning_more_than_requested():
    reader = _OverReturningReader()

    with pytest.raises(ProtocolError, match="frame read failed"):
        _read_exact(reader, 4)

    assert reader.calls == 1


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


@pytest.mark.parametrize(
    "body",
    [
        b'{"value":' + b"1" * 5000 + b"}",
        b'{"x":' * 5000 + b"null" + b"}" * 5000,
    ],
    ids=["5000-digit-integer", "5000-level-object"],
)
def test_frame_reader_normalizes_excessive_json_complexity(body):
    frame = struct.pack(">I", len(body)) + body

    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as exc_info:
        read_frame(io.BytesIO(frame))
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "body",
    [
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1,"value":2}',
        b'{"outer":{"value":1,"value":2}}',
    ],
)
def test_frame_reader_rejects_nonstandard_constants_and_duplicate_keys(body):
    frame = struct.pack(">I", len(body)) + body

    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as exc_info:
        read_frame(io.BytesIO(frame))
    assert exc_info.value.__cause__ is None


def test_frame_reader_rejects_float_overflow():
    body = b'{"number":1e9999}'
    frame = struct.pack(">I", len(body)) + body

    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as exc_info:
        read_frame(io.BytesIO(frame))
    assert exc_info.value.__cause__ is None


def test_frame_reader_preserves_normal_exponents_and_negative_zero():
    body = b'{"exponent":-1.25e2,"negative_zero":-0.0}'
    frame = struct.pack(">I", len(body)) + body

    message = read_frame(io.BytesIO(frame))

    assert message["exponent"] == -125.0
    assert math.copysign(1.0, message["negative_zero"]) == -1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_outbound_json_rejects_nonstandard_constants(value):
    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as frame_error:
        encode_frame({"value": value})
    assert frame_error.value.__cause__ is None

    with pytest.raises(ProtocolError, match="^message is not valid JSON$") as canonical_error:
        canonical_bytes({"value": value})
    assert canonical_error.value.__cause__ is None


def test_outbound_json_normalizes_excessive_nesting():
    value = None
    for _ in range(5000):
        value = {"value": value}

    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as frame_error:
        encode_frame(value)
    assert frame_error.value.__cause__ is None

    with pytest.raises(ProtocolError, match="^message is not valid JSON$") as canonical_error:
        canonical_bytes(value)
    assert canonical_error.value.__cause__ is None


@pytest.mark.parametrize(
    "message",
    [
        {1: "integer", "1": "string"},
        {None: "none", "null": "string"},
        {"nested": {1: "integer"}},
        {"tuple": (1, 2)},
    ],
    ids=["integer-key-collision", "none-key-collision", "nested-key", "tuple-value"],
)
def test_outbound_json_rejects_values_that_cannot_round_trip(message):
    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as frame_error:
        encode_frame(message)
    assert frame_error.value.__cause__ is None

    with pytest.raises(ProtocolError, match="^message is not valid JSON$") as canonical_error:
        canonical_bytes(message)
    assert canonical_error.value.__cause__ is None


def test_frame_round_trip_is_closed_over_strict_json_values():
    message = {
        "string": "\u8bb0",
        "integer": 1,
        "float": -125.0,
        "boolean": True,
        "null": None,
        "array": [1, "two", False, None],
        "nested": {"key": "value"},
    }

    assert read_frame(io.BytesIO(encode_frame(message))) == message


def test_outbound_json_normalizes_circular_references():
    message = {}
    message["self"] = message

    with pytest.raises(ProtocolError, match="^invalid frame JSON$") as frame_error:
        encode_frame(message)
    assert frame_error.value.__cause__ is None

    with pytest.raises(ProtocolError, match="^message is not valid JSON$") as canonical_error:
        canonical_bytes(message)
    assert canonical_error.value.__cause__ is None


@pytest.mark.parametrize("operation", [canonical_bytes, encode_frame])
@pytest.mark.parametrize("error_type", [MemoryError, KeyboardInterrupt])
def test_json_serialization_does_not_swallow_process_exceptions(
    monkeypatch, operation, error_type
):
    def raise_error(*_args, **_kwargs):
        raise error_type

    monkeypatch.setattr(protocol_module.json, "dumps", raise_error)

    with pytest.raises(error_type):
        operation({})


@pytest.mark.parametrize("error_type", [MemoryError, KeyboardInterrupt])
def test_json_parsing_does_not_swallow_process_exceptions(monkeypatch, error_type):
    def raise_error(*_args, **_kwargs):
        raise error_type

    monkeypatch.setattr(protocol_module.json, "loads", raise_error)
    frame = encode_frame({})

    with pytest.raises(error_type):
        read_frame(io.BytesIO(frame))


def test_ci_lints_backend_and_host_agent():
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "ruff check backend host_agent" in workflow
