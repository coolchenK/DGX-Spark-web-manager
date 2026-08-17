import hashlib
import hmac
import io
import json
import math
import os
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from host_agent.dgx_ops_agent import protocol as protocol_module
from host_agent.dgx_ops_agent import runner as runner_module
from host_agent.dgx_ops_agent import server as server_module
from host_agent.dgx_ops_agent.policy import (
    READ_ONLY_ACTIONS,
    PolicyError,
    ValidatedAction,
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
from host_agent.dgx_ops_agent.redaction import StreamingRedactor
from host_agent.dgx_ops_agent.runner import SAFE_PATH, JobRunner
from host_agent.dgx_ops_agent.server import AgentServer, UnixSocketAgentServer

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


def _wait_for_terminal(runner, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job.status not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


class _FakeRunner:
    def __init__(self):
        self.started = []
        self.jobs = {}

    def start(self, action):
        self.started.append(action)
        job = {
            "job_id": "00000000-0000-4000-8000-000000000001",
            "status": "queued",
            "output": "",
            "output_offset": 0,
            "truncated_before": 0,
            "exit_code": None,
            "started": None,
            "finished": None,
            "error": None,
        }
        self.jobs[job["job_id"]] = job
        return type("FakeJob", (), {"as_dict": lambda self: dict(job)})()

    def get(self, job_id, *, offset=None):
        job = dict(self.jobs[job_id])
        if offset is not None:
            if offset > job["output_offset"]:
                raise ValueError("invalid offset")
            job["output"] = job["output"][offset:]
            job["truncated_before"] = offset
        return type("FakeJob", (), {"as_dict": lambda self: dict(job)})()

    def cancel(self, job_id):
        job = self.jobs[job_id]
        job["status"] = "cancelled"
        return type("FakeJob", (), {"as_dict": lambda self: dict(job)})()


def _server_request(action, parameters, *, nonce, approval=None, now=1000):
    return sign_message(
        new_request(
            action,
            parameters,
            approval=approval,
            now=now,
            nonce=nonce,
            request_id=f"request-{nonce}",
        ),
        SECRET,
    )


def _assert_signed_response(response, *, request_id, ok=True):
    assert set(response) == {
        "protocol_version",
        "request_id",
        "ok",
        "result",
        "error",
        "timestamp",
        "signature",
    }
    assert response["protocol_version"] == PROTOCOL_VERSION
    assert response["request_id"] == request_id
    assert response["ok"] is ok
    expected = hmac.new(SECRET, canonical_bytes(response), hashlib.sha256).hexdigest()
    assert hmac.compare_digest(response["signature"], expected)


def _call_over_socket(server, request, *, timeout=2):
    client, agent = socket.socketpair()
    worker = threading.Thread(
        target=server.serve_connection,
        args=(agent,),
        kwargs={"read_timeout": timeout},
        daemon=True,
    )
    worker.start()
    try:
        write_frame(client, request)
        response = read_frame(client)
    finally:
        client.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    return response


def test_server_dispatches_health_read_and_shell_as_signed_jobs():
    runner = _FakeRunner()
    server = AgentServer(SECRET, runner, clock=lambda: 1000)

    health = server.handle(
        _server_request("agent.health", {}, nonce="health")
    )
    _assert_signed_response(health, request_id="request-health")
    assert health["result"] == {"status": "ok", "protocol_version": 1}
    assert runner.started == []

    read = server.handle(
        _server_request("host.memory", {}, nonce="read")
    )
    _assert_signed_response(read, request_id="request-read")
    assert read["result"]["status"] == "queued"
    assert runner.started[-1].action == "host.memory"

    shell = server.handle(
        _server_request(
            "shell.execute",
            {"command": "printf ok", "cwd": "/", "timeout": 10},
            approval={
                **VALID_APPROVAL,
                "approved_at": "1970-01-01T00:16:40Z",
            },
            nonce="shell",
        )
    )
    _assert_signed_response(shell, request_id="request-shell")
    assert shell["result"]["job_id"]
    assert runner.started[-1].argv == (
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        "printf ok",
    )


def test_server_dispatches_every_policy_read_action():
    runner = _FakeRunner()
    server = AgentServer(SECRET, runner, clock=lambda: 1000)

    for index, action in enumerate(sorted(READ_ONLY_ACTIONS)):
        response = server.handle(
            _server_request(
                action,
                READ_ACTION_CASES[action][0],
                nonce=f"read-{index}",
            )
        )
        assert response["ok"] is True
        assert runner.started[-1].action == action

    assert len(runner.started) == len(READ_ONLY_ACTIONS) == 11


def test_server_rejects_invalid_secret_and_listener_at_startup():
    with pytest.raises(ProtocolError, match="secret"):
        AgentServer(b"short", _FakeRunner())

    with pytest.raises(ValueError, match="listener"):
        UnixSocketAgentServer(
            AgentServer(SECRET, _FakeRunner()),
            listener=object(),
        )

    closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed.close()
    with pytest.raises(ValueError, match="listener"):
        UnixSocketAgentServer(
            AgentServer(SECRET, _FakeRunner()),
            listener=closed,
        )


def test_server_forwards_request_timestamp_to_shell_policy(monkeypatch):
    runner = _FakeRunner()
    captured = {}
    original = validate_action

    def capture(action, parameters, approval, *, request_timestamp=None):
        captured["request_timestamp"] = request_timestamp
        return original(
            action,
            parameters,
            approval,
            request_timestamp=request_timestamp,
        )

    monkeypatch.setattr(server_module, "validate_action", capture)
    server = AgentServer(SECRET, runner, clock=lambda: 1000)
    response = server.handle(
        _server_request(
            "shell.execute",
            {"command": "true", "cwd": "/"},
            approval={**VALID_APPROVAL, "approved_at": "1970-01-01T00:16:40Z"},
            nonce="timestamp",
            now=1000,
        )
    )

    assert response["ok"] is True
    assert captured == {"request_timestamp": 1000}


def test_server_gets_output_and_cancels_jobs_idempotently():
    runner = _FakeRunner()
    server = AgentServer(SECRET, runner, clock=lambda: 1000)
    created = server.handle(_server_request("host.memory", {}, nonce="create"))
    job_id = created["result"]["job_id"]
    runner.jobs[job_id].update(
        status="running", output="abcdef", output_offset=6
    )

    fetched = server.handle(
        _server_request(
            "job.get", {"job_id": job_id, "offset": 2}, nonce="get"
        )
    )
    _assert_signed_response(fetched, request_id="request-get")
    assert fetched["result"]["output"] == "cdef"
    assert fetched["result"]["truncated_before"] == 2

    first = server.handle(
        _server_request("job.cancel", {"job_id": job_id}, nonce="cancel-1")
    )
    second = server.handle(
        _server_request("job.cancel", {"job_id": job_id}, nonce="cancel-2")
    )
    assert first["result"]["status"] == "cancelled"
    assert second["result"] == first["result"]

    beyond = server.handle(
        _server_request(
            "job.get", {"job_id": job_id, "offset": 7}, nonce="beyond"
        )
    )
    assert beyond["error"] == {
        "code": "invalid_parameters",
        "message": "invalid parameters",
    }


@pytest.mark.parametrize(
    ("action", "parameters", "code"),
    [
        ("job.get", {}, "invalid_parameters"),
        ("job.get", {"job_id": "bad", "extra": True}, "invalid_parameters"),
        ("job.get", {"job_id": "bad"}, "invalid_parameters"),
        ("job.get", {"job_id": "bad", "offset": True}, "invalid_parameters"),
        ("job.get", {"job_id": "bad", "offset": -1}, "invalid_parameters"),
        ("job.cancel", {"job_id": 1}, "invalid_parameters"),
        ("job.cancel", {"job_id": "bad"}, "invalid_parameters"),
        ("job.cancel", {"job_id": "bad", "offset": 0}, "invalid_parameters"),
        ("unknown.action", {}, "unknown_action"),
    ],
)
def test_server_returns_stable_errors_for_invalid_actions(action, parameters, code):
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    response = server.handle(
        _server_request(action, parameters, nonce=f"invalid-{code}-{action}")
    )

    _assert_signed_response(
        response,
        request_id=f"request-invalid-{code}-{action}",
        ok=False,
    )
    assert response["result"] is None
    assert response["error"]["code"] == code
    assert set(response["error"]) == {"code", "message"}
    assert "Traceback" not in json.dumps(response)


def test_server_maps_missing_jobs_and_runner_failures_without_internal_details():
    class BrokenRunner(_FakeRunner):
        def start(self, action):
            raise RuntimeError("secret at C:\\private\\agent.py")

    missing_server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    missing = missing_server.handle(
        _server_request(
            "job.get",
            {"job_id": "00000000-0000-4000-8000-000000000099"},
            nonce="missing",
        )
    )
    assert missing["error"] == {"code": "job_not_found", "message": "job not found"}

    broken_server = AgentServer(SECRET, BrokenRunner(), clock=lambda: 1000)
    broken = broken_server.handle(
        _server_request("host.memory", {}, nonce="broken")
    )
    assert broken["error"] == {
        "code": "operation_failed",
        "message": "operation failed",
    }
    assert "private" not in json.dumps(broken)
    assert "secret" not in json.dumps(broken)


def test_server_authentication_errors_are_signed_and_do_not_consume_valid_nonce():
    runner = _FakeRunner()
    server = AgentServer(SECRET, runner, clock=lambda: 1000)
    valid = _server_request("host.memory", {}, nonce="auth")
    tampered = dict(valid)
    tampered["parameters"] = {"changed": True}

    rejected = server.handle(tampered)
    _assert_signed_response(rejected, request_id=None, ok=False)
    assert rejected["error"] == {
        "code": "authentication_failed",
        "message": "request authentication failed",
    }
    assert runner.started == []

    accepted = server.handle(valid)
    assert accepted["ok"] is True
    assert len(runner.started) == 1
    replayed = server.handle(valid)
    assert replayed["error"]["code"] == "replay_detected"
    assert replayed["request_id"] is None
    assert len(runner.started) == 1


def test_server_rejects_invalid_health_and_shell_policy_parameters():
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    health = server.handle(
        _server_request("agent.health", {"extra": True}, nonce="bad-health")
    )
    assert health["error"]["code"] == "invalid_parameters"

    shell = server.handle(
        _server_request(
            "shell.execute",
            {"command": "true", "cwd": "/"},
            nonce="bad-shell",
        )
    )
    assert shell["error"] == {
        "code": "policy_rejected",
        "message": "request rejected by policy",
    }


def test_server_maps_expired_and_malformed_signed_requests_to_stable_errors():
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    expired = server.handle(
        _server_request("agent.health", {}, nonce="expired", now=900)
    )
    assert expired["error"]["code"] == "expired_request"
    assert expired["request_id"] is None

    malformed = sign_message(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "malformed",
            "action": "agent.health",
            "parameters": {},
            "timestamp": 1000,
            "nonce": "malformed",
            "unexpected": True,
        },
        SECRET,
    )
    rejected = server.handle(malformed)
    assert rejected["error"] == {
        "code": "invalid_request",
        "message": "invalid request",
    }
    assert rejected["request_id"] is None


def test_server_processes_exactly_one_framed_request_per_connection():
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    response = _call_over_socket(
        server,
        _server_request("agent.health", {}, nonce="socket"),
    )

    _assert_signed_response(response, request_id="request-socket")
    assert response["result"]["status"] == "ok"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (struct.pack(">I", MAX_FRAME_SIZE + 1), "frame_too_large"),
        (struct.pack(">I", 2) + b"{]", "invalid_frame"),
    ],
)
def test_server_signs_transport_frame_errors(payload, code):
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    client, agent = socket.socketpair()
    worker = threading.Thread(
        target=server.serve_connection,
        args=(agent,),
        kwargs={"read_timeout": 1},
        daemon=True,
    )
    worker.start()
    client.sendall(payload)
    response = read_frame(client)
    client.close()
    worker.join(timeout=2)

    _assert_signed_response(response, request_id=None, ok=False)
    assert response["error"]["code"] == code


def test_server_times_out_partial_slow_frames_with_signed_error():
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    client, agent = socket.socketpair()
    worker = threading.Thread(
        target=server.serve_connection,
        args=(agent,),
        kwargs={"read_timeout": 0.05},
        daemon=True,
    )
    worker.start()
    client.sendall(struct.pack(">I", 100) + b"{")
    response = read_frame(client)
    client.close()
    worker.join(timeout=2)

    _assert_signed_response(response, request_id=None, ok=False)
    assert response["error"] == {
        "code": "read_timeout",
        "message": "request read timed out",
    }


def test_server_read_timeout_is_an_absolute_deadline_for_trickle_clients():
    server = AgentServer(SECRET, _FakeRunner(), clock=lambda: 1000)
    client, agent = socket.socketpair()
    worker = threading.Thread(
        target=server.serve_connection,
        args=(agent,),
        kwargs={"read_timeout": 0.1},
        daemon=True,
    )
    worker.start()
    client.sendall(struct.pack(">I", 100) + b"{")

    def trickle():
        for _ in range(6):
            time.sleep(0.04)
            try:
                client.sendall(b"x")
            except OSError:
                return

    sender = threading.Thread(target=trickle, daemon=True)
    started = time.monotonic()
    sender.start()
    response = read_frame(client)
    elapsed = time.monotonic() - started
    client.close()
    sender.join(timeout=1)
    worker.join(timeout=1)

    assert response["error"]["code"] == "read_timeout"
    assert elapsed < 0.2
    assert not worker.is_alive()


def test_unix_socket_server_uses_injected_listener_and_bounds_concurrency():
    class BlockingAgent:
        def __init__(self):
            self.lock = threading.Lock()
            self.release = threading.Event()
            self.two_active = threading.Event()
            self.active = 0
            self.maximum = 0

        def serve_connection(self, connection, *, read_timeout):
            with connection:
                with self.lock:
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                    if self.active == 2:
                        self.two_active.set()
                self.release.wait(2)
                with self.lock:
                    self.active -= 1

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    agent = BlockingAgent()
    stop = threading.Event()
    transport = UnixSocketAgentServer(
        agent,
        listener=listener,
        max_connections=2,
        read_timeout=0.1,
    )
    worker = threading.Thread(
        target=transport.serve_forever,
        kwargs={"stop_event": stop},
        daemon=True,
    )
    worker.start()
    clients = [
        socket.create_connection(listener.getsockname(), timeout=1)
        for _ in range(3)
    ]
    try:
        assert agent.two_active.wait(1)
        time.sleep(0.05)
        assert agent.maximum == 2
    finally:
        agent.release.set()
        for client in clients:
            client.close()
        stop.set()
        transport.close()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert agent.maximum == 2


def test_systemd_socket_activation_rejects_untrusted_environment():
    agent = AgentServer(SECRET, _FakeRunner())
    with pytest.raises(ValueError, match="systemd socket activation"):
        UnixSocketAgentServer.from_systemd(
            agent,
            descriptor=-1,
            environment={"LISTEN_PID": str(os.getpid()), "LISTEN_FDS": "1"},
        )
    with pytest.raises(ValueError, match="systemd socket activation"):
        UnixSocketAgentServer.from_systemd(
            agent,
            descriptor=3,
            environment={"LISTEN_PID": "1", "LISTEN_FDS": "1"},
        )
    with pytest.raises(ValueError, match="systemd socket activation"):
        UnixSocketAgentServer.from_systemd(
            agent,
            descriptor=3,
            environment={"LISTEN_PID": str(os.getpid()), "LISTEN_FDS": "2"},
        )


@pytest.mark.skipif(os.name != "posix", reason="systemd file descriptors are POSIX")
def test_systemd_socket_activation_adopts_unix_stream_listener(tmp_path):
    socket_path = tmp_path / "activated.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.fspath(socket_path))
    listener.listen(1)
    try:
        transport = UnixSocketAgentServer.from_systemd(
            AgentServer(SECRET, _FakeRunner()),
            descriptor=listener.fileno(),
            environment={"LISTEN_PID": str(os.getpid()), "LISTEN_FDS": "1"},
        )
        transport.close()
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        listener.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
def test_unix_socket_server_serves_real_path_and_cleans_up(tmp_path):
    socket_path = tmp_path / "ops.sock"
    agent = AgentServer(SECRET, _FakeRunner())
    transport = UnixSocketAgentServer(
        agent,
        socket_path=socket_path,
        read_timeout=1,
    )
    stop = threading.Event()
    worker = threading.Thread(
        target=transport.serve_forever,
        kwargs={"stop_event": stop},
        daemon=True,
    )
    worker.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(os.fspath(socket_path))
        request = _server_request(
            "agent.health",
            {},
            nonce="real-unix",
            now=int(time.time()),
        )
        write_frame(client, request)
        response = read_frame(client)
        _assert_signed_response(response, request_id="request-real-unix")
    finally:
        client.close()
        stop.set()
        transport.close()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert not socket_path.exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
def test_unix_socket_server_refuses_to_replace_active_listener(tmp_path):
    socket_path = tmp_path / "active.sock"
    existing = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    existing.bind(os.fspath(socket_path))
    existing.listen(1)
    try:
        with pytest.raises(ValueError, match="active socket"):
            UnixSocketAgentServer(
                AgentServer(SECRET, _FakeRunner()),
                socket_path=socket_path,
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(os.fspath(socket_path))
        finally:
            probe.close()
    finally:
        existing.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX inode ownership only")
def test_unix_socket_server_close_preserves_replaced_path(tmp_path):
    socket_path = tmp_path / "replace.sock"
    transport = UnixSocketAgentServer(
        AgentServer(SECRET, _FakeRunner()),
        socket_path=socket_path,
    )
    socket_path.unlink()
    socket_path.write_text("replacement", encoding="utf-8")

    transport.close()

    assert socket_path.read_text(encoding="utf-8") == "replacement"


def _test_action(tmp_path, code, *, timeout=5, environment=()):
    return ValidatedAction(
        action="test.execute",
        argv=(sys.executable, "-c", code),
        cwd=str(tmp_path),
        timeout=timeout,
        read_only=True,
        approval=None,
        environment=environment,
    )


VALID_APPROVAL = {
    "plan_id": "plan-1",
    "step_id": "step-1",
    "approved_by": "admin",
    "approved_at": "2026-08-16T00:00:00Z",
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
        ("systemd.status", {"service": "1"}),
        ("systemd.journal", {"service": "0001", "tail": 10}),
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


@pytest.mark.parametrize("service", ["ssh", "nginx.service", "worker@.service"])
def test_policy_systemd_accepts_non_numeric_unit_names(service):
    validated = validate_action("systemd.status", {"service": service}, approval=None)

    assert validated.argv[-1] == service


def test_policy_shell_requires_complete_approval_and_binds_bash_argv():
    parameters = {"command": "printf ok", "cwd": "/var/tmp", "timeout": 30}

    with pytest.raises(PolicyError, match="approval"):
        validate_action("shell.execute", parameters, approval=None)

    validated = validate_action("shell.execute", parameters, approval=VALID_APPROVAL)

    assert validated.action == "shell.execute"
    assert validated.argv == (
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        "printf ok",
    )
    assert validated.cwd == "/var/tmp"
    assert validated.timeout == 30
    assert validated.read_only is False
    assert validated.approval is not None
    assert validated.approval.plan_id == "plan-1"
    assert validated.approval.step_id == "step-1"
    assert validated.approval.approved_by == "admin"
    assert validated.approval.approved_at == "2026-08-16T00:00:00Z"


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


@pytest.mark.parametrize(
    "identifier",
    [
        " leading",
        "contains space",
        "-leading-option",
        "unicode-\u8bb0",
        "control-\x85",
        "bidi-\u202e",
        "line-\u2028separator",
        "paragraph-\u2029separator",
        "path/segment",
    ],
)
def test_policy_approval_identifiers_are_strict_ascii(identifier):
    approval = dict(VALID_APPROVAL, approved_by=identifier)

    with pytest.raises(PolicyError, match="^invalid approval$") as exc_info:
        validate_action(
            "shell.execute",
            {"command": "id", "cwd": "/"},
            approval=approval,
        )

    assert identifier not in str(exc_info.value)


def test_policy_approval_rejects_more_than_five_minutes_in_request_future():
    at_boundary = dict(VALID_APPROVAL, approved_at="2026-08-17T00:05:00Z")
    too_far = dict(VALID_APPROVAL, approved_at="2026-08-17T00:05:01Z")
    reference = 1_786_924_800

    accepted = validate_action(
        "shell.execute",
        {"command": "id", "cwd": "/"},
        approval=at_boundary,
        request_timestamp=reference,
    )

    assert accepted.approval is not None
    assert accepted.approval.approved_at == "2026-08-17T00:05:00Z"
    with pytest.raises(PolicyError, match="^invalid approval$"):
        validate_action(
            "shell.execute",
            {"command": "id", "cwd": "/"},
            approval=too_far,
            request_timestamp=reference,
        )


def test_policy_approval_rejects_far_future_timestamp_against_system_time():
    approval = dict(VALID_APPROVAL, approved_at="9999-12-31T23:59:59Z")

    with pytest.raises(PolicyError, match="^invalid approval$"):
        validate_action(
            "shell.execute",
            {"command": "id", "cwd": "/"},
            approval=approval,
        )


@pytest.mark.parametrize("request_timestamp", [True, 1.0, "1786924800", -1, 10**100])
def test_policy_shell_rejects_invalid_request_timestamp(request_timestamp):
    with pytest.raises(PolicyError, match="^invalid request timestamp$"):
        validate_action(
            "shell.execute",
            {"command": "id", "cwd": "/"},
            approval=VALID_APPROVAL,
            request_timestamp=request_timestamp,
        )


def test_policy_read_tools_ignore_request_timestamp():
    validated = validate_action(
        "host.memory",
        {},
        approval=None,
        request_timestamp=10**100,
    )

    assert validated.read_only is True


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


def test_streaming_redactor_holds_split_credentials_until_they_are_safe():
    redactor = StreamingRedactor()
    chunks = [
        b"before Authorization: Bea",
        b"rer bearer-secret ",
        b"api_",
        b"key='assignment-secret' hf_",
        b"hub-secret dgx_agent-secret after",
    ]

    intermediate = [redactor.feed(chunk) for chunk in chunks]
    output = b"".join([*intermediate, redactor.finish()]).decode("utf-8")

    assert not any(b"bearer-secret" in chunk for chunk in intermediate)
    assert "bearer-secret" not in output
    assert "assignment-secret" not in output
    assert "hf_hub-secret" not in output
    assert "dgx_agent-secret" not in output
    assert output.count("[REDACTED]") == 4
    assert output.startswith("before Authorization: Bearer [REDACTED]")
    assert output.endswith(" after")


def test_streaming_redactor_handles_split_and_invalid_utf8_without_leaking():
    encoded = "prefix 记 token=secret-value suffix".encode()
    split = encoded.index("记".encode()) + 1
    redactor = StreamingRedactor()

    output = b"".join(
        (
            redactor.feed(encoded[:split]),
            redactor.feed(encoded[split:] + b"\xff"),
            redactor.finish(),
        )
    )

    decoded = output.decode("utf-8")
    assert "记" in decoded
    assert "secret-value" not in decoded
    assert "token=[REDACTED]" in decoded
    assert "\ufffd" in decoded


def _redact_chunks(chunks):
    redactor = StreamingRedactor()
    return b"".join(
        [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
    )


@pytest.mark.parametrize(
    ("payload", "secret", "expected"),
    [
        (
            b'{"DGX_OPS_RECOVERY_TOKEN": "recovery-secret"}',
            b"recovery-secret",
            b'{"DGX_OPS_RECOVERY_TOKEN": "[REDACTED]"}',
        ),
        (
            b'{"DGX_API_KEY": "api-secret"}',
            b"api-secret",
            b'{"DGX_API_KEY": "[REDACTED]"}',
        ),
        (
            b'{"HF_ENDPOINT": "endpoint-secret"}',
            b"endpoint-secret",
            b'{"HF_ENDPOINT": "[REDACTED]"}',
        ),
    ],
)
def test_streaming_redactor_namespaced_direct_json_keys_are_chunk_equivalent(
    payload,
    secret,
    expected,
):
    whole = _redact_chunks([payload])
    assert whole == expected

    for split in range(len(payload) + 1):
        split_output = _redact_chunks([payload[:split], payload[split:]])
        assert split_output == whole, f"split={split}"
        assert secret not in split_output

    bytewise = _redact_chunks([bytes((byte,)) for byte in payload])
    assert bytewise == whole
    assert secret not in bytewise


@pytest.mark.parametrize("backslash_count", [0, 2, 4, 6])
def test_streaming_redactor_assignment_opening_quote_is_chunk_equivalent(
    backslash_count,
):
    payload = b'API_KEY="secret' + b"\\" * backslash_count + b'" tail'
    expected = b'API_KEY="[REDACTED]" tail'
    whole = _redact_chunks([payload])
    assert whole == expected

    for split in range(len(payload) + 1):
        split_output = _redact_chunks([payload[:split], payload[split:]])
        assert split_output == whole, f"split={split}"
        assert b"secret" not in split_output

    bytewise = _redact_chunks([bytes((byte,)) for byte in payload])
    assert bytewise == whole
    assert b"secret" not in bytewise


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"prefix dgx_token-value suffix", b"prefix [REDACTED] suffix"),
        (b'"dgx_token-value" tail', b'"[REDACTED]" tail'),
        (b'{"value":"dgx_token-value"}', b'{"value":"[REDACTED]"}'),
        (
            b'{"PUBLIC_KEY":"visible","value":"ordinary"}',
            b'{"PUBLIC_KEY":"visible","value":"ordinary"}',
        ),
    ],
)
def test_streaming_redactor_direct_and_plain_json_are_chunk_equivalent(
    payload,
    expected,
):
    whole = _redact_chunks([payload])
    assert whole == expected

    for split in range(len(payload) + 1):
        assert _redact_chunks([payload[:split], payload[split:]]) == whole
    assert _redact_chunks([bytes((byte,)) for byte in payload]) == whole


@pytest.mark.parametrize("whitespace", [b"\n", b"\t", b"\r\n", b" \t\r\n"])
def test_streaming_redactor_direct_noncolon_whitespace_is_byte_exact(whitespace):
    payload = b'"dgx_token-value"' + whitespace + b"ERROR next-line"
    expected = b'"[REDACTED]"' + whitespace + b"ERROR next-line"
    whole = _redact_chunks([payload])
    assert whole == expected

    for split in range(len(payload) + 1):
        assert _redact_chunks([payload[:split], payload[split:]]) == whole
    assert _redact_chunks([bytes((byte,)) for byte in payload]) == whole


@pytest.mark.parametrize(
    ("key", "secret_values", "expected_key"),
    [
        (b"HF_ENDPOINT", (), b"HF_ENDPOINT"),
        (b"API_KEY", (), b"API_KEY"),
        (b"configured-secret", ("configured-secret",), b"[REDACTED]"),
    ],
)
def test_streaming_redactor_json_key_mixed_whitespace_is_byte_exact(
    key,
    secret_values,
    expected_key,
):
    whitespace = b" \t\r\n "
    secret = b"mixed-whitespace-secret"
    payload = b'{"' + key + b'"' + whitespace + b': "' + secret + b'"}'
    expected = b'{"' + expected_key + b'"' + whitespace + b': "[REDACTED]"}'

    def redact(chunks):
        redactor = StreamingRedactor(secret_values=secret_values)
        return b"".join(
            [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
        )

    whole = redact([payload])
    assert whole == expected
    for split in range(len(payload) + 1):
        assert redact([payload[:split], payload[split:]]) == whole
    assert redact([bytes((byte,)) for byte in payload]) == whole


def test_streaming_redactor_bounds_direct_key_and_whitespace_decisions():
    long_key_payload = b'{"DGX_' + b"A" * 1024 + b'": "long-key-secret"}'
    long_key_output = _redact_chunks(
        [bytes((byte,)) for byte in long_key_payload]
    )
    assert b"long-key-secret" not in long_key_output
    assert long_key_output == b'{"[REDACTED]": "[REDACTED]"}'

    redactor = StreamingRedactor()
    prefix = redactor.feed(b'"dgx_token"')
    whitespace = b" \t\r\n" * 256
    released = redactor.feed(whitespace)
    assert released == b'[REDACTED]"' + whitespace
    assert len(redactor._key_whitespace) <= 128
    suffix = redactor.feed(b"plain") + redactor.finish()
    assert prefix + released + suffix == b'"[REDACTED]"' + whitespace + b"plain"


def test_streaming_redactor_never_restores_configured_secret_used_as_json_key():
    payload = b'{"configured-secret": "value-secret"}'
    expected = b'{"[REDACTED]": "[REDACTED]"}'

    for chunks in (
        [payload],
        [payload[:10], payload[10:]],
        [bytes((byte,)) for byte in payload],
    ):
        redactor = StreamingRedactor(secret_values=("configured-secret",))
        output = b"".join(
            [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
        )
        assert output == expected
        assert b"configured-secret" not in output
        assert b"value-secret" not in output


@pytest.mark.parametrize(
    ("key", "space_count", "secret_values", "expected_key"),
    [
        (b"DGX_API_KEY", 33, (), b"DGX_API_KEY"),
        (b"API_KEY", 96, (), b"API_KEY"),
        (b"configured-secret", 33, ("configured-secret",), b"[REDACTED]"),
        (b"HF_ENDPOINT", 4096, (), b"[REDACTED]"),
    ],
)
def test_streaming_redactor_json_key_whitespace_is_unbounded_and_chunk_equivalent(
    key,
    space_count,
    secret_values,
    expected_key,
):
    secret = b"must-not-leak"
    spaces = b" " * space_count
    payload = b'{"' + key + b'"' + spaces + b': "' + secret + b'"}'
    expected = b'{"' + expected_key + b'"' + spaces + b': "[REDACTED]"}'

    def redact(chunks):
        redactor = StreamingRedactor(secret_values=secret_values)
        return b"".join(
            [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
        )

    whole = redact([payload])
    assert whole == expected
    for split in range(len(payload) + 1):
        split_output = redact([payload[:split], payload[split:]])
        assert split_output == whole, f"split={split}"
        assert secret not in split_output

    bytewise = redact([bytes((byte,)) for byte in payload])
    assert bytewise == whole
    assert secret not in bytewise


def test_streaming_redactor_json_key_wait_state_is_strictly_bounded():
    redactor = StreamingRedactor()
    whitespace = b" \t\r\n" * (4 * 1024)
    prefix = b'{"HF_ENDPOINT"' + whitespace

    released = redactor.feed(prefix)

    string_state_size = sum(
        len(value)
        for value in vars(redactor).values()
        if isinstance(value, str)
    )
    assert released == b'{"[REDACTED]"' + whitespace
    assert redactor._buffer == ""
    assert string_state_size <= 512
    assert len(redactor._key_whitespace) <= 128

    suffix = redactor.feed(b': "bounded-state-secret"}') + redactor.finish()
    output = released + suffix
    assert b"bounded-state-secret" not in output
    assert output.endswith(b': "[REDACTED]"}')


def test_streaming_redactor_covers_namespaced_sensitive_assignments():
    redactor = StreamingRedactor()
    chunks = [
        b"AWS_SECRET_AC",
        b"CESS_KEY=aws-value OPENAI_API_",
        b"KEY:'openai-value' POSTGRES_PASSWORD=db-value ",
        b"CLIENT_SECRET=client-value PRIVATE_KEY=private-value ",
        b'"DGX_OPS_RECOVERY_TOKEN": "internal-value" ',
        b"NOTSECRET=visible almost_notsecret=also-visible PUBLIC_KEY=public-value",
    ]

    output = b"".join(
        [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
    ).decode("utf-8")

    for secret in (
        "aws-value",
        "openai-value",
        "db-value",
        "client-value",
        "private-value",
        "internal-value",
    ):
        assert secret not in output
    assert output.count("[REDACTED]") == 6
    assert "NOTSECRET=visible" in output
    assert "almost_notsecret=also-visible" in output
    assert "PUBLIC_KEY=public-value" in output


@pytest.mark.parametrize("split_offset", [97, 99, 101, 199])
def test_streaming_redactor_finds_sensitive_suffix_after_long_namespace(split_offset):
    payload = (b"A_" * 100) + b"API_KEY=leaked "
    redactor = StreamingRedactor()

    output = b"".join(
        [
            redactor.feed(payload[:split_offset]),
            redactor.feed(payload[split_offset:]),
            redactor.finish(),
        ]
    ).decode("utf-8")

    assert "leaked" not in output
    assert output.endswith("API_KEY=[REDACTED] ")


@pytest.mark.parametrize(
    "visible_assignment",
    [
        b"NOTSECRET=visible ",
        b"almost_notsecret=also-visible ",
        b"PUBLIC_KEY=public-value ",
        b"MONKEY=value ",
    ],
)
def test_streaming_redactor_does_not_treat_plain_suffixes_as_secrets(
    visible_assignment,
):
    redactor = StreamingRedactor()

    output = b"".join(
        [
            *(redactor.feed(bytes((byte,))) for byte in visible_assignment),
            redactor.finish(),
        ]
    )

    assert output == visible_assignment


@pytest.mark.parametrize("backslash_count", [2, 4, 6])
def test_streaming_redactor_closes_quote_after_even_cross_chunk_backslashes(
    backslash_count,
):
    redactor = StreamingRedactor()
    chunks = [b'API_KEY="secret']
    chunks.extend(b"\\" for _ in range(backslash_count))
    chunks.extend((b'" visible',))

    output = b"".join(
        [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
    ).decode("utf-8")

    assert output == 'API_KEY="[REDACTED]" visible'


@pytest.mark.parametrize("backslash_count", [1, 3, 5])
def test_streaming_redactor_keeps_quote_escaped_after_odd_cross_chunk_backslashes(
    backslash_count,
):
    redactor = StreamingRedactor()
    chunks = [b'API_KEY="secret']
    chunks.extend(b"\\" for _ in range(backslash_count))
    chunks.extend((b'"still-secret" visible',))

    output = b"".join(
        [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
    ).decode("utf-8")

    assert "still-secret" not in output
    assert output == 'API_KEY="[REDACTED]" visible'


@pytest.mark.parametrize("backslash_count", [2, 3, 6])
def test_streaming_redactor_discards_unterminated_quoted_secret_at_eof(
    backslash_count,
):
    redactor = StreamingRedactor()
    chunks = [b'API_KEY="secret']
    chunks.extend(b"\\" for _ in range(backslash_count))

    output = b"".join(
        [*(redactor.feed(chunk) for chunk in chunks), redactor.finish()]
    ).decode("utf-8")

    assert output == 'API_KEY="[REDACTED]'


def test_runner_uses_clean_environment_devnull_and_merged_output(tmp_path, monkeypatch):
    monkeypatch.setenv("BASH_ENV", "inherited-bash")
    monkeypatch.setenv("ENV", "inherited-env")
    monkeypatch.setenv("LD_PRELOAD", "inherited-library")
    monkeypatch.setenv("PYTHONPATH", "inherited-python")
    monkeypatch.setenv("DGX_OPS_AGENT_SECRET", "inherited-secret")
    code = (
        "import json, os, sys; "
        "sys.stderr.write('stderr\\n'); "
        "print('stdin=' + repr(sys.stdin.read())); "
        "print(json.dumps(dict(os.environ), sort_keys=True)); "
        "print(os.environ['DGX_OPS_RECOVERY_TOKEN'])"
    )
    runner = JobRunner(tmp_path)

    job = runner.start(
        _test_action(tmp_path, code, environment=(("LANG", "runner-test"),))
    )
    result = _wait_for_terminal(runner, job.id)

    assert result.status == "succeeded"
    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[:2] == ["stderr", "stdin=''"]
    environment = json.loads(lines[-2])
    assert environment == {
        "DGX_OPS_RECOVERY_TOKEN": "[REDACTED]",
        "LANG": "runner-test",
        "PATH": SAFE_PATH,
    }
    assert lines[-1] == "[REDACTED]"
    assert "inherited-secret" not in result.output


def test_runner_redacts_before_bounding_output_and_tracks_byte_offsets(tmp_path):
    raw_output = "A" * 200 + " password=runner-secret " + "记" * 8
    runner = JobRunner(tmp_path, output_limit=64)

    started = runner.start(_test_action(tmp_path, f"print({raw_output!r})"))
    result = _wait_for_terminal(runner, started.id)

    assert result.status == "succeeded"
    assert "runner-secret" not in result.output
    assert "[REDACTED]" in result.output
    assert len(result.output.encode("utf-8")) <= 64
    assert result.output.encode("utf-8").decode("utf-8") == result.output
    assert result.output_offset > len(result.output.encode("utf-8"))
    assert result.truncated_before == (
        result.output_offset - len(result.output.encode("utf-8"))
    )

    retained = runner.get(started.id, offset=result.truncated_before)
    exhausted = runner.get(started.id, offset=result.output_offset)
    truncated = runner.get(started.id, offset=0)
    assert retained.output == result.output
    assert retained.truncated_before == result.truncated_before
    assert exhausted.output == ""
    assert exhausted.truncated_before == result.output_offset
    assert truncated.output == result.output
    assert truncated.truncated_before == result.truncated_before


def test_runner_accepts_explicit_argv_without_shell_interpretation(tmp_path):
    marker = tmp_path / "must-not-exist"
    argument = f"literal;touch {marker}"
    runner = JobRunner(tmp_path)

    job = runner.start(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", argument],
        cwd=tmp_path,
        timeout=5,
    )
    result = _wait_for_terminal(runner, job.id)

    assert result.status == "succeeded"
    assert result.output.strip() == argument
    assert not marker.exists()


def test_runner_times_out_quickly_and_sets_one_terminal_state(tmp_path):
    runner = JobRunner(tmp_path, termination_grace=0.05)
    job = runner.start(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        timeout=0.1,
    )

    result = _wait_for_terminal(runner, job.id)
    time.sleep(0.05)

    assert result.status == "timed_out"
    assert runner.get(job.id).status == "timed_out"
    assert result.started is not None
    assert result.finished is not None
    assert result.finished >= result.started


def test_cancel_is_idempotent_and_terminates_a_running_job(tmp_path):
    runner = JobRunner(tmp_path, termination_grace=0.05)
    job = runner.start(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        timeout=5,
    )

    first = runner.cancel(job.id)
    second = runner.cancel(job.id)
    result = _wait_for_terminal(runner, job.id)

    assert first.id == second.id == job.id
    assert result.status == "cancelled"
    assert runner.cancel(job.id).status == "cancelled"


def test_cancel_sets_event_when_leader_exited_but_output_reader_is_active(
    tmp_path, monkeypatch
):
    reader_started = threading.Event()
    release_reader = threading.Event()

    class ExitedLeader:
        pid = 4242
        stdout = io.BytesIO()

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def wait(timeout=None):
            return 0

    process = ExitedLeader()
    runner = JobRunner(tmp_path, termination_grace=0)

    def blocking_reader(_state, _process, _redactor, _errors):
        reader_started.set()
        assert release_reader.wait(1)

    def capture_identity(_pid, recovery_token):
        return runner_module._ProcessIdentity(
            pid=process.pid,
            process_group=None,
            session_id=None,
            start_time_ticks=None,
            boot_id=None,
            recovery_token=recovery_token,
        )

    monkeypatch.setattr(runner_module, "_HAS_PROCESS_GROUPS", False)
    monkeypatch.setattr(
        runner,
        "_spawn_process",
        lambda *_args, **_kwargs: (process, None),
    )
    monkeypatch.setattr(runner, "_capture_process_identity", capture_identity)
    monkeypatch.setattr(runner, "_read_output", blocking_reader)
    monkeypatch.setattr(runner, "_terminate", lambda *_args: release_reader.set())

    job = runner.start(["command"], cwd=tmp_path, timeout=0.5)
    assert reader_started.wait(1)
    started_at = time.monotonic()
    runner.cancel(job.id)
    result = _wait_for_terminal(runner, job.id, timeout=1)

    assert result.status == "cancelled"
    assert time.monotonic() - started_at < 0.25


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_cancel_terminates_the_entire_posix_process_group(tmp_path):
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(10)"
    )
    code = (
        "import pathlib, subprocess, sys, time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); "
        "time.sleep(10)"
    )
    runner = JobRunner(tmp_path, termination_grace=0.05)
    job = runner.start([sys.executable, "-c", code], cwd=tmp_path, timeout=5)
    deadline = time.monotonic() + 2
    while not child_pid.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    pid = int(child_pid.read_text())

    runner.cancel(job.id)
    assert _wait_for_terminal(runner, job.id).status == "cancelled"
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("child process group member survived cancellation")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_timeout_covers_background_group_after_leader_exits(tmp_path):
    child_code = "import time; time.sleep(3)"
    parent_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('leader-exited')"
    )
    runner = JobRunner(tmp_path, termination_grace=0.05)

    started_at = time.monotonic()
    job = runner.start(
        [sys.executable, "-c", parent_code], cwd=tmp_path, timeout=0.2
    )
    result = _wait_for_terminal(runner, job.id)

    assert result.status == "timed_out"
    assert "leader-exited" in result.output
    assert time.monotonic() - started_at < 1.5


def test_timeout_tracks_injected_process_group_after_leader_exit(
    tmp_path, monkeypatch
):
    class ExitedLeader:
        pid = 4242
        stdout = io.BytesIO()

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def wait(timeout=None):
            return 0

    group_alive = True
    signals = []

    def fake_killpg(process_group, sent_signal):
        nonlocal group_alive
        signals.append((process_group, sent_signal))
        if sent_signal == signal.SIGKILL:
            group_alive = False

    runner = JobRunner(tmp_path, termination_grace=0)
    monkeypatch.setattr(runner_module, "_HAS_PROCESS_GROUPS", True)
    monkeypatch.setattr(runner_module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(runner_module.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(
        JobRunner,
        "_process_group_exists",
        staticmethod(lambda _process_group: group_alive),
    )
    monkeypatch.setattr(
        runner,
        "_spawn_process",
        lambda *_args, **_kwargs: (ExitedLeader(), None),
    )

    job = runner.start(
        [sys.executable, "-c", "unused"], cwd=tmp_path, timeout=0.05
    )
    result = _wait_for_terminal(runner, job.id)

    assert result.status == "timed_out"
    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


def test_launcher_waits_for_go_and_eof_never_executes(tmp_path):
    from host_agent.dgx_ops_agent import launcher

    request_read, request_write = os.pipe()
    gate_read, gate_write = os.pipe()
    calls = []
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            launcher.run_launcher(
                request_read,
                gate_read,
                exec_function=lambda *args: calls.append(args),
            )
        )
    )
    worker.start()
    launcher.write_request(
        request_write,
        argv=(sys.executable, "-c", "literal;not-a-shell"),
        cwd=str(tmp_path),
        environment={"PATH": SAFE_PATH, "LANG": "C"},
    )
    time.sleep(0.02)

    assert calls == []
    os.close(gate_write)
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert calls == []
    assert result == [launcher.EXIT_GATE_CLOSED]

    request_read, request_write = os.pipe()
    gate_read, gate_write = os.pipe()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            launcher.run_launcher(
                request_read,
                gate_read,
                exec_function=lambda *args: calls.append(args),
            )
        )
    )
    worker.start()
    expected_argv = (sys.executable, "-c", "literal;not-a-shell")
    expected_environment = {"PATH": SAFE_PATH, "LANG": "C"}
    launcher.write_request(
        request_write,
        argv=expected_argv,
        cwd=str(tmp_path),
        environment=expected_environment,
    )
    assert calls == []
    os.write(gate_write, launcher.GO)
    os.close(gate_write)
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert calls == [(expected_argv[0], list(expected_argv), expected_environment)]
    assert result == [0]


def test_launcher_request_serialization_handles_utf8_worst_cases():
    from host_agent.dgx_ops_agent import launcher

    command = "x"
    control_argument = "\x01" * (launcher.MAX_ARGUMENT_BYTES - len(command))
    control_payload = launcher.serialize_request(
        argv=(command, control_argument),
        cwd="/",
        environment={"PATH": SAFE_PATH},
    )
    assert len(control_payload) <= launcher.MAX_LAUNCH_REQUEST_SIZE
    assert json.loads(control_payload.decode("utf-8"))["argv"] == [
        command,
        control_argument,
    ]

    emoji_count = (launcher.MAX_ARGUMENT_BYTES - len(command)) // 4
    emoji_remainder = launcher.MAX_ARGUMENT_BYTES - len(command) - emoji_count * 4
    emoji_argument = "\U0001f642" * emoji_count + "x" * emoji_remainder
    emoji_payload = launcher.serialize_request(
        argv=(command, emoji_argument),
        cwd="/",
        environment={"PATH": SAFE_PATH},
    )
    assert len(emoji_payload) <= launcher.MAX_LAUNCH_REQUEST_SIZE
    assert json.loads(emoji_payload.decode("utf-8"))["argv"] == [
        command,
        emoji_argument,
    ]


def test_launcher_write_request_closes_owned_descriptor_on_all_failures(
    monkeypatch,
):
    from host_agent.dgx_ops_agent import launcher

    def assert_closed_after_failure(**request):
        request_read, request_write = os.pipe()
        with pytest.raises((TypeError, ValueError)):
            launcher.write_request(request_write, **request)
        with pytest.raises(OSError):
            os.fstat(request_write)
        os.close(request_read)

    valid_request = {
        "argv": ("command",),
        "cwd": "/",
        "environment": {"PATH": SAFE_PATH},
    }
    assert_closed_after_failure(
        argv=(),
        cwd=valid_request["cwd"],
        environment=valid_request["environment"],
    )

    original_dumps = launcher.json.dumps
    monkeypatch.setattr(
        launcher.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("serialize")),
    )
    assert_closed_after_failure(**valid_request)
    monkeypatch.setattr(launcher.json, "dumps", original_dumps)

    assert_closed_after_failure(
        argv=valid_request["argv"],
        cwd="x" * launcher.MAX_LAUNCH_REQUEST_SIZE,
        environment=valid_request["environment"],
    )


def test_runner_rejects_oversized_launcher_request_before_creating_job(tmp_path):
    from host_agent.dgx_ops_agent import launcher

    runner = JobRunner(tmp_path)

    with pytest.raises(ValueError, match="launcher request"):
        runner.start(
            ["command"],
            cwd="x" * launcher.MAX_LAUNCH_REQUEST_SIZE,
            environment={"LANG": "C"},
        )

    assert list(tmp_path.glob("*.json")) == []
    assert runner._jobs == {}


def test_spawn_closes_first_pipe_when_second_pipe_creation_fails(
    tmp_path, monkeypatch
):
    runner = JobRunner(tmp_path)
    real_pipe = os.pipe
    created_descriptors = []
    calls = 0

    def fail_second_pipe():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second pipe failure")
        descriptors = real_pipe()
        created_descriptors.extend(descriptors)
        return descriptors

    monkeypatch.setattr(runner_module, "_HAS_PROCESS_GROUPS", True)
    monkeypatch.setattr(runner_module.os, "pipe", fail_second_pipe)

    with pytest.raises(OSError, match="second pipe failure"):
        runner._spawn_process(
            ("command",),
            None,
            {"PATH": SAFE_PATH},
            "recovery-token",
        )

    assert len(created_descriptors) == 2
    for descriptor in created_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_spawn_closes_all_pipes_when_launcher_resolution_fails(
    tmp_path, monkeypatch
):
    runner = JobRunner(tmp_path)
    real_pipe = os.pipe
    created_descriptors = []

    def tracking_pipe():
        descriptors = real_pipe()
        created_descriptors.extend(descriptors)
        return descriptors

    def fail_resolve(*_args, **_kwargs):
        raise OSError("injected launcher resolution failure")

    monkeypatch.setattr(runner_module, "_HAS_PROCESS_GROUPS", True)
    monkeypatch.setattr(runner_module.os, "pipe", tracking_pipe)
    monkeypatch.setattr(runner_module.Path, "resolve", fail_resolve)

    with pytest.raises(OSError, match="launcher resolution failure"):
        runner._spawn_process(
            ("command",),
            None,
            {"PATH": SAFE_PATH},
            "recovery-token",
        )

    assert len(created_descriptors) == 4
    for descriptor in created_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_cancel_before_launcher_go_never_executes_and_reaps(
    tmp_path, monkeypatch
):
    marker = tmp_path / "must-not-run"
    ready_persisted = threading.Event()
    release_worker = threading.Event()
    launcher_exited = threading.Event()
    wait_calls = []
    gate_read, gate_write = os.pipe()

    class FakeLauncherProcess:
        pid = 4242
        stdout = io.BytesIO()
        exit_code = None

        def poll(self):
            return self.exit_code

        def wait(self, timeout=None):
            wait_calls.append(timeout)
            if not launcher_exited.wait(timeout):
                raise subprocess.TimeoutExpired("launcher", timeout)
            return self.exit_code

    process = FakeLauncherProcess()

    def fake_launcher():
        try:
            if os.read(gate_read, 1) == b"G":
                marker.touch()
                process.exit_code = 0
            else:
                process.exit_code = 75
        finally:
            os.close(gate_read)
            launcher_exited.set()

    launcher_thread = threading.Thread(target=fake_launcher)
    launcher_thread.start()
    runner = JobRunner(tmp_path, termination_grace=0.05)
    original_persist = runner._persist
    blocked_once = False

    def block_after_ready_fsync(state):
        nonlocal blocked_once
        original_persist(state)
        if state.launch_phase == "ready" and not blocked_once:
            blocked_once = True
            runner._lock.release()
            try:
                ready_persisted.set()
                assert release_worker.wait(1)
            finally:
                runner._lock.acquire()

    def capture_identity(_pid, recovery_token):
        return runner_module._ProcessIdentity(
            pid=process.pid,
            process_group=process.pid,
            session_id=process.pid,
            start_time_ticks=1,
            boot_id="boot-id",
            recovery_token=recovery_token,
        )

    monkeypatch.setattr(runner_module, "_HAS_PROCESS_GROUPS", True)
    monkeypatch.setattr(runner, "_persist", block_after_ready_fsync)
    monkeypatch.setattr(
        runner,
        "_spawn_process",
        lambda *_args, **_kwargs: (process, gate_write),
    )
    monkeypatch.setattr(runner, "_capture_process_identity", capture_identity)
    monkeypatch.setattr(
        runner,
        "_process_group_exists",
        lambda _process_group: process.poll() is None,
    )
    monkeypatch.setattr(
        runner,
        "_terminate",
        lambda *_args: setattr(process, "exit_code", -signal.SIGTERM),
    )
    monkeypatch.setattr(
        runner,
        "_kill",
        lambda *_args: setattr(process, "exit_code", -signal.SIGKILL),
    )

    job = runner.start(["command"], cwd=tmp_path, timeout=1)
    assert ready_persisted.wait(1)
    cancelled = runner.cancel(job.id)
    assert cancelled.status == "running"
    release_worker.set()

    result = _wait_for_terminal(runner, job.id)
    launcher_thread.join(timeout=1)
    assert not launcher_thread.is_alive()
    assert result.status == "cancelled"
    assert not marker.exists()
    assert wait_calls


def test_runner_persists_pre_spawn_recovery_token(tmp_path, monkeypatch):
    observed_launch = {}

    def refuse_spawn(*_args, **_kwargs):
        metadata_path = next(tmp_path.glob("*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        observed_launch.update(metadata.get("launch") or {})
        raise OSError("injected spawn refusal")

    runner = JobRunner(tmp_path)
    monkeypatch.setattr(runner_module, "_HAS_PROCESS_GROUPS", True)
    monkeypatch.setattr(runner_module.subprocess, "Popen", refuse_spawn)

    job = runner.start(
        [sys.executable, "-c", "unused"], cwd=tmp_path, timeout=1
    )
    result = _wait_for_terminal(runner, job.id)

    assert result.status == "failed"
    assert observed_launch["phase"] == "prepared"
    assert len(observed_launch["recovery_token"]) >= 32


@pytest.mark.skipif(os.name != "posix", reason="POSIX launcher handshake only")
def test_launcher_persist_failure_before_go_never_executes_command(
    tmp_path, monkeypatch
):
    marker = tmp_path / "must-not-run"
    runner = JobRunner(tmp_path, termination_grace=0.05)
    original_persist = runner._persist
    injected = False

    def fail_ready_identity(state):
        nonlocal injected
        if getattr(state, "launch_phase", None) == "ready" and not injected:
            injected = True
            raise OSError("ready identity persistence failed")
        return original_persist(state)

    monkeypatch.setattr(runner, "_persist", fail_ready_identity)
    job = runner.start(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
        cwd=tmp_path,
        timeout=2,
    )
    result = _wait_for_terminal(runner, job.id)

    assert injected is True
    assert result.status == "failed"
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX launcher handshake only")
def test_launcher_releases_extremely_short_command_after_identity_fsync(tmp_path):
    runner = JobRunner(tmp_path)

    result = _wait_for_terminal(
        runner,
        runner.start(["/bin/true"], cwd="/", timeout=1).id,
    )

    assert result.status == "succeeded"
    assert result.exit_code == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_cancel_covers_background_group_after_leader_exits(tmp_path):
    child_pid = tmp_path / "orphan-child.pid"
    child_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3)"
    parent_code = (
        "import pathlib, subprocess, sys; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))"
    )
    runner = JobRunner(tmp_path, termination_grace=0.05)
    job = runner.start(
        [sys.executable, "-c", parent_code], cwd=tmp_path, timeout=2
    )
    deadline = time.monotonic() + 1
    while not child_pid.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid.exists()
    time.sleep(0.05)

    started_at = time.monotonic()
    runner.cancel(job.id)
    result = _wait_for_terminal(runner, job.id)

    assert result.status == "cancelled"
    assert time.monotonic() - started_at < 1.5


def test_runner_metadata_is_atomic_redacted_and_recovers_interrupted_jobs(tmp_path):
    runner = JobRunner(tmp_path)
    command_secret = "token=metadata-command-secret"
    job = runner.start(
        _test_action(
            tmp_path,
            f"print({command_secret!r})",
            environment=(("LANG", "metadata-env-secret"),),
        )
    )
    result = _wait_for_terminal(runner, job.id)
    metadata_path = tmp_path / f"{job.id}.json"
    metadata = metadata_path.read_text(encoding="utf-8")

    assert result.status == "succeeded"
    assert "metadata-command-secret" not in metadata
    assert "metadata-env-secret" not in metadata
    assert not list(tmp_path.glob("*.tmp"))

    saved = json.loads(metadata)
    saved["status"] = "running"
    saved["finished"] = None
    metadata_path.write_text(json.dumps(saved), encoding="utf-8")
    recovered = JobRunner(tmp_path).get(job.id)
    assert recovered.status == "failed"
    assert recovered.exit_code is None
    assert recovered.finished is not None
    assert "interrupted" in recovered.error


def test_runner_metadata_never_contains_namespaced_assignment_values(tmp_path):
    assignments = (
        "AWS_SECRET_ACCESS_KEY=aws-metadata-value ",
        "OPENAI_API_KEY=openai-metadata-value ",
        "POSTGRES_PASSWORD=postgres-metadata-value ",
        "CLIENT_SECRET=client-metadata-value ",
        "PRIVATE_KEY=private-metadata-value",
    )
    runner = JobRunner(tmp_path)
    job = runner.start(_test_action(tmp_path, f"print({''.join(assignments)!r})"))

    result = _wait_for_terminal(runner, job.id)
    metadata = (tmp_path / f"{job.id}.json").read_text(encoding="utf-8")

    assert result.output.count("[REDACTED]") == 5
    for secret in (
        "aws-metadata-value",
        "openai-metadata-value",
        "postgres-metadata-value",
        "client-metadata-value",
        "private-metadata-value",
    ):
        assert secret not in result.output
        assert secret not in metadata


def test_runner_persists_private_process_identity_immediately_after_spawn(tmp_path):
    runner = JobRunner(tmp_path, termination_grace=0.05)
    job = runner.start(
        [sys.executable, "-c", "import time; time.sleep(3)"],
        cwd=tmp_path,
        timeout=5,
    )
    metadata_path = tmp_path / f"{job.id}.json"
    deadline = time.monotonic() + 1
    identity = None
    while time.monotonic() < deadline:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = metadata.get("process_identity")
        if metadata["status"] == "running" and identity and identity.get("pid"):
            break
        time.sleep(0.01)

    assert identity is not None
    assert identity["pid"] > 0
    assert identity["recovery_token"]
    assert set(identity) == {
        "pid",
        "process_group",
        "session_id",
        "start_time_ticks",
        "boot_id",
        "recovery_token",
    }
    assert "recovery_token" not in job.as_dict()
    assert identity["recovery_token"] not in json.dumps(job.as_dict())
    runner.cancel(job.id)
    assert _wait_for_terminal(runner, job.id).status == "cancelled"


def test_runner_cleans_up_when_spawn_identity_cannot_be_persisted(
    tmp_path, monkeypatch
):
    runner = JobRunner(tmp_path, termination_grace=0.05)
    original_persist = runner._persist
    injected = False

    def fail_first_identity_persist(state):
        nonlocal injected
        if state.process_identity is not None and not injected:
            injected = True
            raise OSError("identity persistence failed")
        return original_persist(state)

    monkeypatch.setattr(runner, "_persist", fail_first_identity_persist)

    job = runner.start(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        cwd=tmp_path,
        timeout=5,
    )
    result = _wait_for_terminal(runner, job.id, timeout=0.8)

    assert injected is True
    assert result.status == "failed"
    assert "identity persistence failed" in result.error


def test_restart_kills_only_exact_fake_proc_identity_match(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    proc_root = tmp_path / "proc"
    jobs.mkdir()
    boot_id = "11111111-1111-4111-8111-111111111111"
    (proc_root / "sys" / "kernel" / "random").mkdir(parents=True)
    (proc_root / "sys" / "kernel" / "random" / "boot_id").write_text(
        f"{boot_id}\n", encoding="ascii"
    )

    def write_process(pid, token):
        process_dir = proc_root / str(pid)
        process_dir.mkdir()
        stat_fields = ["S", "1", str(pid), str(pid), *(["0"] * 15), "777"]
        (process_dir / "stat").write_text(
            f"{pid} (python worker) {' '.join(stat_fields)}\n", encoding="ascii"
        )
        (process_dir / "environ").write_bytes(
            f"DGX_OPS_RECOVERY_TOKEN={token}\0".encode()
        )

    def write_job(job_id, pid, token):
        metadata = {
            "job_id": job_id,
            "status": "running",
            "output": "",
            "output_offset": 0,
            "truncated_before": 0,
            "exit_code": None,
            "started": 1.0,
            "finished": None,
            "error": None,
            "process_identity": {
                "pid": pid,
                "process_group": pid,
                "session_id": pid,
                "start_time_ticks": 777,
                "boot_id": boot_id,
                "recovery_token": token,
            },
        }
        path = jobs / f"{job_id}.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        path.chmod(0o600)

    matching_id = "00000000-0000-4000-8000-000000000101"
    mismatch_id = "00000000-0000-4000-8000-000000000202"
    write_process(101, "matching-recovery-token")
    write_process(202, "different-process-token")
    write_job(matching_id, 101, "matching-recovery-token")
    write_job(mismatch_id, 202, "metadata-token-does-not-match")

    opened_pids = []
    descriptor_pids = {}
    sent_signals = []

    def fake_pidfd_open(pid, flags=0):
        assert flags == 0
        opened_pids.append(pid)
        descriptor = os.open(os.devnull, os.O_RDONLY)
        descriptor_pids[descriptor] = pid
        return descriptor

    def fake_pidfd_send_signal(descriptor, sent_signal, siginfo=None, flags=0):
        assert siginfo is None
        assert flags == 0
        pid = descriptor_pids[descriptor]
        sent_signals.append((pid, sent_signal))
        if sent_signal == 9:
            process_dir = proc_root / str(pid)
            (process_dir / "environ").unlink()
            (process_dir / "stat").unlink()
            process_dir.rmdir()

    def forbid_recovery_killpg(*_args):
        pytest.fail("restart recovery must not call killpg")

    monkeypatch.setattr(runner_module, "PROC_ROOT", proc_root, raising=False)
    monkeypatch.setattr(runner_module, "_IS_LINUX", True, raising=False)
    monkeypatch.setattr(runner_module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(runner_module.os, "pidfd_open", fake_pidfd_open, raising=False)
    monkeypatch.setattr(
        runner_module.signal,
        "pidfd_send_signal",
        fake_pidfd_send_signal,
        raising=False,
    )
    monkeypatch.setattr(runner_module.os, "killpg", forbid_recovery_killpg, raising=False)

    runner = JobRunner(jobs, termination_grace=0)

    assert opened_pids == [101, 101]
    assert sent_signals == [(101, signal.SIGTERM), (101, 9)]
    assert runner.get(matching_id).status == "failed"
    assert runner.get(mismatch_id).status == "failed"
    assert "interrupted" in runner.get(matching_id).error
    assert "interrupted" in runner.get(mismatch_id).error
    assert json.loads((jobs / f"{matching_id}.json").read_text())["process_identity"]

    unsupported_id = "00000000-0000-4000-8000-000000000303"
    write_process(303, "unsupported-pidfd-token")
    write_job(unsupported_id, 303, "unsupported-pidfd-token")
    monkeypatch.delattr(runner_module.os, "pidfd_open")
    monkeypatch.delattr(runner_module.signal, "pidfd_send_signal")

    unsupported_runner = JobRunner(jobs, termination_grace=0)

    assert unsupported_runner.get(unsupported_id).status == "failed"
    assert "interrupted" in unsupported_runner.get(unsupported_id).error
    assert opened_pids == [101, 101]
    assert sent_signals == [(101, signal.SIGTERM), (101, 9)]


def test_runner_rejects_invalid_limits_offsets_environment_and_job_ids(tmp_path):
    with pytest.raises(ValueError, match="output_limit"):
        JobRunner(tmp_path, output_limit=0)
    with pytest.raises(ValueError, match="output_limit"):
        JobRunner(tmp_path, output_limit=20_000_000)

    runner = JobRunner(tmp_path)
    with pytest.raises(ValueError, match="environment"):
        runner.start(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            timeout=1,
            environment={"PYTHONPATH": "unsafe"},
        )
    with pytest.raises(KeyError, match="job not found"):
        runner.get("../metadata")
    with pytest.raises(KeyError, match="job not found"):
        runner.cancel("../metadata")

    result = _wait_for_terminal(
        runner,
        runner.start([sys.executable, "-c", "print('ok')"], cwd=tmp_path, timeout=1).id,
    )
    with pytest.raises(ValueError, match="offset"):
        runner.get(result.id, offset=-1)
    with pytest.raises(ValueError, match="offset"):
        runner.get(result.id, offset=result.output_offset + 1)


def test_runner_job_directory_chmod_failure_is_fatal(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    original_chmod = Path.chmod

    def deny_job_directory_chmod(path, mode):
        if path == job_dir:
            raise PermissionError("chmod denied")
        return original_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", deny_job_directory_chmod)

    with pytest.raises(PermissionError, match="chmod denied"):
        JobRunner(job_dir)


def test_runner_metadata_chmod_failure_makes_start_fail(tmp_path, monkeypatch):
    runner = JobRunner(tmp_path)
    original_chmod = Path.chmod

    def deny_metadata_chmod(path, mode):
        if path.suffix == ".json":
            raise PermissionError("metadata chmod denied")
        return original_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", deny_metadata_chmod)

    with pytest.raises(PermissionError, match="metadata chmod denied"):
        runner.start([sys.executable, "-c", "pass"], cwd=tmp_path, timeout=1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and modes only")
def test_runner_rejects_symlink_job_directory(tmp_path):
    real_directory = tmp_path / "real-jobs"
    real_directory.mkdir(mode=0o700)
    symlink = tmp_path / "linked-jobs"
    symlink.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="job_dir"):
        JobRunner(symlink)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and modes only")
def test_runner_enforces_private_directory_and_regular_metadata_modes(tmp_path):
    job_dir = tmp_path / "jobs"
    job_dir.mkdir(mode=0o755)
    runner = JobRunner(job_dir)
    directory_stat = job_dir.stat()

    assert directory_stat.st_uid == os.getuid()
    assert stat.S_IMODE(directory_stat.st_mode) == 0o700

    result = _wait_for_terminal(
        runner,
        runner.start(
            [sys.executable, "-c", "print('ok')"], cwd=tmp_path, timeout=1
        ).id,
    )
    metadata_stat = (job_dir / f"{result.id}.json").lstat()
    assert stat.S_ISREG(metadata_stat.st_mode)
    assert metadata_stat.st_uid == os.getuid()
    assert stat.S_IMODE(metadata_stat.st_mode) == 0o600

    metadata_path = job_dir / f"{result.id}.json"
    metadata_path.chmod(0o644)
    with pytest.raises(PermissionError, match="metadata permissions"):
        JobRunner(job_dir)


def test_ci_lints_backend_and_host_agent():
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "ruff check backend host_agent" in workflow
