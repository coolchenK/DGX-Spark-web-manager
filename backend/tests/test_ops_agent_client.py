from __future__ import annotations

import socket
import struct
import threading
import time
import tomllib
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from app.config import Settings
from app.services.ops_agent import (
    OpsAgentClient,
    OpsAgentProtocolError,
    OpsAgentRemoteError,
    OpsAgentUnavailable,
)
from pydantic import ValidationError

from host_agent.dgx_ops_agent.protocol import (
    MAX_FRAME_SIZE,
    PROTOCOL_VERSION,
    encode_frame,
    read_frame,
    sign_message,
)

ResponseFactory = Callable[[dict[str, Any]], bytes]
_ACTIVE_SERVERS: dict[Path, _FakeUnixServer] = {}


def _response(
    request: dict[str, Any],
    secret: bytes,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, str] | None = None,
    request_id: str | None = None,
    timestamp: int | None = None,
) -> dict[str, Any]:
    return sign_message(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request["request_id"] if request_id is None else request_id,
            "ok": error is None,
            "result": result,
            "error": error,
            "timestamp": int(time.time()) if timestamp is None else timestamp,
        },
        secret,
    )


class _FakeUnixServer:
    def __init__(
        self,
        socket_path: Path,
        response_factory: ResponseFactory,
        *,
        response_delay: float = 0,
        trickle_delay: float = 0,
    ) -> None:
        self.socket_path = socket_path
        self.last_request: dict[str, Any] | None = None
        self._response_factory = response_factory
        self._response_delay = response_delay
        self._trickle_delay = trickle_delay
        self._ready = threading.Event()
        self._errors: list[BaseException] = []
        self._client_connection, self._server_connection = socket.socketpair()
        self._client_taken = False
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()
        assert self._ready.wait(2), "fake Unix server did not start"

    def close(self) -> None:
        self._thread.join(timeout=2)
        self._client_connection.close()
        self._server_connection.close()
        relevant = [error for error in self._errors if not isinstance(error, OSError)]
        assert not relevant, relevant

    def connection_factory(self) -> socket.socket:
        assert not self._client_taken
        self._client_taken = True
        return self._client_connection

    def _serve(self) -> None:
        try:
            self._ready.set()
            self.last_request = read_frame(self._server_connection)
            response = self._response_factory(self.last_request)
            if self._response_delay:
                time.sleep(self._response_delay)
            if self._trickle_delay:
                for byte in response:
                    self._server_connection.sendall(bytes((byte,)))
                    time.sleep(self._trickle_delay)
            else:
                self._server_connection.sendall(response)
            self._server_connection.shutdown(socket.SHUT_WR)
        except BaseException as exc:  # test helper reports thread failures to the test
            self._errors.append(exc)
        finally:
            self._ready.set()


@contextmanager
def _server(
    tmp_path: Path,
    response_factory: ResponseFactory,
    **options: Any,
) -> Iterator[_FakeUnixServer]:
    server = _FakeUnixServer(tmp_path / "ops.sock", response_factory, **options)
    server.start()
    _ACTIVE_SERVERS[server.socket_path] = server
    try:
        yield server
    finally:
        _ACTIVE_SERVERS.pop(server.socket_path, None)
        server.close()


def _client(
    socket_path: Path,
    key_path: Path,
    **options: Any,
) -> OpsAgentClient:
    server = options.pop("server", None) or _ACTIVE_SERVERS.get(socket_path)
    return OpsAgentClient(
        socket_path,
        key_path,
        connect_timeout_seconds=options.pop("connect_timeout_seconds", 0.5),
        read_timeout_seconds=options.pop("read_timeout_seconds", 0.5),
        output_limit_bytes=options.pop("output_limit_bytes", 10_000),
        _connection_factory=None if server is None else server.connection_factory,
        **options,
    )


def test_client_signs_request_and_verifies_health_response(tmp_path):
    secret = b"x" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)

    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                result={"status": "ok", "protocol_version": PROTOCOL_VERSION},
            )
        ),
    ) as server:
        health = _client(server.socket_path, key_path).health()

    assert health.status == "ok"
    assert health.protocol_version == PROTOCOL_VERSION
    assert server.last_request is not None
    assert server.last_request["action"] == "agent.health"
    assert server.last_request["parameters"] == {}
    assert len(server.last_request["signature"]) == 64


@pytest.mark.parametrize(
    "key_bytes,secret",
    [
        (b" " + b"x" * 30 + b"\n", b" " + b"x" * 30 + b"\n"),
        ((b"y" * 32).hex().encode(), b"y" * 32),
        ((b"z" * 32).hex().encode() + b"\n", b"z" * 32),
    ],
)
def test_client_accepts_raw_whitespace_and_hex_keys(tmp_path, key_bytes, secret):
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(key_bytes)
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                result={"status": "ok", "protocol_version": PROTOCOL_VERSION},
            )
        ),
    ) as server:
        assert _client(server.socket_path, key_path).health().status == "ok"


def test_client_reads_key_only_once_per_instance(tmp_path):
    secret = b"k" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    missing_socket = tmp_path / "missing.sock"
    client = _client(missing_socket, key_path)

    with pytest.raises(OpsAgentUnavailable):
        client.health()
    key_path.write_bytes(b"changed-key-that-must-not-be-read!!")

    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                result={"status": "ok", "protocol_version": PROTOCOL_VERSION},
            )
        ),
    ) as server:
        client._connection_factory = server.connection_factory
        client.socket_path = server.socket_path
        assert client.health().status == "ok"


def test_concurrent_calls_use_independent_connections(tmp_path):
    secret = b"c" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    handlers: list[threading.Thread] = []
    server_errors: list[BaseException] = []
    factory_lock = threading.Lock()
    client_connections: list[socket.socket] = []

    def handle(connection: socket.socket) -> None:
        try:
            with connection:
                request = read_frame(connection)
                connection.sendall(
                    encode_frame(_response(request, secret, result={"value": 1}))
                )
        except BaseException as exc:
            server_errors.append(exc)

    def connection_factory() -> socket.socket:
        client_connection, server_connection = socket.socketpair()
        with factory_lock:
            client_connections.append(client_connection)
        thread = threading.Thread(target=handle, args=(server_connection,), daemon=True)
        handlers.append(thread)
        thread.start()
        return client_connection

    client = OpsAgentClient(
        tmp_path / "unused.sock",
        key_path,
        connect_timeout_seconds=0.5,
        read_timeout_seconds=0.5,
        output_limit_bytes=10_000,
        _connection_factory=connection_factory,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: client.call("host.memory", {}), range(8)))
    for handler in handlers:
        handler.join(timeout=1)

    assert results == [{"value": 1}] * 8
    assert len({id(connection) for connection in client_connections}) == 8
    assert not server_errors


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: {**response, "signature": "0" * 64},
        lambda response: {**response, "request_id": "different-request"},
        lambda response: {key: value for key, value in response.items() if key != "error"},
        lambda response: {**response, "unexpected": True},
        lambda response: {**response, "timestamp": int(time.time()) - 31},
    ],
    ids=["signature", "request-id", "missing-field", "extra-field", "stale-timestamp"],
)
def test_client_rejects_invalid_authenticated_response(tmp_path, mutate):
    secret = b"s" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)

    def response_factory(request):
        valid = _response(request, secret, result={"value": 1})
        changed = mutate(valid)
        if changed.get("signature") == valid["signature"]:
            changed = sign_message(changed, secret)
        return encode_frame(changed)

    with _server(tmp_path, response_factory) as server:
        with pytest.raises(OpsAgentProtocolError) as raised:
            _client(server.socket_path, key_path).call("host.memory", {})

    assert "signature" not in str(raised.value).lower()
    assert secret.decode() not in str(raised.value)


def test_client_maps_remote_error_without_losing_code(tmp_path):
    secret = b"r" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                error={"code": "approval_required", "message": "approval required"},
            )
        ),
    ) as server:
        with pytest.raises(OpsAgentRemoteError) as raised:
            _client(server.socket_path, key_path).call("shell.execute", {})

    assert raised.value.code == "approval_required"
    assert raised.value.message == "Host operations agent returned an error"


def test_remote_error_string_does_not_echo_remote_detail(tmp_path):
    secret = b"e" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    remote_detail = "signature=private-signature key=private-key"
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                error={"code": "operation_failed", "message": remote_detail},
            )
        ),
    ) as server:
        with pytest.raises(OpsAgentRemoteError) as raised:
            _client(server.socket_path, key_path).call("host.memory", {})

    error = raised.value
    assert error.message == "Host operations agent returned an error"
    for rendered in (str(error), repr(error), repr(error.args), repr(error.__dict__)):
        assert "private" not in rendered
        assert "signature" not in rendered
        assert "key=" not in rendered


@pytest.mark.parametrize(
    "code",
    ["unknown-code", "operation failed", "operation_failed\nkey", "操作失败"],
)
def test_client_rejects_unrecognized_or_non_ascii_remote_error_codes(tmp_path, code):
    secret = b"w" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                error={"code": code, "message": "rejected"},
            )
        ),
    ) as server:
        with pytest.raises(OpsAgentProtocolError):
            _client(server.socket_path, key_path).call("host.memory", {})


def test_remote_error_constructor_rejects_unrecognized_code():
    with pytest.raises(OpsAgentProtocolError):
        OpsAgentRemoteError("private/path")


@pytest.mark.parametrize(
    "payload",
    [
        struct.pack(">I", MAX_FRAME_SIZE + 1),
        struct.pack(">I", 100) + b"{}",
        struct.pack(">I", 13) + b'{"x":1,"x":2}',
        struct.pack(">I", 9) + b'{"x":NaN}',
    ],
    ids=["oversize", "partial", "duplicate", "nonfinite"],
)
def test_client_rejects_invalid_frames(tmp_path, payload):
    secret = b"f" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    with _server(tmp_path, lambda _request: payload) as server:
        with pytest.raises(OpsAgentProtocolError):
            _client(server.socket_path, key_path).call("agent.health", {})


def test_client_enforces_absolute_read_deadline_for_trickled_response(tmp_path):
    secret = b"t" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                result={"status": "ok", "protocol_version": PROTOCOL_VERSION},
            )
        ),
        trickle_delay=0.02,
    ) as server:
        started = time.monotonic()
        with pytest.raises(OpsAgentUnavailable):
            _client(
                server.socket_path,
                key_path,
                read_timeout_seconds=0.08,
            ).health()
        elapsed = time.monotonic() - started

    assert elapsed < 0.4


def test_client_rejects_recursive_output_over_configured_limit(tmp_path):
    secret = b"o" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(request, secret, result={"nested": [{"output": "界" * 4_000}]})
        ),
    ) as server:
        with pytest.raises(OpsAgentProtocolError):
            _client(
                server.socket_path,
                key_path,
                output_limit_bytes=10_000,
            ).call("job.get", {"job_id": "example"})


def test_client_maps_lone_surrogate_output_to_protocol_error(tmp_path):
    secret = b"j" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    surrogate = "\ud800"
    with _server(
        tmp_path,
        lambda request: encode_frame(
            _response(
                request,
                secret,
                result={"nested": {surrogate: "safe", "output": surrogate}},
            )
        ),
    ) as server:
        with pytest.raises(OpsAgentProtocolError) as raised:
            _client(server.socket_path, key_path).call("job.get", {"job_id": "example"})

    assert str(raised.value) == "Host operations agent returned an invalid response"
    assert raised.value.__cause__ is None


def test_client_classifies_missing_key_and_socket_as_unavailable(tmp_path):
    client = _client(tmp_path / "missing.sock", tmp_path / "missing.key")
    with pytest.raises(OpsAgentUnavailable) as raised:
        client.health()
    assert str(tmp_path) not in str(raised.value)


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix domain sockets are unavailable on this platform",
)
def test_client_connects_to_real_unix_socket_path(tmp_path):
    secret = b"u" * 32
    key_path = tmp_path / "agent.key"
    key_path.write_bytes(secret)
    socket_path = tmp_path / "real.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                request = read_frame(connection)
                connection.sendall(
                    encode_frame(
                        _response(
                            request,
                            secret,
                            result={"status": "ok", "protocol_version": PROTOCOL_VERSION},
                        )
                    )
                )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        health = _client(socket_path, key_path).health()
    finally:
        listener.close()
        thread.join(timeout=1)

    assert health.status == "ok"
    assert not errors


def test_ops_agent_settings_defaults_and_ranges(tmp_path):
    settings = Settings(
        secret_key="test-secret-key-with-at-least-32-characters",
        admin_password="Test-password-1234",
    )
    assert settings.ops_agent_socket.as_posix() == "/run/dgx-spark-manager/ops-agent.sock"
    assert settings.ops_agent_key_file.as_posix() == "/run/secrets/ops-agent.key"

    common = {
        "secret_key": "test-secret-key-with-at-least-32-characters",
        "admin_password": "Test-password-1234",
    }
    for field, value in (
        ("ops_agent_connect_timeout_seconds", 0.49),
        ("ops_agent_connect_timeout_seconds", 30.1),
        ("ops_agent_read_timeout_seconds", 0.49),
        ("ops_agent_read_timeout_seconds", 120.1),
        ("ops_agent_output_limit_bytes", 9_999),
        ("ops_agent_output_limit_bytes", 10_000_001),
    ):
        with pytest.raises(ValidationError):
            Settings(**common, **{field: value})


def test_manager_image_packages_shared_ops_protocol():
    repository_root = Path(__file__).resolve().parents[2]
    project = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert packages == ["backend/app", "host_agent"]
    assert (repository_root / "host_agent/__init__.py").is_file()

    dockerfile = (repository_root / "Dockerfile").read_text(encoding="utf-8")
    copy_agent = dockerfile.index("COPY host_agent/ ./host_agent/")
    install_project = dockerfile.index("RUN python -m pip install .")
    assert copy_agent < install_project


def test_health_api_requires_admin(client):
    response = client.get("/api/ops-agent/health")
    assert response.status_code == 401


def test_health_api_reports_ok(authenticated_client, monkeypatch):
    class Health:
        status = "ok"
        protocol_version = PROTOCOL_VERSION

    monkeypatch.setattr(authenticated_client.app.state.ops_agent_client, "health", Health)
    response = authenticated_client.get("/api/ops-agent/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "protocol_version": PROTOCOL_VERSION}


@pytest.mark.parametrize(
    "exception,status,detail",
    [
        (
            OpsAgentUnavailable("secret path /run/private/key"),
            "unavailable",
            "Host operations agent is unavailable",
        ),
        (
            OpsAgentProtocolError("signature abc123"),
            "error",
            "Host operations agent returned an invalid response",
        ),
        (
            OpsAgentRemoteError("operation_failed"),
            "error",
            "Host operations agent rejected the health check",
        ),
    ],
)
def test_health_api_reports_safe_bounded_errors(
    authenticated_client,
    monkeypatch,
    exception,
    status,
    detail,
):
    def fail():
        raise exception

    monkeypatch.setattr(authenticated_client.app.state.ops_agent_client, "health", fail)
    response = authenticated_client.get("/api/ops-agent/health")
    assert response.status_code == 200
    assert response.json() == {"status": status, "detail": detail}
    assert len(response.content) < 256
    assert "private" not in response.text
    assert "abc123" not in response.text


def test_manager_health_starts_without_ops_agent_files(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
