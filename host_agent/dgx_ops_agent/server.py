"""Authenticated dispatch and bounded Unix socket serving for the host agent."""

from __future__ import annotations

import argparse
import errno
import importlib
import os
import re
import socket
import stat
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .policy import READ_ONLY_ACTIONS, PolicyError, validate_action
from .protocol import (
    PROTOCOL_VERSION,
    NonceCache,
    ProtocolError,
    read_frame,
    sign_message,
    verify_message,
    write_frame,
)
from .runner import TERMINAL_STATUSES, JobRunner

_READ_COMPLETION_GRACE = 1.0
_READ_POLL_INTERVAL = 0.05
_DEFAULT_KEY_PATH = "/etc/dgx-spark-manager/ops-agent.key"
_DEFAULT_KEY_GROUP = "dgx-spark-ops"
_MAX_KEY_FILE_SIZE = 65
_KEY_GROUP_PATTERN = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_IS_POSIX = os.name == "posix"
_HEX_BYTES = frozenset(b"0123456789abcdefABCDEF")


class _OperationTimeout(RuntimeError):
    def __init__(self, job_id: str) -> None:
        super().__init__("operation did not finish before the server deadline")
        self.job_id = job_id


class _ApprovalRequired(ValueError):
    pass


class _UnknownAction(ValueError):
    pass


class _DeadlineReader:
    def __init__(self, connection: socket.socket, timeout: float) -> None:
        self._connection = connection
        self._deadline = time.monotonic() + timeout

    def recv(self, size: int) -> bytes:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request read deadline exceeded")
        self._connection.settimeout(remaining)
        return self._connection.recv(size)


class AgentServer:
    """Verify, dispatch, and sign one host-agent request."""

    def __init__(
        self,
        secret: bytes,
        runner: JobRunner,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        nonces: NonceCache | None = None,
    ) -> None:
        sign_message({}, secret)
        self._secret = secret
        self._runner = runner
        self._clock = clock
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._nonces = NonceCache() if nonces is None else nonces

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        now = int(self._clock())
        try:
            request = verify_message(
                message,
                self._secret,
                now=now,
                nonces=self._nonces,
            )
        except ProtocolError as exc:
            code, detail = self._protocol_error(exc)
            return self._response(None, error=(code, detail), now=now)

        try:
            result = self._dispatch(request)
        except _UnknownAction:
            return self._response(
                request["request_id"],
                error=("unknown_action", "unknown action"),
                now=now,
            )
        except _ApprovalRequired:
            return self._response(
                request["request_id"],
                error=("approval_required", "shell execution requires approval"),
                now=now,
            )
        except PolicyError:
            return self._response(
                request["request_id"],
                error=("policy_rejected", "request rejected by policy"),
                now=now,
            )
        except _OperationTimeout as exc:
            return self._response(
                request["request_id"],
                result={"job_id": exc.job_id},
                error=(
                    "operation_timeout",
                    "operation did not finish before the server deadline",
                ),
                now=now,
            )
        except KeyError:
            return self._response(
                request["request_id"],
                error=("job_not_found", "job not found"),
                now=now,
            )
        except ValueError:
            return self._response(
                request["request_id"],
                error=("invalid_parameters", "invalid parameters"),
                now=now,
            )
        except Exception:
            return self._response(
                request["request_id"],
                error=("operation_failed", "operation failed"),
                now=now,
            )
        return self._response(request["request_id"], result=result, now=now)

    def serve_connection(
        self,
        connection: socket.socket,
        *,
        read_timeout: float,
    ) -> None:
        """Read and answer exactly one frame, then close the connection."""

        with connection:
            try:
                message = read_frame(_DeadlineReader(connection, read_timeout))
            except ProtocolError as exc:
                if isinstance(exc.__cause__, TimeoutError):
                    error = ("read_timeout", "request read timed out")
                elif str(exc) == "frame too large":
                    error = ("frame_too_large", "frame too large")
                else:
                    error = ("invalid_frame", "invalid request frame")
                response = self._response(
                    None,
                    error=error,
                    now=int(self._clock()),
                )
            else:
                response = self.handle(message)
            connection.settimeout(read_timeout)
            try:
                write_frame(connection, response)
            except ProtocolError:
                return

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request["action"]
        parameters = request["parameters"]
        if action == "agent.health":
            self._require_keys(parameters, frozenset())
            return {"status": "ok", "protocol_version": PROTOCOL_VERSION}
        if action == "job.get":
            keys = frozenset(parameters)
            if keys not in (
                frozenset({"job_id"}),
                frozenset({"job_id", "offset"}),
            ):
                raise ValueError("invalid parameters")
            self._validate_job_id(parameters["job_id"])
            offset = parameters.get("offset")
            if "offset" in parameters and (type(offset) is not int or offset < 0):
                raise ValueError("invalid parameters")
            return self._runner.get(parameters["job_id"], offset=offset).as_dict()
        if action == "job.cancel":
            self._require_keys(parameters, frozenset({"job_id"}))
            self._validate_job_id(parameters["job_id"])
            return self._runner.cancel(parameters["job_id"]).as_dict()

        if action not in READ_ONLY_ACTIONS and action != "shell.execute":
            raise _UnknownAction("unknown action")
        if action == "shell.execute" and "approval" not in request:
            raise _ApprovalRequired("shell execution requires approval")

        validated = validate_action(
            action,
            parameters,
            request.get("approval"),
            request_timestamp=request["timestamp"],
        )
        job = self._runner.start(validated)
        if not validated.read_only:
            return job.as_dict()
        return self._await_read_job(job.as_dict(), validated.timeout)

    def _await_read_job(
        self,
        initial: dict[str, Any],
        action_timeout: int,
    ) -> dict[str, Any]:
        if initial["status"] in TERMINAL_STATUSES:
            return initial
        termination_grace = getattr(self._runner, "termination_grace", 5.0)
        if not isinstance(termination_grace, (int, float)):
            termination_grace = 5.0
        termination_grace = min(max(float(termination_grace), 0.0), 5.0)
        deadline = (
            self._monotonic()
            + action_timeout
            + termination_grace
            + _READ_COMPLETION_GRACE
        )
        job_id = initial["job_id"]
        while True:
            current = self._runner.get(job_id).as_dict()
            if current["status"] in TERMINAL_STATUSES:
                return current
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                cancelled = self._runner.cancel(job_id).as_dict()
                if cancelled["status"] in TERMINAL_STATUSES:
                    return cancelled
                raise _OperationTimeout(job_id)
            self._sleeper(min(_READ_POLL_INTERVAL, remaining))

    @staticmethod
    def _require_keys(parameters: dict[str, Any], expected: frozenset[str]) -> None:
        if frozenset(parameters) != expected:
            raise ValueError("invalid parameters")

    @staticmethod
    def _validate_job_id(job_id: Any) -> None:
        if not isinstance(job_id, str) or len(job_id) != 36:
            raise ValueError("invalid parameters")
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError("invalid parameters")
        except (ValueError, AttributeError):
            raise ValueError("invalid parameters") from None

    @staticmethod
    def _protocol_error(exc: ProtocolError) -> tuple[str, str]:
        message = str(exc)
        if message == "replayed nonce":
            return "replay_detected", "request already processed"
        if message in {"expired request", "request timestamp is too far in the future"}:
            return "expired_request", "request timestamp is outside the allowed window"
        if message == "invalid signature":
            return "authentication_failed", "request authentication failed"
        return "invalid_request", "invalid request"

    def _response(
        self,
        request_id: str | None,
        *,
        result: dict[str, Any] | None = None,
        error: tuple[str, str] | None = None,
        now: int,
    ) -> dict[str, Any]:
        error_payload = (
            None if error is None else {"code": error[0], "message": error[1]}
        )
        return sign_message(
            {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": error is None,
                "result": result,
                "error": error_payload,
                "timestamp": now,
            },
            self._secret,
        )


class UnixSocketAgentServer:
    """Serve AgentServer requests with a strictly bounded worker pool."""

    def __init__(
        self,
        agent: AgentServer,
        *,
        socket_path: str | os.PathLike[str] | None = None,
        listener: socket.socket | None = None,
        max_connections: int = 4,
        read_timeout: float = 2.0,
    ) -> None:
        if (socket_path is None) == (listener is None):
            raise ValueError("provide exactly one listener source")
        if type(max_connections) is not int or not 1 <= max_connections <= 64:
            raise ValueError("invalid max_connections")
        if (
            isinstance(read_timeout, bool)
            or not isinstance(read_timeout, (int, float))
            or not 0.01 <= read_timeout <= 30
        ):
            raise ValueError("invalid read_timeout")
        if listener is not None:
            unix_family = getattr(socket, "AF_UNIX", None)
            try:
                valid_listener = (
                    isinstance(listener, socket.socket)
                    and unix_family is not None
                    and listener.family == unix_family
                    and listener.type & socket.SOCK_STREAM == socket.SOCK_STREAM
                    and listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
                )
            except OSError:
                valid_listener = False
            if not valid_listener:
                raise ValueError("invalid listener")
        self._agent = agent
        self._max_connections = max_connections
        self._read_timeout = float(read_timeout)
        self._closed = threading.Event()
        self._serve_lock = threading.Lock()
        self._socket_path: Path | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._parent_identity: tuple[int, int] | None = None
        if listener is None:
            (
                self._listener,
                self._parent_identity,
                self._socket_identity,
            ) = self._bind(Path(socket_path))
            self._socket_path = Path(socket_path)
        else:
            self._listener = listener

    @classmethod
    def from_systemd(
        cls,
        agent: AgentServer,
        *,
        descriptor: int = 3,
        environment: dict[str, str] | None = None,
        max_connections: int = 4,
        read_timeout: float = 2.0,
    ) -> UnixSocketAgentServer:
        """Adopt systemd's first activated listener (file descriptor 3)."""

        env = os.environ if environment is None else environment
        if env.get("LISTEN_PID") != str(os.getpid()) or env.get("LISTEN_FDS") != "1":
            raise ValueError("invalid systemd socket activation")
        if type(descriptor) is not int or descriptor < 0:
            raise ValueError("invalid systemd socket activation")
        try:
            listener = socket.socket(fileno=os.dup(descriptor))
        except OSError:
            raise ValueError("invalid systemd socket activation") from None
        try:
            if listener.family != socket.AF_UNIX or listener.type & socket.SOCK_STREAM == 0:
                raise ValueError("invalid systemd socket activation")
            return cls(
                agent,
                listener=listener,
                max_connections=max_connections,
                read_timeout=read_timeout,
            )
        except Exception:
            listener.close()
            raise

    def serve_forever(self, *, stop_event: threading.Event | None = None) -> None:
        stop = self._closed if stop_event is None else stop_event
        if not self._serve_lock.acquire(blocking=False):
            raise RuntimeError("server is already running")
        slots = threading.BoundedSemaphore(self._max_connections)
        executor = ThreadPoolExecutor(
            max_workers=self._max_connections,
            thread_name_prefix="dgx-ops-connection",
        )
        try:
            self._listener.settimeout(0.1)
            while not stop.is_set() and not self._closed.is_set():
                if not slots.acquire(timeout=0.1):
                    continue
                try:
                    connection, _address = self._listener.accept()
                except TimeoutError:
                    slots.release()
                    continue
                except OSError:
                    slots.release()
                    if self._closed.is_set() or stop.is_set():
                        break
                    raise
                executor.submit(self._serve_one, connection, slots)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            self._serve_lock.release()

    def _serve_one(
        self,
        connection: socket.socket,
        slots: threading.BoundedSemaphore,
    ) -> None:
        try:
            self._agent.serve_connection(
                connection,
                read_timeout=self._read_timeout,
            )
        finally:
            slots.release()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._listener.close()
        finally:
            if (
                self._socket_path is not None
                and self._socket_identity is not None
                and self._parent_identity is not None
            ):
                self._unlink_if_same_socket(
                    self._socket_path,
                    self._socket_identity,
                    parent_identity=self._parent_identity,
                )

    @staticmethod
    def _socket_matches(path: Path, identity: tuple[int, int]) -> bool:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
        return stat.S_ISSOCK(metadata.st_mode) and (
            metadata.st_dev,
            metadata.st_ino,
        ) == identity

    @staticmethod
    def _unlink_if_same_socket(
        path: Path,
        identity: tuple[int, int],
        *,
        parent_identity: tuple[int, int] | None = None,
    ) -> bool:
        if parent_identity is not None and not UnixSocketAgentServer._path_chain_matches(
            path,
            parent_identity,
        ):
            return False
        if not UnixSocketAgentServer._socket_matches(path, identity):
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _chmod_if_same_socket(
        path: Path,
        identity: tuple[int, int],
        mode: int,
        *,
        parent_identity: tuple[int, int] | None = None,
    ) -> bool:
        if parent_identity is not None and not UnixSocketAgentServer._path_chain_matches(
            path,
            parent_identity,
        ):
            return False
        if not UnixSocketAgentServer._socket_matches(path, identity):
            return False
        try:
            os.chmod(path, mode, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return UnixSocketAgentServer._socket_matches(path, identity)

    @staticmethod
    def _parent_metadata_is_secure(metadata: Any, *, expected_uid: int) -> bool:
        return (
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == expected_uid
            and not metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        )

    @staticmethod
    def _path_chain_metadata_is_secure(
        chain: list[Any],
        *,
        expected_uid: int,
    ) -> bool:
        if not chain:
            return False
        trusted_owners = {0, expected_uid}
        for metadata in chain:
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in trusted_owners
            ):
                return False

        direct_parent = chain[-1]
        if (
            direct_parent.st_uid != expected_uid
            or direct_parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return False

        for index, metadata in enumerate(chain[:-1]):
            if not metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                continue
            if (
                not metadata.st_mode & stat.S_ISVTX
                or chain[index + 1].st_uid not in trusted_owners
            ):
                return False
        return True

    @staticmethod
    def _validate_socket_path_chain(path: Path) -> tuple[int, int]:
        if not path.is_absolute():
            raise ValueError("invalid socket path chain")
        parent = path.parent
        if not _IS_POSIX:
            UnixSocketAgentServer._validate_socket_parent(parent, create=False)
            metadata = parent.lstat()
            return metadata.st_dev, metadata.st_ino

        root = Path(parent.anchor)
        current = root
        chain: list[Any] = []
        try:
            for index, component in enumerate(parent.parts):
                if index > 0:
                    current = current / component
                metadata = current.lstat()
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError("invalid socket path chain")
                chain.append(metadata)
        except OSError:
            raise ValueError("invalid socket path chain") from None

        if not UnixSocketAgentServer._path_chain_metadata_is_secure(
            chain,
            expected_uid=_effective_uid(),
        ):
            raise ValueError("invalid socket path chain")
        direct_parent = chain[-1]
        return direct_parent.st_dev, direct_parent.st_ino

    @staticmethod
    def _path_chain_matches(path: Path, identity: tuple[int, int]) -> bool:
        try:
            return UnixSocketAgentServer._validate_socket_path_chain(path) == identity
        except ValueError:
            return False

    @staticmethod
    def _validate_socket_parent(parent: Path, *, create: bool) -> None:
        try:
            created = not os.path.lexists(parent)
            if created:
                if not create:
                    raise ValueError("invalid socket parent")
                ancestor = parent.parent
                if not os.path.lexists(ancestor):
                    raise ValueError("invalid socket path chain")
                if _IS_POSIX:
                    UnixSocketAgentServer._validate_socket_path_chain(
                        ancestor / ".dgx-ops-parent-check"
                    )
                else:
                    UnixSocketAgentServer._validate_socket_parent(
                        ancestor,
                        create=False,
                    )
                parent.mkdir(parents=False, mode=0o700)
                before = parent.lstat()
                if not stat.S_ISDIR(before.st_mode):
                    raise ValueError("invalid socket parent")
                if _IS_POSIX:
                    os.chmod(parent, 0o700, follow_symlinks=False)
                    after = parent.lstat()
                    if (before.st_dev, before.st_ino) != (
                        after.st_dev,
                        after.st_ino,
                    ):
                        raise ValueError("invalid socket parent")
            metadata = parent.lstat()
            expected_uid = _effective_uid() if _IS_POSIX else metadata.st_uid
            if not UnixSocketAgentServer._parent_metadata_is_secure(
                metadata,
                expected_uid=expected_uid,
            ):
                raise ValueError("invalid socket parent")
        except (OSError, RuntimeError):
            raise ValueError("invalid socket parent") from None

    @staticmethod
    def _bind(
        path: Path,
    ) -> tuple[socket.socket, tuple[int, int], tuple[int, int]]:
        if not path.is_absolute():
            raise ValueError("socket_path must be absolute")
        UnixSocketAgentServer._validate_socket_parent(path.parent, create=True)
        parent_identity = UnixSocketAgentServer._validate_socket_path_chain(path)
        if os.path.lexists(path):
            metadata = path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode):
                raise ValueError("socket_path exists and is not a socket")
            if _IS_POSIX and metadata.st_uid != _effective_uid():
                raise ValueError("socket_path is not owned by this user")
            stale_identity = (metadata.st_dev, metadata.st_ino)
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.1)
                probe.connect(os.fspath(path))
            except OSError as exc:
                if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                    raise ValueError("cannot verify existing socket_path") from None
            else:
                raise ValueError("socket_path is an active socket")
            finally:
                probe.close()
            if not UnixSocketAgentServer._unlink_if_same_socket(
                path,
                stale_identity,
                parent_identity=parent_identity,
            ):
                raise ValueError("socket_path changed while being verified")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound_identity: tuple[int, int] | None = None
        try:
            listener.bind(os.fspath(path))
            metadata = path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode):
                raise ValueError("bound socket_path is invalid")
            bound_identity = (metadata.st_dev, metadata.st_ino)
            if _IS_POSIX and not UnixSocketAgentServer._chmod_if_same_socket(
                path,
                bound_identity,
                0o660,
                parent_identity=parent_identity,
            ):
                raise ValueError("socket_path changed while binding")
            listener.listen(16)
            if not UnixSocketAgentServer._socket_matches(path, bound_identity):
                raise ValueError("socket_path changed while binding")
        except Exception:
            listener.close()
            if bound_identity is not None:
                UnixSocketAgentServer._unlink_if_same_socket(
                    path,
                    bound_identity,
                    parent_identity=parent_identity,
                )
            raise
        return listener, parent_identity, bound_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DGX Spark host operations agent")
    parser.add_argument("--systemd", action="store_true")
    parser.add_argument("--socket", default="/run/dgx-spark-manager/ops-agent.sock")
    parser.add_argument("--key-file", default=_DEFAULT_KEY_PATH)
    parser.add_argument("--key-group", default=_DEFAULT_KEY_GROUP)
    parser.add_argument("--job-dir", default="/var/lib/dgx-spark-ops-agent/jobs")
    parser.add_argument("--max-connections", type=int, default=4)
    parser.add_argument("--read-timeout", type=float, default=2.0)
    options = parser.parse_args(argv)

    expected_gid = _resolve_key_group(options.key_group)
    secret = _read_secret_file(options.key_file, expected_gid=expected_gid)
    runner = JobRunner(options.job_dir)
    agent = AgentServer(secret, runner)
    if options.systemd:
        transport = UnixSocketAgentServer.from_systemd(
            agent,
            descriptor=3,
            max_connections=options.max_connections,
            read_timeout=options.read_timeout,
        )
    else:
        transport = UnixSocketAgentServer(
            agent,
            socket_path=options.socket,
            max_connections=options.max_connections,
            read_timeout=options.read_timeout,
        )
    try:
        transport.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()
    return 0


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise ValueError("effective user is unavailable")
    return int(getter())


def _resolve_key_group(name: str) -> int | None:
    if not isinstance(name, str) or _KEY_GROUP_PATTERN.fullmatch(name) is None:
        raise ValueError("invalid key group")
    if not _IS_POSIX:
        return None
    group_module = importlib.import_module("grp")
    try:
        group = group_module.getgrnam(name)
    except KeyError:
        raise ValueError("invalid key group") from None
    group_id = group.gr_gid
    if type(group_id) is not int or group_id < 0:
        raise ValueError("invalid key group")
    return group_id


def _validate_key_metadata(metadata: os.stat_result, expected_gid: int | None) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("invalid key file")
    if not _IS_POSIX:
        return
    if expected_gid is not None and (type(expected_gid) is not int or expected_gid < 0):
        raise ValueError("invalid key file")
    if metadata.st_uid != _effective_uid():
        raise ValueError("invalid key file")
    mode = metadata.st_mode
    allowed = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
    special = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
    if (
        not mode & stat.S_IRUSR
        or mode & special
        or stat.S_IMODE(mode) & ~allowed
    ):
        raise ValueError("invalid key file")
    if mode & stat.S_IRGRP and (
        expected_gid is None or metadata.st_gid != expected_gid
    ):
        raise ValueError("invalid key file")


def _decode_secret_file(value: bytes) -> bytes:
    if len(value) == 32:
        return value
    if len(value) == 65 and value.endswith(b"\n"):
        value = value[:-1]
    if len(value) != 64 or any(byte not in _HEX_BYTES for byte in value):
        raise ValueError("invalid key file")
    return bytes.fromhex(value.decode("ascii"))


def _read_secret_file(
    path: str | os.PathLike[str],
    *,
    expected_gid: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("invalid key file") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            metadata.st_size <= 0
            or metadata.st_size > _MAX_KEY_FILE_SIZE
        ):
            raise ValueError("invalid key file")
        _validate_key_metadata(metadata, expected_gid)
        chunks = bytearray()
        while len(chunks) <= _MAX_KEY_FILE_SIZE:
            chunk = os.read(
                descriptor,
                _MAX_KEY_FILE_SIZE + 1 - len(chunks),
            )
            if not chunk:
                break
            chunks.extend(chunk)
        value = bytes(chunks)
    finally:
        os.close(descriptor)
    if len(value) > _MAX_KEY_FILE_SIZE:
        raise ValueError("invalid key file")
    return _decode_secret_file(value)


__all__ = ["AgentServer", "UnixSocketAgentServer", "main"]


if __name__ == "__main__":  # pragma: no cover - exercised by systemd integration
    raise SystemExit(main())
