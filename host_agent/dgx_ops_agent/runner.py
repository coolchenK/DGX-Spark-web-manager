"""Thread-safe bounded subprocess jobs for the host operations agent."""

from __future__ import annotations

import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .launcher import (
    GO,
    MAX_ARGUMENT_BYTES,
    MAX_ARGUMENTS,
    serialize_request,
    write_request,
)
from .policy import ValidatedAction
from .redaction import StreamingRedactor, redact_text

SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PROC_ROOT = Path("/proc")
_HAS_PROCESS_GROUPS = os.name == "posix"
_IS_LINUX = sys.platform.startswith("linux")
_IS_POSIX_FILESYSTEM = os.name == "posix"
_RECOVERY_ENVIRONMENT_KEY = "DGX_OPS_RECOVERY_TOKEN"
DEFAULT_OUTPUT_LIMIT = 64 * 1024
MAX_OUTPUT_LIMIT = 16 * 1024 * 1024
MAX_ERROR_BYTES = 512
TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "timed_out", "cancelled"}
)
_STATUSES = frozenset({"queued", "running", *TERMINAL_STATUSES})
_ALLOWED_ENVIRONMENT = frozenset(
    {
        "COLORTERM",
        "DEBIAN_FRONTEND",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_COLOR",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "TERM",
        "TZ",
    }
)
_RECOVERY_TOKEN_PLACEHOLDER = "x" * 43


@dataclass(frozen=True, slots=True)
class Job:
    """Immutable public snapshot of a job."""

    id: str
    status: str
    output: str
    output_offset: int
    truncated_before: int
    exit_code: int | None
    started: float | None
    finished: float | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "output": self.output,
            "output_offset": self.output_offset,
            "truncated_before": self.truncated_before,
            "exit_code": self.exit_code,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _ProcessIdentity:
    pid: int
    process_group: int | None
    session_id: int | None
    start_time_ticks: int | None
    boot_id: str | None
    recovery_token: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_group": self.process_group,
            "session_id": self.session_id,
            "start_time_ticks": self.start_time_ticks,
            "boot_id": self.boot_id,
            "recovery_token": self.recovery_token,
        }


@dataclass(slots=True)
class _JobState:
    id: str
    status: str = "queued"
    output: bytearray = field(default_factory=bytearray)
    output_offset: int = 0
    truncated_before: int = 0
    exit_code: int | None = None
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    process: subprocess.Popen[bytes] | None = None
    process_group: int | None = None
    process_identity: _ProcessIdentity | None = None
    launch_phase: str | None = None
    recovery_token: str | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)


class _OwnedDescriptor:
    def __init__(self, descriptor: int) -> None:
        self._descriptor: int | None = descriptor

    def take(self) -> int:
        descriptor = self._descriptor
        if descriptor is None:
            raise RuntimeError("descriptor ownership already transferred")
        self._descriptor = None
        return descriptor

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            os.close(descriptor)
        except OSError:
            pass


class JobRunner:
    """Run policy-bound commands and retain only bounded redacted output."""

    def __init__(
        self,
        job_dir: str | os.PathLike[str],
        *,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        termination_grace: float = 5.0,
    ) -> None:
        if (
            type(output_limit) is not int
            or not 1 <= output_limit <= MAX_OUTPUT_LIMIT
        ):
            raise ValueError("invalid output_limit")
        if (
            isinstance(termination_grace, bool)
            or not isinstance(termination_grace, (int, float))
            or not 0 <= termination_grace <= 5
        ):
            raise ValueError("invalid termination_grace")
        self.job_dir = Path(job_dir)
        if self.job_dir.is_symlink():
            raise ValueError("invalid job_dir")
        self.job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._secure_job_directory()
        self.output_limit = output_limit
        self.termination_grace = float(termination_grace)
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobState] = {}
        self._load_jobs()

    def start(
        self,
        invocation: ValidatedAction | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        timeout: int | float | None = None,
        environment: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    ) -> Job:
        """Queue a validated action or an explicit argv invocation."""

        argv, resolved_cwd, resolved_timeout, resolved_environment = self._invocation(
            invocation,
            cwd=cwd,
            timeout=timeout,
            environment=environment,
        )
        job_id = str(uuid.uuid4())
        state = _JobState(id=job_id)
        with self._lock:
            self._jobs[job_id] = state
            try:
                self._persist(state)
            except Exception:
                del self._jobs[job_id]
                raise
            snapshot = self._snapshot(state)

        worker = threading.Thread(
            target=self._run,
            args=(state, argv, resolved_cwd, resolved_timeout, resolved_environment),
            name=f"dgx-ops-job-{job_id}",
            daemon=True,
        )
        worker.start()
        return snapshot

    def get(self, job_id: str, *, offset: int | None = None) -> Job:
        """Return retained output at or after an absolute redacted byte offset."""

        state = self._lookup(job_id)
        with self._lock:
            if offset is None:
                requested = state.truncated_before
            elif type(offset) is not int or not 0 <= offset <= state.output_offset:
                raise ValueError("invalid offset")
            else:
                requested = max(offset, state.truncated_before)

            index = requested - state.truncated_before
            while index < len(state.output) and state.output[index] & 0xC0 == 0x80:
                index += 1
                requested += 1
            output = bytes(state.output[index:]).decode("utf-8")
            return self._snapshot(
                state,
                output=output,
                truncated_before=requested,
            )

    def cancel(self, job_id: str) -> Job:
        """Request cancellation once; repeated calls preserve the terminal result."""

        state = self._lookup(job_id)
        with self._lock:
            if state.status in TERMINAL_STATUSES:
                return self._snapshot(state)
            process = state.process
            group_active = (
                state.process_group is not None
                and _HAS_PROCESS_GROUPS
                and self._process_group_exists(state.process_group)
            )
            if process is None or process.poll() is None or group_active:
                state.cancel_requested.set()
            return self._snapshot(state)

    def _invocation(
        self,
        invocation: ValidatedAction | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None,
        timeout: int | float | None,
        environment: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> tuple[tuple[str, ...], str | None, float, dict[str, str]]:
        if isinstance(invocation, ValidatedAction):
            if cwd is not None or timeout is not None or environment:
                raise TypeError("validated action does not accept invocation overrides")
            argv = invocation.argv
            resolved_cwd = invocation.cwd
            resolved_timeout: int | float = invocation.timeout
            environment_items: Mapping[str, str] | Iterable[tuple[str, str]] = (
                invocation.environment
            )
        else:
            argv = self._validate_argv(invocation)
            resolved_cwd = None if cwd is None else os.fspath(cwd)
            resolved_timeout = timeout if timeout is not None else 30
            environment_items = environment

        argv = self._validate_argv(argv)
        if (
            isinstance(resolved_timeout, bool)
            or not isinstance(resolved_timeout, (int, float))
            or not 0 < resolved_timeout <= 3_600
        ):
            raise ValueError("invalid timeout")
        if resolved_cwd is not None and (
            not isinstance(resolved_cwd, str) or not resolved_cwd or "\x00" in resolved_cwd
        ):
            raise ValueError("invalid cwd")
        resolved_environment = safe_environment(environment_items)
        serialize_request(
            argv=argv,
            cwd=resolved_cwd,
            environment={
                **resolved_environment,
                _RECOVERY_ENVIRONMENT_KEY: _RECOVERY_TOKEN_PLACEHOLDER,
            },
        )
        return argv, resolved_cwd, float(resolved_timeout), resolved_environment

    @staticmethod
    def _validate_argv(invocation: Sequence[str]) -> tuple[str, ...]:
        if isinstance(invocation, (str, bytes)) or not isinstance(invocation, Sequence):
            raise ValueError("invalid argv")
        argv = tuple(invocation)
        if not argv or len(argv) > MAX_ARGUMENTS:
            raise ValueError("invalid argv")
        if any(not isinstance(item, str) or not item or "\x00" in item for item in argv):
            raise ValueError("invalid argv")
        if sum(len(item.encode("utf-8")) for item in argv) > MAX_ARGUMENT_BYTES:
            raise ValueError("invalid argv")
        return argv

    def _spawn_process(
        self,
        argv: tuple[str, ...],
        cwd: str | None,
        environment: dict[str, str],
        recovery_token: str,
    ) -> tuple[subprocess.Popen[bytes], int | None]:
        common_options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": False,
            "shell": False,
        }
        if not _HAS_PROCESS_GROUPS:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                **common_options,
            )
            return process, None

        with ExitStack() as descriptor_stack:
            request_read, request_write = os.pipe()
            request_read_owner = _OwnedDescriptor(request_read)
            request_write_owner = _OwnedDescriptor(request_write)
            descriptor_stack.callback(request_read_owner.close)
            descriptor_stack.callback(request_write_owner.close)

            gate_read, gate_write = os.pipe()
            gate_read_owner = _OwnedDescriptor(gate_read)
            gate_write_owner = _OwnedDescriptor(gate_write)
            descriptor_stack.callback(gate_read_owner.close)
            descriptor_stack.callback(gate_write_owner.close)

            launcher_path = Path(__file__).with_name("launcher.py").resolve(strict=True)
            launcher_argv = (
                sys.executable,
                str(launcher_path),
                str(request_read),
                str(gate_read),
            )
            process = subprocess.Popen(
                launcher_argv,
                cwd=None,
                env={
                    "PATH": SAFE_PATH,
                    _RECOVERY_ENVIRONMENT_KEY: recovery_token,
                },
                pass_fds=(request_read, gate_read),
                start_new_session=True,
                **common_options,
            )

            self._close_descriptor(request_read_owner.take())
            self._close_descriptor(gate_read_owner.take())
            try:
                write_request(
                    request_write_owner.take(),
                    argv=argv,
                    cwd=cwd,
                    environment=environment,
                )
            except Exception:
                self._terminate(process, process.pid)
                try:
                    process.wait(timeout=max(self.termination_grace, 0.1))
                except subprocess.TimeoutExpired:
                    self._kill(process, process.pid)
                    process.wait()
                raise
            return process, gate_write_owner.take()

    @staticmethod
    def _close_descriptor(descriptor: int | None) -> None:
        if descriptor is None:
            return
        try:
            os.close(descriptor)
        except OSError:
            pass

    def _run(
        self,
        state: _JobState,
        argv: tuple[str, ...],
        cwd: str | None,
        timeout: float,
        environment: dict[str, str],
    ) -> None:
        with self._lock:
            if state.cancel_requested.is_set():
                self._finish(state, "cancelled", None)
                return
            state.started = time.time()

        recovery_token = secrets.token_urlsafe(32)
        execution_environment = {
            **environment,
            _RECOVERY_ENVIRONMENT_KEY: recovery_token,
        }
        if _HAS_PROCESS_GROUPS:
            preparation_error: OSError | None = None
            with self._lock:
                state.launch_phase = "prepared"
                state.recovery_token = recovery_token
                try:
                    self._persist(state)
                except OSError as exc:
                    preparation_error = exc
            if preparation_error is not None:
                with self._lock:
                    state.status = "failed"
                    state.finished = time.time()
                    state.error = self._bounded_error(preparation_error)
                    try:
                        self._persist(state)
                    except OSError:
                        pass
                return
        try:
            process, gate_descriptor = self._spawn_process(
                argv,
                cwd,
                execution_environment,
                recovery_token,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            with self._lock:
                self._finish(state, "failed", None, error=self._bounded_error(exc))
            return

        process_group = process.pid if _HAS_PROCESS_GROUPS else None
        try:
            identity = self._capture_process_identity(process.pid, recovery_token)
        except (OSError, ValueError) as exc:
            self._close_descriptor(gate_descriptor)
            self._terminate(process, process_group)
            try:
                exit_code = process.wait(timeout=max(self.termination_grace, 0.1))
            except subprocess.TimeoutExpired:
                self._kill(process, process_group)
                exit_code = process.wait()
            with self._lock:
                state.process = process
                state.process_group = process_group
                self._finish(
                    state,
                    "failed",
                    exit_code,
                    error=self._bounded_error(exc),
                )
            return

        persistence_error: OSError | None = None
        with self._lock:
            state.process = process
            state.process_group = process_group
            state.process_identity = identity
            state.status = "running"
            if _HAS_PROCESS_GROUPS:
                state.launch_phase = "ready"
            try:
                self._persist(state)
            except OSError as exc:
                persistence_error = exc
        if persistence_error is not None:
            self._close_descriptor(gate_descriptor)
            self._terminate(process, process_group)
            try:
                exit_code = process.wait(timeout=max(self.termination_grace, 0.1))
            except subprocess.TimeoutExpired:
                self._kill(process, process_group)
                exit_code = process.wait()
            with self._lock:
                try:
                    self._finish(
                        state,
                        "failed",
                        exit_code,
                        error=self._bounded_error(persistence_error),
                    )
                except OSError:
                    pass
            return

        gate_error: OSError | None = None
        cancelled_before_go = False
        if gate_descriptor is not None:
            with self._lock:
                if state.cancel_requested.is_set():
                    self._close_descriptor(gate_descriptor)
                    gate_descriptor = None
                    cancelled_before_go = True
                else:
                    try:
                        if os.write(gate_descriptor, GO) != len(GO):
                            raise OSError("launcher gate write failed")
                    except OSError as exc:
                        gate_error = exc
                    finally:
                        self._close_descriptor(gate_descriptor)
                        gate_descriptor = None

        if cancelled_before_go:
            try:
                exit_code = process.wait(timeout=max(self.termination_grace, 0.1))
            except subprocess.TimeoutExpired:
                self._terminate(process, process_group)
                try:
                    exit_code = process.wait(timeout=max(self.termination_grace, 0.1))
                except subprocess.TimeoutExpired:
                    self._kill(process, process_group)
                    exit_code = process.wait()
            with self._lock:
                self._finish(state, "cancelled", exit_code)
            return

        if gate_error is not None:
            self._terminate(process, process_group)
            try:
                exit_code = process.wait(timeout=max(self.termination_grace, 0.1))
            except subprocess.TimeoutExpired:
                self._kill(process, process_group)
                exit_code = process.wait()
            with self._lock:
                self._finish(
                    state,
                    "failed",
                    exit_code,
                    error=self._bounded_error(gate_error),
                )
            return

        redactor = StreamingRedactor(secret_values=(recovery_token,))
        reader_error: list[Exception] = []
        reader = threading.Thread(
            target=self._read_output,
            args=(state, process, redactor, reader_error),
            name=f"dgx-ops-output-{state.id}",
            daemon=True,
        )
        reader.start()

        deadline = time.monotonic() + timeout
        outcome: str | None = None
        while True:
            leader_active = process.poll() is None
            group_active = (
                state.process_group is not None
                and self._process_group_exists(state.process_group)
            )
            pipe_active = reader.is_alive()
            if not leader_active and not group_active and not pipe_active:
                break
            if state.cancel_requested.wait(timeout=0.01):
                outcome = "cancelled"
                self._terminate(process, state.process_group)
                break
            if time.monotonic() >= deadline:
                outcome = "timed_out"
                self._terminate(process, state.process_group)
                break
        try:
            exit_code = process.wait(timeout=max(self.termination_grace, 0.1))
        except subprocess.TimeoutExpired:
            self._kill(process, state.process_group)
            exit_code = process.wait()

        reader.join(timeout=max(self.termination_grace, 1.0))
        if reader.is_alive() and process.stdout is not None:
            process.stdout.close()
            reader.join(timeout=1.0)

        if outcome is None:
            outcome = "succeeded" if exit_code == 0 else "failed"
        error = self._bounded_error(reader_error[0]) if reader_error else None
        if error is not None and outcome == "succeeded":
            outcome = "failed"
        with self._lock:
            self._finish(state, outcome, exit_code, error=error)

    def _read_output(
        self,
        state: _JobState,
        process: subprocess.Popen[bytes],
        redactor: StreamingRedactor,
        errors: list[Exception],
    ) -> None:
        assert process.stdout is not None
        try:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                self._append_output(state, redactor.feed(chunk))
            self._append_output(state, redactor.finish())
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(exc)

    def _append_output(self, state: _JobState, value: bytes) -> None:
        if not value:
            return
        with self._lock:
            state.output.extend(value)
            state.output_offset += len(value)
            remove = max(0, len(state.output) - self.output_limit)
            while remove < len(state.output) and state.output[remove] & 0xC0 == 0x80:
                remove += 1
            if remove:
                del state.output[:remove]
            state.truncated_before = state.output_offset - len(state.output)
            self._persist(state)

    def _terminate(
        self,
        process: subprocess.Popen[bytes],
        process_group: int | None,
    ) -> None:
        deadline = time.monotonic() + self.termination_grace
        if _HAS_PROCESS_GROUPS and process_group is not None:
            if self._process_group_exists(process_group):
                try:
                    os.killpg(process_group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        elif process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

        if _HAS_PROCESS_GROUPS and process_group is not None:
            while self._process_group_exists(process_group) and time.monotonic() < deadline:
                time.sleep(0.01)
            if self._process_group_exists(process_group):
                self._kill(process, process_group)
        elif process.poll() is None:
            self._kill(process, None)

    @staticmethod
    def _kill(
        process: subprocess.Popen[bytes],
        process_group: int | None,
    ) -> None:
        try:
            if _HAS_PROCESS_GROUPS and process_group is not None:
                os.killpg(process_group, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass

    @staticmethod
    def _process_group_exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _finish(
        self,
        state: _JobState,
        status: str,
        exit_code: int | None,
        *,
        error: str | None = None,
    ) -> None:
        if state.status in TERMINAL_STATUSES:
            return
        state.status = status
        state.exit_code = exit_code
        state.finished = time.time()
        state.error = error
        state.process = None
        self._persist(state)

    def _lookup(self, job_id: str) -> _JobState:
        if not _valid_job_id(job_id):
            raise KeyError("job not found")
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError("job not found") from None

    def _snapshot(
        self,
        state: _JobState,
        *,
        output: str | None = None,
        truncated_before: int | None = None,
    ) -> Job:
        return Job(
            id=state.id,
            status=state.status,
            output=(
                bytes(state.output).decode("utf-8") if output is None else output
            ),
            output_offset=state.output_offset,
            truncated_before=(
                state.truncated_before
                if truncated_before is None
                else truncated_before
            ),
            exit_code=state.exit_code,
            started=state.started,
            finished=state.finished,
            error=state.error,
        )

    def _persist(self, state: _JobState) -> None:
        metadata = self._snapshot(state).as_dict()
        metadata["process_identity"] = (
            None
            if state.process_identity is None
            else state.process_identity.as_dict()
        )
        metadata["launch"] = (
            None
            if state.launch_phase is None or state.recovery_token is None
            else {
                "phase": state.launch_phase,
                "recovery_token": state.recovery_token,
            }
        )
        target = self.job_dir / f"{state.id}.json"
        temporary = self.job_dir / f".{state.id}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            try:
                self._secure_metadata_descriptor(descriptor)
            except Exception:
                os.close(descriptor)
                raise
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_metadata(temporary, target)
            target.chmod(0o600)
            self._verify_metadata_file(target)
            self._sync_job_directory()
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _replace_metadata(temporary: Path, target: Path) -> None:
        deadline = time.monotonic() + 0.2
        while True:
            try:
                os.replace(temporary, target)
                return
            except PermissionError:
                if sys.platform != "win32" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)

    def _secure_job_directory(self) -> None:
        if not _IS_POSIX_FILESYSTEM:
            metadata = self.job_dir.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("invalid job_dir")
            self.job_dir.chmod(0o700)
            return

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.job_dir, flags)
        try:
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise PermissionError("job_dir permissions are not private")
        finally:
            os.close(descriptor)

    def _sync_job_directory(self) -> None:
        if not _IS_POSIX_FILESYSTEM:
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.job_dir, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _secure_metadata_descriptor(descriptor: int) -> None:
        if _IS_POSIX_FILESYSTEM:
            os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError("metadata is not a regular file")
        if _IS_POSIX_FILESYSTEM and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise PermissionError("metadata permissions are not private")

    @staticmethod
    def _verify_metadata_file(path: Path) -> None:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError("metadata is not a regular file")
        if _IS_POSIX_FILESYSTEM and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise PermissionError("metadata permissions are not private")

    def _load_jobs(self) -> None:
        for path in self.job_dir.glob("*.json"):
            job_id = path.stem
            if not _valid_job_id(job_id):
                continue
            self._verify_metadata_file(path)
            try:
                if path.stat().st_size > self.output_limit * 4 + 4096:
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                state = self._state_from_metadata(job_id, data)
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
                continue
            self._jobs[job_id] = state
            if state.status not in TERMINAL_STATUSES:
                if _IS_LINUX and state.process_identity is not None:
                    self._recover_with_pidfds(state.process_identity)
                state.status = "failed"
                state.exit_code = None
                state.finished = time.time()
                state.error = "interrupted by agent restart"
                self._persist(state)

    def _state_from_metadata(self, job_id: str, data: Any) -> _JobState:
        if type(data) is not dict or data.get("job_id") != job_id:
            raise ValueError("invalid job metadata")
        status = data.get("status")
        output = data.get("output")
        output_offset = data.get("output_offset")
        truncated_before = data.get("truncated_before")
        if (
            status not in _STATUSES
            or not isinstance(output, str)
            or type(output_offset) is not int
            or type(truncated_before) is not int
        ):
            raise ValueError("invalid job metadata")
        encoded = output.encode("utf-8")
        if (
            len(encoded) > self.output_limit
            or truncated_before < 0
            or output_offset != truncated_before + len(encoded)
        ):
            raise ValueError("invalid job metadata")

        exit_code = data.get("exit_code")
        started = data.get("started")
        finished = data.get("finished")
        error = data.get("error")
        process_identity = self._parse_process_identity(data.get("process_identity"))
        launch_phase, recovery_token = self._parse_launch(data.get("launch"))
        if exit_code is not None and type(exit_code) is not int:
            raise ValueError("invalid job metadata")
        if started is not None and not isinstance(started, (int, float)):
            raise ValueError("invalid job metadata")
        if finished is not None and not isinstance(finished, (int, float)):
            raise ValueError("invalid job metadata")
        if error is not None and (
            not isinstance(error, str) or len(error.encode("utf-8")) > MAX_ERROR_BYTES
        ):
            raise ValueError("invalid job metadata")
        return _JobState(
            id=job_id,
            status=status,
            output=bytearray(encoded),
            output_offset=output_offset,
            truncated_before=truncated_before,
            exit_code=exit_code,
            started=None if started is None else float(started),
            finished=None if finished is None else float(finished),
            error=error,
            process_group=(
                None
                if process_identity is None
                else process_identity.process_group
            ),
            process_identity=process_identity,
            launch_phase=launch_phase,
            recovery_token=recovery_token,
        )

    @staticmethod
    def _parse_launch(value: Any) -> tuple[str | None, str | None]:
        if value is None:
            return None, None
        if (
            type(value) is not dict
            or set(value) != {"phase", "recovery_token"}
            or value["phase"] not in {"prepared", "ready"}
            or not isinstance(value["recovery_token"], str)
            or not 32 <= len(value["recovery_token"]) <= 128
            or not value["recovery_token"].isascii()
        ):
            raise ValueError("invalid job metadata")
        return value["phase"], value["recovery_token"]

    @staticmethod
    def _parse_process_identity(value: Any) -> _ProcessIdentity | None:
        if value is None:
            return None
        expected = {
            "pid",
            "process_group",
            "session_id",
            "start_time_ticks",
            "boot_id",
            "recovery_token",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid job metadata")
        pid = value["pid"]
        process_group = value["process_group"]
        session_id = value["session_id"]
        start_time_ticks = value["start_time_ticks"]
        boot_id = value["boot_id"]
        recovery_token = value["recovery_token"]
        if type(pid) is not int or pid <= 0:
            raise ValueError("invalid job metadata")
        if process_group is not None and (
            type(process_group) is not int or process_group <= 0
        ):
            raise ValueError("invalid job metadata")
        if session_id is not None and (
            type(session_id) is not int or session_id <= 0
        ):
            raise ValueError("invalid job metadata")
        if start_time_ticks is not None and (
            type(start_time_ticks) is not int or start_time_ticks <= 0
        ):
            raise ValueError("invalid job metadata")
        if boot_id is not None and (
            not isinstance(boot_id, str) or not 1 <= len(boot_id) <= 128
        ):
            raise ValueError("invalid job metadata")
        if (
            not isinstance(recovery_token, str)
            or not 16 <= len(recovery_token) <= 128
            or not recovery_token.isascii()
        ):
            raise ValueError("invalid job metadata")
        return _ProcessIdentity(
            pid=pid,
            process_group=process_group,
            session_id=session_id,
            start_time_ticks=start_time_ticks,
            boot_id=boot_id,
            recovery_token=recovery_token,
        )

    @staticmethod
    def _capture_process_identity(pid: int, recovery_token: str) -> _ProcessIdentity:
        if not _IS_LINUX:
            return _ProcessIdentity(pid, None, None, None, None, recovery_token)
        process_group, session_id, start_time_ticks = _read_linux_stat(
            PROC_ROOT / str(pid) / "stat"
        )
        boot_id = _read_boot_id()
        if process_group != pid or session_id != pid:
            raise ValueError("process did not create an isolated session")
        return _ProcessIdentity(
            pid=pid,
            process_group=process_group,
            session_id=session_id,
            start_time_ticks=start_time_ticks,
            boot_id=boot_id,
            recovery_token=recovery_token,
        )

    def _recover_with_pidfds(self, identity: _ProcessIdentity) -> None:
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if not callable(pidfd_open) or not callable(pidfd_send_signal):
            return
        if not self._recovery_identity_is_current(identity):
            return

        deadline = time.monotonic() + self.termination_grace
        terminated: set[int] = set()
        while True:
            matching = self._matching_recovery_pids(identity)
            if not matching:
                return
            new_members = [pid for pid in matching if pid not in terminated]
            self._signal_recovery_pids(
                identity,
                new_members,
                signal.SIGTERM,
                pidfd_open,
                pidfd_send_signal,
            )
            terminated.update(new_members)
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))

        self._signal_recovery_pids(
            identity,
            self._matching_recovery_pids(identity),
            signal.SIGKILL,
            pidfd_open,
            pidfd_send_signal,
        )

    @staticmethod
    def _recovery_identity_is_current(identity: _ProcessIdentity) -> bool:
        if (
            identity.process_group is None
            or identity.session_id is None
            or identity.start_time_ticks is None
            or identity.boot_id is None
            or identity.pid != identity.process_group
            or identity.process_group != identity.session_id
        ):
            return False
        try:
            return _read_boot_id() == identity.boot_id
        except (OSError, ValueError):
            return False

    @staticmethod
    def _matching_recovery_pids(identity: _ProcessIdentity) -> list[int]:
        candidates = [PROC_ROOT / str(identity.pid)]
        try:
            candidates.extend(
                path
                for path in PROC_ROOT.iterdir()
                if path.name.isdecimal() and path.name != str(identity.pid)
            )
        except OSError:
            return []
        matching: list[int] = []
        for process_dir in candidates:
            if JobRunner._recovery_process_matches(identity, process_dir):
                matching.append(int(process_dir.name))
        return matching

    @staticmethod
    def _recovery_process_matches(
        identity: _ProcessIdentity,
        process_dir: Path,
    ) -> bool:
        try:
            process_group, session_id, start_time_ticks = _read_linux_stat(
                process_dir / "stat"
            )
            if process_group != identity.process_group or session_id != identity.session_id:
                return False
            if process_dir.name == str(identity.pid) and (
                start_time_ticks != identity.start_time_ticks
            ):
                return False
            return _proc_environment_has_token(
                process_dir / "environ", identity.recovery_token
            )
        except (OSError, ValueError):
            return False

    @staticmethod
    def _signal_recovery_pids(
        identity: _ProcessIdentity,
        process_ids: list[int],
        sent_signal: int,
        pidfd_open: Any,
        pidfd_send_signal: Any,
    ) -> None:
        for pid in process_ids:
            try:
                descriptor = pidfd_open(pid, 0)
            except OSError:
                continue
            try:
                if not JobRunner._recovery_process_matches(
                    identity, PROC_ROOT / str(pid)
                ):
                    continue
                try:
                    pidfd_send_signal(descriptor, sent_signal, None, 0)
                except OSError:
                    continue
            finally:
                os.close(descriptor)

    @staticmethod
    def _bounded_error(exc: Exception) -> str:
        value = redact_text(f"{type(exc).__name__}: {exc}")
        encoded = value.encode("utf-8")
        if len(encoded) <= MAX_ERROR_BYTES:
            return value
        encoded = encoded[:MAX_ERROR_BYTES]
        while encoded and encoded[-1] & 0xC0 == 0x80:
            encoded = encoded[:-1]
        return encoded.decode("utf-8", errors="ignore")


def safe_environment(
    environment: Mapping[str, str] | Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Build a deterministic environment without inheriting process secrets."""

    try:
        items = environment.items() if isinstance(environment, Mapping) else environment
        values = list(items)
    except (TypeError, ValueError, AttributeError):
        raise ValueError("invalid environment") from None
    if len(values) > 16:
        raise ValueError("invalid environment")
    clean = {"PATH": SAFE_PATH}
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("invalid environment")
        key, value = item
        if (
            not isinstance(key, str)
            or key not in _ALLOWED_ENVIRONMENT
            or not isinstance(value, str)
            or len(value) > 256
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
        ):
            raise ValueError("invalid environment")
        clean[key] = value
    return clean


def _valid_job_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _read_linux_stat(path: Path) -> tuple[int, int, int]:
    raw = _read_limited_bytes(path, 4096).decode("ascii")
    command_end = raw.rfind(")")
    if command_end < 0:
        raise ValueError("invalid process stat")
    fields = raw[command_end + 1 :].split()
    if len(fields) <= 19:
        raise ValueError("invalid process stat")
    process_group = int(fields[2])
    session_id = int(fields[3])
    start_time_ticks = int(fields[19])
    if process_group <= 0 or session_id <= 0 or start_time_ticks <= 0:
        raise ValueError("invalid process stat")
    return process_group, session_id, start_time_ticks


def _read_boot_id() -> str:
    value = _read_limited_bytes(
        PROC_ROOT / "sys" / "kernel" / "random" / "boot_id", 128
    ).decode("ascii").strip()
    if not value or len(value) > 128 or not value.isascii():
        raise ValueError("invalid boot id")
    return value


def _proc_environment_has_token(path: Path, token: str) -> bool:
    expected = f"{_RECOVERY_ENVIRONMENT_KEY}={token}".encode("ascii")
    return expected in _read_limited_bytes(path, 1024 * 1024).split(b"\0")


def _read_limited_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("process metadata is too large")
    return value


__all__ = [
    "DEFAULT_OUTPUT_LIMIT",
    "Job",
    "JobRunner",
    "MAX_OUTPUT_LIMIT",
    "SAFE_PATH",
    "TERMINAL_STATUSES",
    "safe_environment",
]
