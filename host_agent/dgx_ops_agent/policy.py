"""Fail-closed policy for operations executed by the host agent."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

MAX_COMMAND_LENGTH = 16_384
MAX_CWD_LENGTH = 4_096
MAX_ENVIRONMENT_VARIABLES = 16
MAX_ENVIRONMENT_VALUE_LENGTH = 256
MAX_IDENTIFIER_LENGTH = 128
MAX_SERVICE_LENGTH = 256
DEFAULT_TIMEOUT = 30
MIN_TIMEOUT = 1
MAX_TIMEOUT = 3_600
MIN_TAIL = 1
MAX_TAIL = 5_000

_CONTAINER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SERVICE_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+\Z")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)
_ALLOWED_ENVIRONMENT_KEYS = frozenset(
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


class PolicyError(ValueError):
    """Raised when a requested operation does not satisfy the agent policy."""


@dataclass(frozen=True, slots=True)
class ApprovalMetadata:
    """Bounded, non-secret metadata proving a Shell step was approved."""

    plan_id: str
    step_id: str
    approved_by: str
    approved_at: str


@dataclass(frozen=True, slots=True)
class ValidatedAction:
    """An immutable process invocation safe to hand to an executor."""

    action: str
    argv: tuple[str, ...]
    cwd: str | None
    timeout: int
    read_only: bool
    approval: ApprovalMetadata | None
    environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Fixed executable arguments and a strict parameter schema for one tool."""

    argv: tuple[str, ...]
    parameter_kind: str
    timeout: int

    def bind(self, action: str, parameters: dict[str, Any]) -> ValidatedAction:
        if self.parameter_kind == "none":
            _require_exact_keys(parameters, frozenset())
            argv = self.argv
        elif self.parameter_kind == "container":
            _require_exact_keys(parameters, frozenset({"container"}))
            container = _validate_container(parameters["container"])
            argv = (*self.argv, container)
        elif self.parameter_kind == "container_tail":
            _require_exact_keys(parameters, frozenset({"container", "tail"}))
            container = _validate_container(parameters["container"])
            tail = _validate_tail(parameters["tail"])
            argv = (*self.argv, str(tail), container)
        elif self.parameter_kind == "service":
            _require_exact_keys(parameters, frozenset({"service"}))
            service = _validate_service(parameters["service"])
            argv = (*self.argv, service)
        elif self.parameter_kind == "service_tail":
            _require_exact_keys(parameters, frozenset({"service", "tail"}))
            service = _validate_service(parameters["service"])
            tail = _validate_tail(parameters["tail"])
            argv = (*self.argv, service, "--lines", str(tail))
        else:  # pragma: no cover - specifications are module constants
            raise PolicyError("invalid tool policy")
        return ValidatedAction(
            action=action,
            argv=argv,
            cwd=None,
            timeout=self.timeout,
            read_only=True,
            approval=None,
            environment=(),
        )


READ_ONLY_TOOLS: Mapping[str, ToolSpec] = MappingProxyType(
    {
        "host.memory": ToolSpec(("/usr/bin/free", "--bytes"), "none", 10),
        "host.disk": ToolSpec(
            (
                "/usr/bin/df",
                "--block-size=1",
                "--output=source,target,size,used,avail,pcent",
            ),
            "none",
            10,
        ),
        "host.gpu": ToolSpec(
            (
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ),
            "none",
            15,
        ),
        "host.ports": ToolSpec(("/usr/bin/ss", "-lntupH"), "none", 10),
        "host.processes": ToolSpec(
            (
                "/usr/bin/ps",
                "-eo",
                "pid,ppid,user,stat,%cpu,%mem,etimes,comm",
                "--sort=-%cpu",
            ),
            "none",
            10,
        ),
        "docker.list": ToolSpec(
            ("/usr/bin/docker", "container", "list", "--all", "--no-trunc"),
            "none",
            15,
        ),
        "docker.inspect": ToolSpec(
            ("/usr/bin/docker", "container", "inspect"), "container", 15
        ),
        "docker.logs": ToolSpec(
            ("/usr/bin/docker", "logs", "--tail"), "container_tail", 15
        ),
        "docker.stats": ToolSpec(
            ("/usr/bin/docker", "stats", "--no-stream"), "container", 15
        ),
        "systemd.status": ToolSpec(
            ("/usr/bin/systemctl", "--no-pager", "--full", "status"), "service", 15
        ),
        "systemd.journal": ToolSpec(
            (
                "/usr/bin/journalctl",
                "--no-pager",
                "--output=short-iso",
                "--unit",
            ),
            "service_tail",
            15,
        ),
    }
)
READ_ONLY_ACTIONS = frozenset(READ_ONLY_TOOLS)


def validate_action(
    action: str,
    parameters: dict[str, Any],
    approval: dict[str, Any] | None,
) -> ValidatedAction:
    """Validate an action and bind it to a fixed executable invocation."""

    if not isinstance(action, str):
        raise PolicyError("unknown action")
    if type(parameters) is not dict:
        raise PolicyError("invalid parameters")

    spec = READ_ONLY_TOOLS.get(action)
    if spec is not None:
        return spec.bind(action, parameters)
    if action != "shell.execute":
        raise PolicyError("unknown action")

    command, cwd, timeout, environment = _validate_shell_parameters(parameters)
    approval_metadata = _validate_approval(approval)
    return ValidatedAction(
        action=action,
        argv=("/bin/bash", "-lc", command),
        cwd=cwd,
        timeout=timeout,
        read_only=False,
        approval=approval_metadata,
        environment=environment,
    )


def _require_exact_keys(parameters: dict[str, Any], expected: frozenset[str]) -> None:
    if frozenset(parameters) != expected:
        raise PolicyError("invalid parameters")


def _validate_container(value: Any) -> str:
    if not isinstance(value, str) or _CONTAINER_PATTERN.fullmatch(value) is None:
        raise PolicyError("invalid parameters")
    return value


def _validate_service(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_SERVICE_LENGTH
        or value.startswith("-")
        or _SERVICE_PATTERN.fullmatch(value) is None
    ):
        raise PolicyError("invalid parameters")
    return value


def _validate_tail(value: Any) -> int:
    if type(value) is not int or not MIN_TAIL <= value <= MAX_TAIL:
        raise PolicyError("invalid parameters")
    return value


def _validate_shell_parameters(
    parameters: dict[str, Any],
) -> tuple[str, str, int, tuple[tuple[str, str], ...]]:
    keys = frozenset(parameters)
    if not frozenset({"command", "cwd"}) <= keys <= frozenset(
        {"command", "cwd", "env", "timeout"}
    ):
        raise PolicyError("invalid parameters")
    command = parameters["command"]
    cwd = parameters["cwd"]
    timeout = parameters.get("timeout", DEFAULT_TIMEOUT)
    environment = _validate_environment(parameters.get("env", {}))

    if (
        not isinstance(command, str)
        or not command.strip()
        or len(command) > MAX_COMMAND_LENGTH
        or "\x00" in command
    ):
        raise PolicyError("invalid parameters")
    if (
        not isinstance(cwd, str)
        or not cwd
        or len(cwd) > MAX_CWD_LENGTH
        or "\x00" in cwd
        or not cwd.startswith("/")
        or cwd.startswith("//")
        or posixpath.normpath(cwd) != cwd
    ):
        raise PolicyError("invalid parameters")
    if type(timeout) is not int or not MIN_TIMEOUT <= timeout <= MAX_TIMEOUT:
        raise PolicyError("invalid parameters")
    return command, cwd, timeout, environment


def _validate_environment(value: Any) -> tuple[tuple[str, str], ...]:
    if type(value) is not dict or len(value) > MAX_ENVIRONMENT_VARIABLES:
        raise PolicyError("invalid parameters")

    environment: list[tuple[str, str]] = []
    for key, item in value.items():
        if (
            type(key) is not str
            or key not in _ALLOWED_ENVIRONMENT_KEYS
            or type(item) is not str
            or len(item) > MAX_ENVIRONMENT_VALUE_LENGTH
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in item)
        ):
            raise PolicyError("invalid parameters")
        environment.append((key, item))
    return tuple(sorted(environment))


def _validate_approval(approval: dict[str, Any] | None) -> ApprovalMetadata:
    if type(approval) is not dict:
        raise PolicyError("invalid approval")
    expected = frozenset({"plan_id", "step_id", "approved_by", "approved_at"})
    if frozenset(approval) != expected:
        raise PolicyError("invalid approval")

    plan_id = _validate_approval_identifier(approval["plan_id"])
    step_id = _validate_approval_identifier(approval["step_id"])
    approved_by = _validate_approval_identifier(approval["approved_by"])
    approved_at = approval["approved_at"]
    if (
        not isinstance(approved_at, str)
        or _UTC_TIMESTAMP_PATTERN.fullmatch(approved_at) is None
    ):
        raise PolicyError("invalid approval")
    try:
        datetime.fromisoformat(approved_at[:-1] + "+00:00")
    except ValueError:
        raise PolicyError("invalid approval") from None

    return ApprovalMetadata(
        plan_id=plan_id,
        step_id=step_id,
        approved_by=approved_by,
        approved_at=approved_at,
    )


def _validate_approval_identifier(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_IDENTIFIER_LENGTH
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PolicyError("invalid approval")
    return value


__all__ = [
    "ApprovalMetadata",
    "PolicyError",
    "READ_ONLY_ACTIONS",
    "READ_ONLY_TOOLS",
    "ToolSpec",
    "ValidatedAction",
    "validate_action",
]
