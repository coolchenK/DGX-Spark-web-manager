from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.models import (
    Deployment,
    ModelAsset,
    Provider,
    RequestMetric,
    SecretSetting,
    TaskRecord,
)
from app.security import SecretBox
from app.services.ops_agent import OpsAgentClient, OpsAgentError
from app.services.ops_provider import (
    ManagerGatewayToolArguments,
    ManagerTasksToolArguments,
    ReadOnlyToolName,
    ReadOnlyToolRequest,
)

AGENT_READ_TOOLS = frozenset(
    {
        "host.memory",
        "host.disk",
        "host.gpu",
        "host.ports",
        "host.processes",
        "docker.list",
        "docker.inspect",
        "docker.logs",
        "docker.stats",
        "systemd.status",
        "systemd.journal",
    }
)
MANAGER_READ_TOOLS = frozenset({"manager.summary", "manager.tasks", "manager.gateway"})
AUTOMATIC_READ_TOOLS = AGENT_READ_TOOLS | MANAGER_READ_TOOLS

MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_TOOL_RESULT_CHARS = 30_000
MAX_STRUCTURE_DEPTH = 8
MAX_COLLECTION_ITEMS = 100
MAX_STRING_CHARS = 12_000
_TRUNCATED = "[truncated]"
_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "encrypted_api_key",
        "encrypted_value",
        "token",
        "access_token",
        "api_token",
        "auth_token",
        "bearer_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "secret_key",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_token",
    "_password",
    "_passwd",
    "_secret",
    "_secret_key",
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<quote>[\"']?)"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9_-]{0,127})(?P=quote)\s*[:=]\s*"
    r"(?:(?:bearer|basic)\s+)?"
    r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;&}\]]+)''',
    re.IGNORECASE,
)


class _AgentCaller(Protocol):
    def call(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]: ...


SessionFactory = Callable[[], Session]


class ToolExecutionError(RuntimeError):
    """An unexpected registry boundary failure that must not be persisted as output."""


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: ReadOnlyToolName
    risk: Literal["read_only"] = "read_only"
    status: Literal["succeeded", "failed"]
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=1000)


class OpsToolRegistry:
    """Dispatch the fixed automatic read-only tool surface and bound its output."""

    def __init__(
        self,
        agent_client: OpsAgentClient | _AgentCaller,
        session_factory: SessionFactory,
        secret_box: SecretBox | None = None,
    ) -> None:
        self.agent = agent_client
        self.session_factory = session_factory
        self.secret_box = secret_box

    def execute(self, request: ReadOnlyToolRequest) -> ToolResult:
        name = request.name
        if name not in AUTOMATIC_READ_TOOLS:
            raise ValueError(f"{name} is not an automatic read-only tool")

        secrets = self._load_known_secrets()
        if name in AGENT_READ_TOOLS:
            try:
                output = self.agent.call(name, request.argument_dict())
            except OpsAgentError as exc:
                return _fit_tool_result(
                    ToolResult(
                        name=name,
                        status="failed",
                        output={},
                        error=_sanitize_string(str(exc), secrets, 1000),
                    )
                )
            except Exception as exc:
                raise ToolExecutionError("unexpected Agent failure") from exc
        else:
            output = self._execute_manager_tool(request)

        sanitized_output = sanitize_and_bound(output, secrets=secrets)
        if name in AGENT_READ_TOOLS and output.get("status") in {
            "failed",
            "timed_out",
            "cancelled",
        }:
            raw_error = output.get("error")
            error = (
                _sanitize_string(raw_error, secrets, 1000)
                if isinstance(raw_error, str) and raw_error
                else "Host read-only operation did not succeed"
            )
            return _fit_tool_result(
                ToolResult(
                    name=name,
                    status="failed",
                    output=sanitized_output,
                    error=error,
                )
            )
        return _fit_tool_result(
            ToolResult(
                name=name,
                status="succeeded",
                output=sanitized_output,
            )
        )

    def _load_known_secrets(self) -> tuple[str, ...]:
        values: set[str] = set()
        try:
            with self.session_factory() as db:
                providers = db.execute(select(Provider.encrypted_api_key, Provider.headers)).all()
                settings = db.scalars(select(SecretSetting.encrypted_value)).all()
                for encrypted_api_key, headers in providers:
                    self._add_encrypted_secret(values, encrypted_api_key)
                    if isinstance(headers, Mapping):
                        for header_value in headers.values():
                            if isinstance(header_value, str) and header_value:
                                values.add(header_value)
                                scheme, separator, credential = header_value.partition(" ")
                                if separator and scheme.casefold() in {"bearer", "basic"}:
                                    if credential:
                                        values.add(credential)
                for encrypted_value in settings:
                    self._add_encrypted_secret(values, encrypted_value)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError("configured secrets could not be loaded safely") from exc
        return tuple(sorted(values, key=len, reverse=True))

    def _add_encrypted_secret(self, values: set[str], encrypted: Any) -> None:
        if not isinstance(encrypted, str) or not encrypted or self.secret_box is None:
            raise ToolExecutionError("configured secrets could not be loaded safely")
        values.add(encrypted)
        try:
            plaintext = self.secret_box.decrypt(encrypted)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("configured secrets could not be loaded safely") from exc
        if plaintext:
            values.add(plaintext)

    def _execute_manager_tool(self, request: ReadOnlyToolRequest) -> dict[str, Any]:
        with self.session_factory() as db:
            if request.name == "manager.summary":
                return self._manager_summary(db)
            if request.name == "manager.tasks":
                arguments = request.arguments
                if not isinstance(arguments, ManagerTasksToolArguments):
                    raise ToolExecutionError("validated tool arguments were lost")
                return self._manager_tasks(db, arguments.limit)
            if request.name == "manager.gateway":
                arguments = request.arguments
                if not isinstance(arguments, ManagerGatewayToolArguments):
                    raise ToolExecutionError("validated tool arguments were lost")
                return self._manager_gateway(db, arguments.minutes, arguments.limit)
        raise ValueError(f"{request.name} is not an automatic read-only tool")

    @staticmethod
    def _manager_summary(db: Session) -> dict[str, Any]:
        model_total = db.scalar(select(func.count(ModelAsset.id))) or 0
        model_available = (
            db.scalar(select(func.count(ModelAsset.id)).where(ModelAsset.status == "available"))
            or 0
        )
        deployment_total = db.scalar(select(func.count(Deployment.id))) or 0
        deployment_running = (
            db.scalar(select(func.count(Deployment.id)).where(Deployment.status == "running")) or 0
        )
        deployment_healthy = (
            db.scalar(select(func.count(Deployment.id)).where(Deployment.health == "healthy")) or 0
        )
        provider_total = db.scalar(select(func.count(Provider.id))) or 0
        provider_enabled = (
            db.scalar(select(func.count(Provider.id)).where(Provider.enabled.is_(True))) or 0
        )
        provider_healthy = (
            db.scalar(select(func.count(Provider.id)).where(Provider.last_test_status == "healthy"))
            or 0
        )
        active_tasks = (
            db.scalar(
                select(func.count(TaskRecord.id)).where(
                    TaskRecord.status.in_(("queued", "running", "paused"))
                )
            )
            or 0
        )
        return {
            "models": {"total": model_total, "available": model_available},
            "deployments": {
                "total": deployment_total,
                "running": deployment_running,
                "healthy": deployment_healthy,
            },
            "providers": {
                "total": provider_total,
                "enabled": provider_enabled,
                "healthy": provider_healthy,
            },
            "system": {"database": "ok", "active_tasks": active_tasks},
        }

    @staticmethod
    def _manager_tasks(db: Session, limit: int) -> dict[str, Any]:
        rows = db.execute(
            select(
                TaskRecord.id,
                TaskRecord.type,
                TaskRecord.status,
                TaskRecord.title,
                TaskRecord.progress,
                TaskRecord.completed_bytes,
                TaskRecord.total_bytes,
                TaskRecord.cancel_requested,
                TaskRecord.error,
                TaskRecord.log,
                TaskRecord.created_at,
                TaskRecord.updated_at,
                TaskRecord.started_at,
                TaskRecord.finished_at,
            )
            .order_by(desc(TaskRecord.created_at), desc(TaskRecord.id))
            .limit(limit)
        ).all()
        return {
            "tasks": [
                {
                    "id": row.id,
                    "type": row.type,
                    "status": row.status,
                    "title": row.title,
                    "progress": row.progress,
                    "completed_bytes": row.completed_bytes,
                    "total_bytes": row.total_bytes,
                    "cancel_requested": row.cancel_requested,
                    "error": row.error,
                    "log": row.log,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "started_at": row.started_at,
                    "finished_at": row.finished_at,
                }
                for row in rows
            ]
        }

    @staticmethod
    def _manager_gateway(db: Session, minutes: int, limit: int) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        total = (
            db.scalar(
                select(func.count(RequestMetric.id)).where(
                    RequestMetric.created_at >= cutoff
                )
            )
            or 0
        )
        failed = (
            db.scalar(
                select(func.count(RequestMetric.id)).where(
                    RequestMetric.created_at >= cutoff,
                    RequestMetric.status_code >= 400,
                )
            )
            or 0
        )
        average_latency = (
            db.scalar(
                select(func.avg(RequestMetric.latency_ms)).where(
                    RequestMetric.created_at >= cutoff
                )
            )
            or 0
        )
        recent_rows = db.execute(
            select(
                RequestMetric.model,
                RequestMetric.endpoint,
                RequestMetric.status_code,
                RequestMetric.latency_ms,
                RequestMetric.prompt_tokens,
                RequestMetric.completion_tokens,
                RequestMetric.created_at,
            )
            .where(RequestMetric.created_at >= cutoff)
            .order_by(desc(RequestMetric.created_at), desc(RequestMetric.id))
            .limit(limit)
        ).all()
        return {
            "total_requests": total,
            "failed_requests": failed,
            "error_rate": failed / total if total else 0,
            "average_latency_ms": round(float(average_latency), 2),
            "window_minutes": minutes,
            "recent": [
                {
                    "model": row.model,
                    "endpoint": row.endpoint,
                    "status_code": row.status_code,
                    "latency_ms": row.latency_ms,
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "created_at": row.created_at,
                }
                for row in recent_rows
            ],
        }


def _is_sensitive_key(value: str) -> bool:
    normalized = re.sub(r"[-\s]+", "_", value.casefold()).strip("_")
    return normalized in _SENSITIVE_KEY_NAMES or normalized.endswith(
        _SENSITIVE_KEY_SUFFIXES
    )


def _redact_credential_assignment(match: re.Match[str]) -> str:
    if _is_sensitive_key(match.group("label")):
        return _REDACTED
    return match.group(0)


def _sanitize_string(value: str, secrets: tuple[str, ...], limit: int) -> str:
    sanitized = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        _redact_credential_assignment, value
    )
    for secret in secrets:
        sanitized = sanitized.replace(secret, _REDACTED)
    if len(sanitized) <= limit:
        return sanitized
    marker = f"\n{_TRUNCATED}\n"
    available = max(0, limit - len(marker))
    head = (available + 1) // 2
    tail = available - head
    return f"{sanitized[:head]}{marker}{sanitized[-tail:] if tail else ''}"


def _sanitize_value(value: Any, secrets: tuple[str, ...], depth: int = 0) -> Any:
    if depth >= MAX_STRUCTURE_DEPTH:
        return _TRUNCATED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _sanitize_string(value, secrets, MAX_STRING_CHARS)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_COLLECTION_ITEMS:
                result["truncated"] = True
                break
            key_text = _sanitize_string(str(key), secrets, 256)
            if _is_sensitive_key(key_text):
                result[f"redacted_field_{index}"] = _REDACTED
            else:
                result[key_text] = _sanitize_value(item, secrets, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _sanitize_value(item, secrets, depth + 1) for item in value[:MAX_COLLECTION_ITEMS]
        ]
        if len(value) > MAX_COLLECTION_ITEMS:
            result.append(_TRUNCATED)
        return result
    return f"[unsupported:{type(value).__name__}]"


def _bound_serialized(value: dict[str, Any], limit: int) -> dict[str, Any]:
    serialized = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if len(serialized) <= limit:
        return value
    preview_limit = max(0, limit - 64)
    while preview_limit >= 0:
        bounded = {
            "truncated": True,
            "preview": serialized[:preview_limit],
        }
        if len(json.dumps(bounded, ensure_ascii=True)) <= limit:
            return bounded
        preview_limit -= max(1, preview_limit // 10)
    return {"truncated": True}


def _fit_tool_result(result: ToolResult) -> ToolResult:
    if len(result.model_dump_json()) <= MAX_TOOL_RESULT_CHARS:
        return result

    empty = result.model_copy(update={"output": {}})
    output_budget = max(
        2,
        MAX_TOOL_RESULT_CHARS - len(empty.model_dump_json()) + 2,
    )
    fitted = result.model_copy(
        update={"output": _bound_serialized(result.output, output_budget)}
    )
    if len(fitted.model_dump_json()) <= MAX_TOOL_RESULT_CHARS:
        return fitted

    fail_safe = result.model_copy(update={"output": {"truncated": True}})
    if len(fail_safe.model_dump_json()) <= MAX_TOOL_RESULT_CHARS:
        return fail_safe
    raise ToolExecutionError("tool result could not be bounded safely")


def sanitize_and_bound(
    output: Mapping[str, Any],
    *,
    secrets: tuple[str, ...] = (),
    limit: int = MAX_TOOL_OUTPUT_CHARS,
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise ToolExecutionError("tool output must be an object")
    sanitized = _sanitize_value(output, secrets)
    if not isinstance(sanitized, dict):  # pragma: no cover - Mapping always maps to dict
        raise ToolExecutionError("tool output could not be sanitized")
    return _bound_serialized(sanitized, limit)


__all__ = [
    "AGENT_READ_TOOLS",
    "AUTOMATIC_READ_TOOLS",
    "MANAGER_READ_TOOLS",
    "MAX_TOOL_OUTPUT_CHARS",
    "MAX_TOOL_RESULT_CHARS",
    "OpsToolRegistry",
    "ToolExecutionError",
    "ToolResult",
    "sanitize_and_bound",
]
