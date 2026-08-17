from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session

from app.models import (
    Deployment,
    ModelAsset,
    Provider,
    RequestMetric,
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
from app.services.ops_secrets import (
    TRUNCATED as _TRUNCATED,
)
from app.services.ops_secrets import (
    SecretLoadError,
    load_known_secrets,
    sanitize_string,
    sanitize_value,
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
MAX_TASK_LOG_CHARS = 4_000
MAX_TASK_ERROR_CHARS = 2_000
_MANAGER_PUBLIC_KEYS = frozenset(
    {
        "active_tasks",
        "available",
        "average_latency_ms",
        "cancel_requested",
        "completed_bytes",
        "completion_tokens",
        "created_at",
        "database",
        "deployments",
        "enabled",
        "endpoint",
        "error",
        "error_rate",
        "error_truncated",
        "failed_requests",
        "finished_at",
        "healthy",
        "id",
        "latency_ms",
        "log",
        "log_truncated",
        "model",
        "models",
        "progress",
        "prompt_tokens",
        "providers",
        "recent",
        "running",
        "started_at",
        "status",
        "status_code",
        "system",
        "tasks",
        "title",
        "total",
        "total_bytes",
        "total_requests",
        "type",
        "updated_at",
        "window_minutes",
    }
)


class _AgentCaller(Protocol):
    def call(
        self,
        action: str,
        parameters: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]: ...


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

    def execute(
        self,
        request: ReadOnlyToolRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        name = request.name
        if name not in AUTOMATIC_READ_TOOLS:
            raise ValueError(f"{name} is not an automatic read-only tool")

        secrets = self._load_known_secrets()
        if name in AGENT_READ_TOOLS:
            try:
                call = self.agent.call
                if timeout_seconds is not None and self._accepts_timeout(call):
                    output = call(
                        name,
                        request.argument_dict(),
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    output = call(name, request.argument_dict())
            except OpsAgentError as exc:
                return _fit_tool_result(
                    ToolResult(
                        name=name,
                        status="failed",
                        output={},
                        error=sanitize_string(str(exc), secrets, 1000),
                    )
                )
            except Exception as exc:
                raise ToolExecutionError("unexpected Agent failure") from exc
        else:
            try:
                output = self._execute_manager_tool(request)
                sanitized_output = sanitize_and_bound(
                    output,
                    secrets=secrets,
                    trusted_keys=_MANAGER_PUBLIC_KEYS,
                )
                return _fit_tool_result(
                    ToolResult(
                        name=name,
                        status="succeeded",
                        output=sanitized_output,
                    )
                )
            except ToolExecutionError:
                raise
            except Exception as exc:
                raise ToolExecutionError("Manager read-only tool failed") from exc

        sanitized_output = sanitize_and_bound(output, secrets=secrets)
        if name in AGENT_READ_TOOLS and output.get("status") in {
            "failed",
            "timed_out",
            "cancelled",
        }:
            raw_error = output.get("error")
            error = (
                sanitize_string(raw_error, secrets, 1000)
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

    @staticmethod
    def _accepts_timeout(call: Callable[..., Any]) -> bool:
        try:
            parameters = inspect.signature(call).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == "timeout_seconds"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _load_known_secrets(self) -> tuple[str, ...]:
        try:
            return load_known_secrets(self.session_factory, self.secret_box).values
        except SecretLoadError as exc:
            raise ToolExecutionError(str(exc)) from None

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
        log_length = func.length(TaskRecord.log)
        error_length = func.length(TaskRecord.error)
        log_tail = func.substr(
            TaskRecord.log,
            case(
                (
                    log_length > MAX_TASK_LOG_CHARS,
                    log_length - MAX_TASK_LOG_CHARS + 1,
                ),
                else_=1,
            ),
            MAX_TASK_LOG_CHARS,
        ).label("log_excerpt")
        error_excerpt = func.substr(TaskRecord.error, 1, MAX_TASK_ERROR_CHARS).label(
            "error_excerpt"
        )
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
                error_excerpt,
                (error_length > MAX_TASK_ERROR_CHARS).label("error_truncated"),
                log_tail,
                (log_length > MAX_TASK_LOG_CHARS).label("log_truncated"),
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
                    "error": (
                        f"{row.error_excerpt}\n{_TRUNCATED}"
                        if row.error_truncated
                        else row.error_excerpt
                    ),
                    "error_truncated": bool(row.error_truncated),
                    "log": (
                        f"{_TRUNCATED}\n{row.log_excerpt}" if row.log_truncated else row.log_excerpt
                    ),
                    "log_truncated": bool(row.log_truncated),
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
        aggregate = db.execute(
            select(
                func.count(RequestMetric.id).label("total"),
                func.sum(case((RequestMetric.status_code >= 400, 1), else_=0)).label("failed"),
                func.avg(RequestMetric.latency_ms).label("average_latency"),
            ).where(RequestMetric.created_at >= cutoff)
        ).one()
        total = aggregate.total or 0
        failed = aggregate.failed or 0
        average_latency = aggregate.average_latency or 0
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
    fitted = result.model_copy(update={"output": _bound_serialized(result.output, output_budget)})
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
    trusted_keys: frozenset[str] = frozenset(),
    limit: int = MAX_TOOL_OUTPUT_CHARS,
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise ToolExecutionError("tool output must be an object")
    normalized_secrets = tuple(
        sorted({secret for secret in secrets if secret}, key=lambda item: (-len(item), item))
    )
    sanitized = sanitize_value(output, normalized_secrets, trusted_keys)
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
