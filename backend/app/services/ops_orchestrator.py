from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.audit import record_audit
from app.models import (
    Deployment,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
    new_id,
)
from app.security import SecretBox
from app.services.ops_provider import AssistantTurn, ChangeStep, OpsProviderClient
from app.services.ops_secrets import KnownSecrets, load_known_secrets
from app.services.ops_tools import OpsToolRegistry
from app.services.provider_errors import OpsProviderError
from app.tasks.engine import TaskCancelled, TaskContext, TaskPaused

MAX_HISTORY_MESSAGES = 100
MAX_HISTORY_CHARS = 100_000
MAX_PROMPT_CHARS = 10_000
_PROCESSABLE_STATUSES = ("active", "answered", "needs_input", "failed")
_SYSTEM_PROMPT = (
    "You are the DGX Spark operations assistant. You do not have native tool calling. "
    "Return exactly one JSON object in message.content, with no markdown or commentary. "
    "For a read-only inspection use: "
    '{"action":"tool","summary":"why this read is needed","tool":'
    '{"name":"host.memory","arguments":{}},"steps":[]}. '
    "For a final response use: "
    '{"action":"answer","summary":"answer text","tool":null,"steps":[]}. '
    'For a clarification use action "question" with the same answer shape. '
    "For any change use: "
    '{"action":"plan","summary":"plan summary","tool":null,"steps":['
    '{"operation":"shell","deployment_id":null,"command":"exact command",'
    '"cwd":"/absolute/path","timeout":60,"reason":"reason","impact":"impact",'
    '"rollback":"rollback"}]}. '
    "Allowed read-only names are host.memory, host.disk, host.gpu, host.ports, "
    "host.processes, docker.list, docker.inspect, docker.logs, docker.stats, "
    "systemd.status, systemd.journal, manager.summary, manager.tasks, and "
    "manager.gateway. Container tools require a container string; docker.logs also "
    "requires integer tail. Systemd tools require a service string; systemd.journal "
    "also requires integer tail. manager.tasks requires integer limit. manager.gateway "
    "requires integer minutes and limit. Other tools use empty arguments. Never emit "
    "tool_calls. Never expose credentials. Read-only tools may be requested automatically. "
    "Every change, including shell, must be a plan for explicit approval."
)


class ProviderCompleter(Protocol):
    def complete(self, provider: Provider, messages: list[dict[str, Any]]) -> AssistantTurn: ...


@dataclass(frozen=True)
class OpsResponseResult:
    session_id: str
    status: str
    tool_count: int = 0
    message_id: str | None = None
    plan_id: str | None = None


class OpsSessionNotFound(ValueError):
    pass


class OpsSessionConflict(ValueError):
    pass


class OpsDeadlineExceeded(TimeoutError):
    pass


class _CallState:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class _BoundedCallRunner:
    def __init__(self, *, max_workers: int = 4, poll_seconds: float = 0.02) -> None:
        self._capacity = threading.BoundedSemaphore(max_workers)
        self._poll_seconds = poll_seconds

    def run(
        self,
        call: Callable[[], Any],
        *,
        deadline: float,
        monotonic: Callable[[], float],
        check_control: Callable[[], None],
    ) -> Any:
        while not self._capacity.acquire(blocking=False):
            check_control()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OpsDeadlineExceeded()
            threading.Event().wait(min(self._poll_seconds, remaining))

        state = _CallState()

        def worker() -> None:
            try:
                state.result = call()
            except BaseException as exc:
                state.error = exc
            finally:
                state.done.set()
                self._capacity.release()

        thread = threading.Thread(target=worker, name="ops-bounded-call", daemon=True)
        try:
            thread.start()
        except BaseException:
            self._capacity.release()
            raise
        while True:
            check_control()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OpsDeadlineExceeded()
            if state.done.wait(min(self._poll_seconds, remaining)):
                check_control()
                if deadline - monotonic() <= 0:
                    raise OpsDeadlineExceeded()
                if state.error is not None:
                    raise state.error
                return state.result


class OpsOrchestrator:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        provider_client: OpsProviderClient | ProviderCompleter,
        tools: OpsToolRegistry,
        secret_box: SecretBox,
        max_tool_turns: int = 6,
        max_total_seconds: float = 180,
        max_tool_result_chars: int = 30_000,
        max_total_tool_chars: int = 120_000,
        monotonic: Callable[[], float] = time.monotonic,
        call_runner: _BoundedCallRunner | None = None,
    ) -> None:
        if max_tool_turns < 0 or max_total_seconds <= 0:
            raise ValueError("orchestrator limits must be positive")
        self.session_factory = session_factory
        self.provider_client = provider_client
        self.tools = tools
        self.secret_box = secret_box
        self.max_tool_turns = max_tool_turns
        self.max_total_seconds = max_total_seconds
        self.max_tool_result_chars = max_tool_result_chars
        self.max_total_tool_chars = max_total_tool_chars
        self.monotonic = monotonic
        self._call_runner = call_runner or _BoundedCallRunner()
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    def respond(
        self,
        *,
        session_id: str,
        prompt: str,
        actor: str,
        check_control: Callable[[], None] | None = None,
        request_id: str | None = None,
    ) -> OpsResponseResult:
        self._validate_identifier(session_id, "session_id")
        self._validate_identifier(actor, "actor")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError("prompt must contain 1 to 10000 characters")
        secrets = self._load_known_secrets()
        if secrets.contains(prompt) or secrets.contains(actor):
            raise ValueError("request contains secret material")
        request_id = request_id or new_id()
        self._validate_identifier(request_id, "request_id")
        control = check_control or (lambda: None)

        lock = self._session_lock(session_id)
        if not lock.acquire(blocking=False):
            raise OpsSessionConflict("operations session is already processing")
        claimed = False
        baseline_tool_count = self._tool_run_count(session_id)
        try:
            self._claim_and_append_prompt(session_id, prompt, actor, request_id, control)
            claimed = True
            return self._run_loop(
                session_id,
                actor,
                secrets,
                request_id=request_id,
                check_control=control,
            )
        except (OpsSessionNotFound, OpsSessionConflict):
            raise
        except (TaskCancelled, TaskPaused) as exc:
            if claimed:
                reason = "cancelled" if isinstance(exc, TaskCancelled) else "paused"
                self._release_controlled_session(session_id, actor, reason)
            raise
        except OpsProviderError:
            if claimed:
                return self._save_failure(
                    session_id,
                    actor,
                    action="ops.failure",
                    reason="provider_failure",
                    content=(
                        "The configured AI provider could not complete this request. "
                        "Check its connection and model, then retry."
                    ),
                    tool_count=self._tool_run_count(session_id) - baseline_tool_count,
                )
            raise
        except ValueError:
            if claimed:
                self._save_failure(
                    session_id,
                    actor,
                    action="ops.failure",
                    reason="validation_failure",
                    content=(
                        "The proposed operation was rejected because it contained "
                        "unsafe or invalid data."
                    ),
                )
            raise
        except Exception:
            if claimed:
                self._save_failure(
                    session_id,
                    actor,
                    action="ops.failure",
                    reason="internal_failure",
                    content=(
                        "The operations assistant encountered an internal error. "
                        "Review service health and retry."
                    ),
                )
            raise RuntimeError("operations orchestration failed") from None
        finally:
            with self._locks_guard:
                lock.release()
                if self._session_locks.get(session_id) is lock:
                    self._session_locks.pop(session_id, None)

    def recover_interrupted(self) -> int:
        with self.session_factory() as db:
            sessions = list(db.scalars(select(OpsSession).where(OpsSession.status == "processing")))
            session_ids = [session.id for session in sessions]
            interrupted_at = datetime.now(UTC)
            if session_ids:
                tools = list(
                    db.scalars(
                        select(OpsToolRun).where(
                            OpsToolRun.session_id.in_(session_ids),
                            OpsToolRun.status.in_(("queued", "running")),
                        )
                    )
                )
                for tool_run in tools:
                    internal = {
                        key: value
                        for key, value in (tool_run.result_json or {}).items()
                        if key in {"_request_id", "_tool_fingerprint"}
                    }
                    tool_run.status = "failed"
                    tool_run.result_json = {
                        "status": "failed",
                        "reason": "manager_restart",
                    }
                    tool_run.error = "Interrupted by manager restart"
                    tool_run.finished_at = interrupted_at
                    db.add(
                        OpsMessage(
                            session_id=tool_run.session_id,
                            role="tool",
                            content=json.dumps(
                                tool_run.result_json,
                                ensure_ascii=True,
                                separators=(",", ":"),
                            ),
                            metadata_json={
                                "tool_name": tool_run.tool_name,
                                "tool_run_id": tool_run.id,
                                "status": "failed",
                                "request_id": internal.get("_request_id"),
                                "tool_fingerprint": internal.get("_tool_fingerprint"),
                            },
                        )
                    )
                    record_audit(
                        db,
                        actor="system",
                        action="ops.tool.execute",
                        resource_type="ops_tool_run",
                        resource_id=tool_run.id,
                        outcome="failure",
                        details={
                            "session_id": tool_run.session_id,
                            "tool_name": tool_run.tool_name,
                            "risk": "read_only",
                            "status": "failed",
                            "argument_keys": sorted(tool_run.arguments_json),
                        },
                    )
            for session in sessions:
                session.status = "active"
                record_audit(
                    db,
                    actor="system",
                    action="ops.failure",
                    resource_type="ops_session",
                    resource_id=session.id,
                    outcome="failure",
                    details={"reason": "manager_restart"},
                )
            db.commit()
            return len(sessions)

    def handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != {"session_id", "prompt", "actor"}:
            raise ValueError("ops.respond payload is invalid")
        session_id = self._required_payload_string(payload, "session_id")
        prompt = self._required_payload_string(payload, "prompt")
        actor = self._required_payload_string(payload, "actor")
        context.check_control()
        result = self.respond(
            session_id=session_id,
            prompt=prompt,
            actor=actor,
            check_control=context.check_control,
            request_id=getattr(context, "task_id", None),
        )
        return {
            "session_id": result.session_id,
            "status": result.status,
            "tool_count": result.tool_count,
            "message_id": result.message_id,
            "plan_id": result.plan_id,
        }

    def _run_loop(
        self,
        session_id: str,
        actor: str,
        secrets: KnownSecrets,
        *,
        request_id: str,
        check_control: Callable[[], None],
    ) -> OpsResponseResult:
        deadline = self.monotonic() + self.max_total_seconds
        wall_deadline = time.monotonic() + self.max_total_seconds
        tool_count = 0
        total_tool_chars = 0
        failed_calls = self._load_failed_calls(session_id, request_id)

        def stop_with_limit(reason: str) -> OpsResponseResult:
            check_control()
            return self._save_limit(
                session_id,
                actor,
                reason,
                tool_count,
                check_control=check_control,
            )

        while True:
            check_control()
            if self.monotonic() >= deadline:
                return stop_with_limit("time_limit")

            provider, messages = self._load_provider_context(session_id, secrets)
            try:
                turn = self._call_runner.run(
                    lambda provider=provider, messages=messages: self._complete_provider(
                        provider,
                        messages,
                        max(0.001, wall_deadline - time.monotonic()),
                    ),
                    deadline=wall_deadline,
                    monotonic=time.monotonic,
                    check_control=check_control,
                )
            except OpsDeadlineExceeded:
                return stop_with_limit("time_limit")
            if self.monotonic() >= deadline:
                return stop_with_limit("time_limit")
            if secrets.contains(turn.model_dump(mode="json")):
                raise ValueError("provider response contains secret material")

            if turn.action == "tool":
                assert turn.tool is not None
                if tool_count >= self.max_tool_turns:
                    return stop_with_limit("tool_turn_limit")
                fingerprint = self._tool_fingerprint(turn)
                if fingerprint in failed_calls:
                    return stop_with_limit("repeated_failed_tool")
                self._save_tool_intent(
                    session_id,
                    turn,
                    request_id,
                    fingerprint,
                    check_control,
                )
                tool_run_id = self._queue_tool(
                    session_id,
                    turn,
                    request_id,
                    fingerprint,
                    check_control,
                )
                try:
                    self._mark_tool_running(tool_run_id, check_control)
                    check_control()
                    result = self._call_runner.run(
                        lambda turn=turn: self._execute_tool(
                            turn,
                            max(0.001, wall_deadline - time.monotonic()),
                        ),
                        deadline=wall_deadline,
                        monotonic=time.monotonic,
                        check_control=check_control,
                    )
                except OpsDeadlineExceeded:
                    self._finish_tool(
                        tool_run_id,
                        session_id,
                        actor,
                        {
                            "name": turn.tool.name,
                            "risk": "read_only",
                            "status": "failed",
                            "output": {},
                            "error": "Read-only tool exceeded the operation deadline",
                        },
                    )
                    return stop_with_limit("time_limit")
                except (TaskCancelled, TaskPaused) as exc:
                    controlled_reason = (
                        "cancelled" if isinstance(exc, TaskCancelled) else "paused"
                    )
                    self._finish_tool(
                        tool_run_id,
                        session_id,
                        actor,
                        {
                            "name": turn.tool.name,
                            "risk": "read_only",
                            "status": "failed",
                            "output": {},
                            "error": "Read-only tool did not run because the task stopped",
                        },
                        controlled_reason=controlled_reason,
                    )
                    raise
                except Exception:
                    self._finish_tool(
                        tool_run_id,
                        session_id,
                        actor,
                        {
                            "name": turn.tool.name,
                            "risk": "read_only",
                            "status": "failed",
                            "output": {},
                            "error": "Read-only tool execution failed internally",
                        },
                    )
                    raise
                tool_count += 1
                try:
                    check_control()
                    payload = secrets.redact(result.model_dump(mode="json"))
                    payload = self._bound_payload(payload, self.max_tool_result_chars)
                    serialized_chars = len(
                        json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                    )
                    if total_tool_chars + serialized_chars > self.max_total_tool_chars:
                        payload = {
                            "name": result.name,
                            "risk": "read_only",
                            "status": result.status,
                            "output": {"truncated": True},
                            "error": "Cumulative tool output limit reached",
                        }
                        payload = self._bound_payload(
                            payload,
                            max(0, self.max_total_tool_chars - total_tool_chars),
                        )
                        self._finish_tool(
                            tool_run_id,
                            session_id,
                            actor,
                            payload,
                            check_control=check_control,
                        )
                        return stop_with_limit("tool_output_limit")
                    total_tool_chars += serialized_chars
                    self._finish_tool(
                        tool_run_id,
                        session_id,
                        actor,
                        payload,
                        check_control=check_control,
                    )
                except (TaskCancelled, TaskPaused) as exc:
                    controlled_reason = (
                        "cancelled" if isinstance(exc, TaskCancelled) else "paused"
                    )
                    self._finish_tool(
                        tool_run_id,
                        session_id,
                        actor,
                        {
                            "name": turn.tool.name,
                            "risk": "read_only",
                            "status": "failed",
                            "output": {},
                            "error": "Read-only tool stopped before its result was committed",
                        },
                        controlled_reason=controlled_reason,
                    )
                    raise
                if result.status == "failed":
                    failed_calls.add(fingerprint)
                continue

            if turn.action == "plan":
                return self._create_plan(
                    session_id,
                    actor,
                    turn,
                    tool_count,
                    secrets,
                    check_control,
                )
            if turn.action == "question":
                return self._save_answer(
                    session_id,
                    actor,
                    turn.summary,
                    "needs_input",
                    tool_count,
                    check_control,
                )
            return self._save_answer(
                session_id,
                actor,
                turn.summary,
                "answered",
                tool_count,
                check_control,
            )

    def _complete_provider(
        self,
        provider: Provider,
        messages: list[dict[str, Any]],
        timeout_seconds: float,
    ) -> AssistantTurn:
        complete = self.provider_client.complete
        if self._accepts_keyword(complete, "timeout_seconds"):
            return complete(provider, messages, timeout_seconds=timeout_seconds)
        return complete(provider, messages)

    def _execute_tool(self, turn: AssistantTurn, timeout_seconds: float) -> Any:
        assert turn.tool is not None
        execute = self.tools.execute
        if self._accepts_keyword(execute, "timeout_seconds"):
            return execute(turn.tool, timeout_seconds=timeout_seconds)
        return execute(turn.tool)

    @staticmethod
    def _accepts_keyword(call: Callable[..., Any], keyword: str) -> bool:
        try:
            parameters = inspect.signature(call).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == keyword
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _claim_and_append_prompt(
        self,
        session_id: str,
        prompt: str,
        actor: str,
        request_id: str,
        check_control: Callable[[], None],
    ) -> None:
        with self.session_factory() as db:
            session = db.get(OpsSession, session_id)
            if session is None:
                raise OpsSessionNotFound("operations session was not found")
            provider = db.get(Provider, session.provider_id) if session.provider_id else None
            if provider is None or not provider.enabled:
                raise ValueError("operations session provider is missing or disabled")
            if session.deployment_id and db.get(Deployment, session.deployment_id) is None:
                raise ValueError("operations session deployment is missing")
            claimed = db.execute(
                update(OpsSession)
                .where(
                    OpsSession.id == session_id,
                    OpsSession.status.in_(_PROCESSABLE_STATUSES),
                )
                .values(status="processing", updated_at=datetime.now(UTC))
            )
            if claimed.rowcount != 1:
                raise OpsSessionConflict("operations session is already processing")
            user_messages = db.scalars(
                select(OpsMessage).where(
                    OpsMessage.session_id == session_id,
                    OpsMessage.role == "user",
                )
            )
            if not any(
                (message.metadata_json or {}).get("request_id") == request_id
                for message in user_messages
            ):
                db.add(
                    OpsMessage(
                        session_id=session_id,
                        role="user",
                        content=prompt,
                        metadata_json={"actor": actor, "request_id": request_id},
                    )
                )
            check_control()
            db.commit()

    def _load_failed_calls(self, session_id: str, request_id: str) -> set[str]:
        with self.session_factory() as db:
            messages = db.scalars(
                select(OpsMessage).where(
                    OpsMessage.session_id == session_id,
                    OpsMessage.role == "tool",
                )
            )
            return {
                str(metadata["tool_fingerprint"])
                for message in messages
                if (metadata := message.metadata_json or {}).get("request_id") == request_id
                and metadata.get("status") == "failed"
                and metadata.get("tool_fingerprint")
            }

    def _load_provider_context(
        self, session_id: str, secrets: KnownSecrets
    ) -> tuple[Provider, list[dict[str, Any]]]:
        with self.session_factory() as db:
            session = db.get(OpsSession, session_id)
            if session is None:
                raise OpsSessionNotFound("operations session was not found")
            provider = db.get(Provider, session.provider_id) if session.provider_id else None
            if provider is None or not provider.enabled:
                raise ValueError("operations session provider is missing or disabled")
            rows = list(
                db.scalars(
                    select(OpsMessage)
                    .where(OpsMessage.session_id == session_id)
                    .order_by(OpsMessage.created_at.desc(), OpsMessage.id.desc())
                    .limit(MAX_HISTORY_MESSAGES)
                )
            )
            db.expunge(provider)
        rows.reverse()
        messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        used_chars = len(_SYSTEM_PROMPT)
        selected: list[dict[str, Any]] = []
        for message in reversed(rows):
            content = secrets.redact(message.content)
            if message.role == "tool":
                tool_name = str(message.metadata_json.get("tool_name") or "read-only tool")
                content = f"Tool result ({tool_name}): {content}"
                role = "user"
            else:
                role = message.role if message.role in {"user", "assistant"} else "user"
            remaining = MAX_HISTORY_CHARS - used_chars
            if remaining <= 0:
                break
            content = str(content)[:remaining]
            selected.append({"role": role, "content": content})
            used_chars += len(content)
        messages.extend(reversed(selected))
        return provider, messages

    def _save_tool_intent(
        self,
        session_id: str,
        turn: AssistantTurn,
        request_id: str,
        fingerprint: str,
        check_control: Callable[[], None],
    ) -> str:
        assert turn.tool is not None
        with self.session_factory() as db:
            messages = db.scalars(
                select(OpsMessage).where(
                    OpsMessage.session_id == session_id,
                    OpsMessage.role == "assistant",
                )
            )
            for existing in messages:
                metadata = existing.metadata_json or {}
                if (
                    metadata.get("request_id") == request_id
                    and metadata.get("tool_fingerprint") == fingerprint
                ):
                    return existing.id
            message = OpsMessage(
                session_id=session_id,
                role="assistant",
                content=turn.summary,
                metadata_json={
                    "action": "tool",
                    "tool_name": turn.tool.name,
                    "argument_keys": sorted(turn.tool.argument_dict()),
                    "request_id": request_id,
                    "tool_fingerprint": fingerprint,
                },
            )
            db.add(message)
            check_control()
            db.commit()
            return message.id

    def _queue_tool(
        self,
        session_id: str,
        turn: AssistantTurn,
        request_id: str,
        fingerprint: str,
        check_control: Callable[[], None],
    ) -> str:
        assert turn.tool is not None
        with self.session_factory() as db:
            tool_run = OpsToolRun(
                session_id=session_id,
                tool_name=turn.tool.name,
                risk="read_only",
                status="queued",
                arguments_json=turn.tool.argument_dict(),
                result_json={
                    "_request_id": request_id,
                    "_tool_fingerprint": fingerprint,
                },
            )
            db.add(tool_run)
            check_control()
            db.commit()
            return tool_run.id

    def _mark_tool_running(
        self, tool_run_id: str, check_control: Callable[[], None]
    ) -> None:
        with self.session_factory() as db:
            tool_run = db.get(OpsToolRun, tool_run_id)
            if tool_run is None:
                raise RuntimeError("queued tool run disappeared")
            tool_run.status = "running"
            tool_run.started_at = datetime.now(UTC)
            check_control()
            db.commit()

    def _finish_tool(
        self,
        tool_run_id: str,
        session_id: str,
        actor: str,
        payload: dict[str, Any],
        *,
        check_control: Callable[[], None] | None = None,
        controlled_reason: str | None = None,
    ) -> None:
        with self.session_factory() as db:
            tool_run = db.get(OpsToolRun, tool_run_id)
            if tool_run is None:
                raise RuntimeError("running tool run disappeared")
            status = str(payload.get("status") or "failed")
            tool_run.status = status
            internal = {
                key: value
                for key, value in (tool_run.result_json or {}).items()
                if key in {"_request_id", "_tool_fingerprint"}
            }
            tool_run.result_json = payload
            error = payload.get("error")
            tool_run.error = str(error)[:1000] if error else None
            if tool_run.started_at is None:
                tool_run.started_at = datetime.now(UTC)
            tool_run.finished_at = datetime.now(UTC)
            message = OpsMessage(
                session_id=session_id,
                role="tool",
                content=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                metadata_json={
                    "tool_name": tool_run.tool_name,
                    "tool_run_id": tool_run.id,
                    "status": status,
                    "request_id": internal.get("_request_id"),
                    "tool_fingerprint": internal.get("_tool_fingerprint"),
                },
            )
            db.add(message)
            record_audit(
                db,
                actor=actor,
                action="ops.tool.execute",
                resource_type="ops_tool_run",
                resource_id=tool_run.id,
                outcome="success" if status == "succeeded" else "failure",
                details={
                    "session_id": session_id,
                    "tool_name": tool_run.tool_name,
                    "risk": "read_only",
                    "status": status,
                    "argument_keys": sorted(tool_run.arguments_json),
                },
            )
            if controlled_reason is not None:
                session = db.get(OpsSession, session_id)
                if session is not None:
                    session.status = "active"
                    record_audit(
                        db,
                        actor=actor,
                        action="ops.failure",
                        resource_type="ops_session",
                        resource_id=session_id,
                        outcome="failure",
                        details={"reason": controlled_reason},
                    )
            if check_control is not None:
                check_control()
            db.commit()

    def _create_plan(
        self,
        session_id: str,
        actor: str,
        turn: AssistantTurn,
        tool_count: int,
        secrets: KnownSecrets,
        check_control: Callable[[], None],
    ) -> OpsResponseResult:
        serialized_steps = [self._serialize_step(step) for step in turn.steps]
        candidate = {"summary": turn.summary, "steps": serialized_steps}
        if secrets.contains(candidate):
            raise ValueError("operation plan contains secret material")
        with self.session_factory() as db:
            session = db.get(OpsSession, session_id)
            if session is None:
                raise OpsSessionNotFound("operations session was not found")
            deployment_ids = {
                step.deployment_id for step in turn.steps if step.deployment_id is not None
            }
            if deployment_ids:
                existing_ids = set(
                    db.scalars(select(Deployment.id).where(Deployment.id.in_(deployment_ids)))
                )
                if existing_ids != deployment_ids:
                    raise ValueError("operation plan references a missing deployment")
            plan = OperationPlan(
                provider_id=session.provider_id,
                deployment_id=session.deployment_id,
                summary=turn.summary,
                diagnosis=turn.summary,
                risk=self._plan_risk(turn.steps),
                steps=serialized_steps,
                status="pending",
                requested_by=actor,
            )
            db.add(plan)
            db.flush()
            message = OpsMessage(
                session_id=session_id,
                role="assistant",
                content=turn.summary,
                metadata_json={"action": "plan", "risk": plan.risk},
                operation_plan_id=plan.id,
            )
            db.add(message)
            session.status = "approval_required"
            record_audit(
                db,
                actor=actor,
                action="ops.plan.create",
                resource_type="operation_plan",
                resource_id=plan.id,
                details={
                    "session_id": session_id,
                    "risk": plan.risk,
                    "step_count": len(serialized_steps),
                },
            )
            check_control()
            db.commit()
            return OpsResponseResult(
                session_id=session_id,
                status="approval_required",
                tool_count=tool_count,
                message_id=message.id,
                plan_id=plan.id,
            )

    def _save_answer(
        self,
        session_id: str,
        actor: str,
        content: str,
        status: str,
        tool_count: int,
        check_control: Callable[[], None],
    ) -> OpsResponseResult:
        with self.session_factory() as db:
            session = db.get(OpsSession, session_id)
            if session is None:
                raise OpsSessionNotFound("operations session was not found")
            message = OpsMessage(
                session_id=session_id,
                role="assistant",
                content=content,
                metadata_json={"action": "question" if status == "needs_input" else "answer"},
            )
            db.add(message)
            session.status = status
            db.flush()
            record_audit(
                db,
                actor=actor,
                action="ops.answer.create",
                resource_type="ops_message",
                resource_id=message.id,
                details={"session_id": session_id, "status": status},
            )
            check_control()
            db.commit()
            return OpsResponseResult(
                session_id=session_id,
                status=status,
                tool_count=tool_count,
                message_id=message.id,
            )

    def _release_controlled_session(self, session_id: str, actor: str, reason: str) -> None:
        with self.session_factory() as db:
            session = db.get(OpsSession, session_id)
            if session is None:
                return
            if session.status == "active":
                return
            session.status = "active"
            record_audit(
                db,
                actor=actor,
                action="ops.failure",
                resource_type="ops_session",
                resource_id=session_id,
                outcome="failure",
                details={"reason": reason},
            )
            db.commit()

    def _save_limit(
        self,
        session_id: str,
        actor: str,
        reason: str,
        tool_count: int,
        *,
        check_control: Callable[[], None],
    ) -> OpsResponseResult:
        messages = {
            "time_limit": "The operation reached its time limit. Narrow the request and retry.",
            "tool_turn_limit": (
                "The operation reached its read-only tool limit. Review the collected "
                "results and continue with a narrower request."
            ),
            "tool_output_limit": (
                "The collected diagnostic output reached its size limit. Narrow the "
                "requested scope and retry."
            ),
            "repeated_failed_tool": (
                "The same read-only diagnostic failed repeatedly. Check the target and "
                "Agent health before retrying."
            ),
        }
        return self._save_terminal_message(
            session_id,
            actor,
            action="ops.limit",
            reason=reason,
            content=messages[reason],
            status="failed",
            tool_count=tool_count,
            check_control=check_control,
        )

    def _save_failure(
        self,
        session_id: str,
        actor: str,
        *,
        action: str,
        reason: str,
        content: str,
        tool_count: int = 0,
    ) -> OpsResponseResult:
        return self._save_terminal_message(
            session_id,
            actor,
            action=action,
            reason=reason,
            content=content,
            status="failed",
            tool_count=tool_count,
        )

    def _save_terminal_message(
        self,
        session_id: str,
        actor: str,
        *,
        action: str,
        reason: str,
        content: str,
        status: str,
        tool_count: int,
        check_control: Callable[[], None] | None = None,
    ) -> OpsResponseResult:
        with self.session_factory() as db:
            session = db.get(OpsSession, session_id)
            if session is None:
                raise OpsSessionNotFound("operations session was not found")
            message = OpsMessage(
                session_id=session_id,
                role="assistant",
                content=content,
                metadata_json={"action": "failure", "reason": reason},
            )
            db.add(message)
            session.status = status
            db.flush()
            record_audit(
                db,
                actor=actor,
                action=action,
                resource_type="ops_session",
                resource_id=session_id,
                outcome="failure",
                details={"reason": reason, "message_id": message.id},
            )
            if check_control is not None:
                check_control()
            db.commit()
            return OpsResponseResult(
                session_id=session_id,
                status=status,
                tool_count=tool_count,
                message_id=message.id,
            )

    def _load_known_secrets(self) -> KnownSecrets:
        return load_known_secrets(self.session_factory, self.secret_box)

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def _tool_run_count(self, session_id: str) -> int:
        with self.session_factory() as db:
            return len(
                db.scalars(select(OpsToolRun.id).where(OpsToolRun.session_id == session_id)).all()
            )

    @staticmethod
    def _serialize_step(step: ChangeStep) -> dict[str, Any]:
        value = step.model_dump(mode="json", exclude_none=True)
        value["id"] = new_id()
        if step.operation == "shell":
            value["cwd"] = step.cwd or "/"
        value["executable"] = step.operation != "explain_only"
        return value

    @staticmethod
    def _plan_risk(steps: list[ChangeStep]) -> str:
        if any(step.operation == "shell" for step in steps):
            return "high"
        if any(step.operation != "explain_only" for step in steps):
            return "medium"
        return "low"

    @staticmethod
    def _tool_fingerprint(turn: AssistantTurn) -> str:
        assert turn.tool is not None
        return json.dumps(
            {"name": turn.tool.name, "arguments": turn.tool.argument_dict()},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _bound_payload(payload: dict[str, Any], limit: int) -> dict[str, Any]:
        if limit <= 0:
            return {}
        serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(serialized) <= limit:
            return payload
        preview = serialized[: max(0, limit - 40)]
        bounded = {"truncated": True, "preview": preview}
        while len(json.dumps(bounded, ensure_ascii=True, separators=(",", ":"))) > limit:
            if not preview:
                return {} if limit < 18 else {"truncated": True}
            preview = preview[: max(0, len(preview) - 16)]
            bounded["preview"] = preview
        return bounded

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip() or len(value) > 255:
            raise ValueError(f"{name} is invalid")

    @staticmethod
    def _required_payload_string(payload: dict[str, Any], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ops.respond payload is invalid")
        return value


__all__ = [
    "OpsOrchestrator",
    "OpsResponseResult",
    "OpsSessionConflict",
    "OpsSessionNotFound",
]
