from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
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
    SecretSetting,
)
from app.security import SecretBox
from app.services.ops_provider import AssistantTurn, ChangeStep, OpsProviderClient
from app.services.ops_tools import OpsToolRegistry
from app.services.provider_errors import OpsProviderError
from app.tasks.engine import TaskCancelled, TaskContext, TaskPaused

MAX_HISTORY_MESSAGES = 100
MAX_HISTORY_CHARS = 100_000
MAX_PROMPT_CHARS = 10_000
MIN_EMBEDDED_SECRET_CHARS = 3
_PROCESSABLE_STATUSES = ("active", "answered", "needs_input", "failed")
_SENSITIVE_HEADER = re.compile(
    r"(?:auth|authorization|api[-_]?key|token|password|secret|credential|cookie)",
    re.IGNORECASE,
)
_SYSTEM_PROMPT = (
    "You are the DGX Spark operations assistant. Return exactly one structured action. "
    "Use only the supplied read-only tools for automatic inspection. Any change, including "
    "shell, must be returned as a plan for explicit approval. Never expose credentials."
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


class _KnownSecrets:
    def __init__(self, values: set[str], *, unsafe_short: bool) -> None:
        self.values = tuple(sorted(values, key=lambda value: (-len(value), value)))
        self.unsafe_short = unsafe_short

    def contains(self, value: Any) -> bool:
        serialized = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
        for secret in self.values:
            if serialized == secret:
                return True
            if len(secret) >= 8 and secret in serialized:
                return True
            if MIN_EMBEDDED_SECRET_CHARS <= len(secret) < 8:
                pattern = rf"(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])"
                if re.search(pattern, serialized):
                    return True
        return False

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for secret in self.values:
                if result == secret:
                    result = "[REDACTED]"
                elif len(secret) >= 8:
                    result = result.replace(secret, "[REDACTED]")
                elif len(secret) >= MIN_EMBEDDED_SECRET_CHARS:
                    result = re.sub(
                        rf"(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])",
                        "[REDACTED]",
                        result,
                    )
            return result
        if isinstance(value, Mapping):
            return {str(key): self.redact(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self.redact(item) for item in value]
        return value


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
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    def respond(
        self,
        *,
        session_id: str,
        prompt: str,
        actor: str,
        check_control: Callable[[], None] | None = None,
    ) -> OpsResponseResult:
        self._validate_identifier(session_id, "session_id")
        self._validate_identifier(actor, "actor")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError("prompt must contain 1 to 10000 characters")
        secrets = self._load_known_secrets()
        if secrets.contains(prompt) or secrets.contains(actor):
            raise ValueError("request contains secret material")

        lock = self._session_lock(session_id)
        if not lock.acquire(blocking=False):
            raise OpsSessionConflict("operations session is already processing")
        claimed = False
        baseline_tool_count = self._tool_run_count(session_id)
        try:
            self._claim_and_append_prompt(session_id, prompt, actor)
            claimed = True
            return self._run_loop(
                session_id,
                actor,
                secrets,
                check_control=check_control or (lambda: None),
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
        )
        context.check_control()
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
        secrets: _KnownSecrets,
        *,
        check_control: Callable[[], None],
    ) -> OpsResponseResult:
        started = self.monotonic()
        tool_count = 0
        total_tool_chars = 0
        failed_calls: set[str] = set()
        while True:
            check_control()
            if self.monotonic() - started > self.max_total_seconds:
                return self._save_limit(session_id, actor, "time_limit", tool_count)

            provider, messages = self._load_provider_context(session_id, secrets)
            turn = self.provider_client.complete(provider, messages)
            if self.monotonic() - started > self.max_total_seconds:
                return self._save_limit(session_id, actor, "time_limit", tool_count)
            if secrets.contains(turn.model_dump(mode="json")):
                raise ValueError("provider response contains secret material")

            if turn.action == "tool":
                assert turn.tool is not None
                self._save_tool_intent(session_id, turn)
                if tool_count >= self.max_tool_turns:
                    return self._save_limit(session_id, actor, "tool_turn_limit", tool_count)
                fingerprint = self._tool_fingerprint(turn)
                if fingerprint in failed_calls:
                    return self._save_limit(session_id, actor, "repeated_failed_tool", tool_count)
                tool_run_id = self._queue_tool(session_id, turn)
                self._mark_tool_running(tool_run_id)
                try:
                    check_control()
                except (TaskCancelled, TaskPaused):
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
                    )
                    raise
                try:
                    result = self.tools.execute(turn.tool)
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
                    self._finish_tool(tool_run_id, session_id, actor, payload)
                    return self._save_limit(session_id, actor, "tool_output_limit", tool_count)
                total_tool_chars += serialized_chars
                self._finish_tool(tool_run_id, session_id, actor, payload)
                if result.status == "failed":
                    failed_calls.add(fingerprint)
                continue

            if turn.action == "plan":
                return self._create_plan(session_id, actor, turn, tool_count, secrets)
            if turn.action == "question":
                return self._save_answer(session_id, actor, turn.summary, "needs_input", tool_count)
            return self._save_answer(session_id, actor, turn.summary, "answered", tool_count)

    def _claim_and_append_prompt(self, session_id: str, prompt: str, actor: str) -> None:
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
            db.add(
                OpsMessage(
                    session_id=session_id,
                    role="user",
                    content=prompt,
                    metadata_json={"actor": actor},
                )
            )
            db.commit()

    def _load_provider_context(
        self, session_id: str, secrets: _KnownSecrets
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

    def _save_tool_intent(self, session_id: str, turn: AssistantTurn) -> str:
        assert turn.tool is not None
        with self.session_factory() as db:
            message = OpsMessage(
                session_id=session_id,
                role="assistant",
                content=turn.summary,
                metadata_json={
                    "action": "tool",
                    "tool_name": turn.tool.name,
                    "argument_keys": sorted(turn.tool.argument_dict()),
                },
            )
            db.add(message)
            db.commit()
            return message.id

    def _queue_tool(self, session_id: str, turn: AssistantTurn) -> str:
        assert turn.tool is not None
        with self.session_factory() as db:
            tool_run = OpsToolRun(
                session_id=session_id,
                tool_name=turn.tool.name,
                risk="read_only",
                status="queued",
                arguments_json=turn.tool.argument_dict(),
            )
            db.add(tool_run)
            db.commit()
            return tool_run.id

    def _mark_tool_running(self, tool_run_id: str) -> None:
        with self.session_factory() as db:
            tool_run = db.get(OpsToolRun, tool_run_id)
            if tool_run is None:
                raise RuntimeError("queued tool run disappeared")
            tool_run.status = "running"
            tool_run.started_at = datetime.now(UTC)
            db.commit()

    def _finish_tool(
        self,
        tool_run_id: str,
        session_id: str,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        with self.session_factory() as db:
            tool_run = db.get(OpsToolRun, tool_run_id)
            if tool_run is None:
                raise RuntimeError("running tool run disappeared")
            status = str(payload.get("status") or "failed")
            tool_run.status = status
            tool_run.result_json = payload
            error = payload.get("error")
            tool_run.error = str(error)[:1000] if error else None
            tool_run.finished_at = datetime.now(UTC)
            message = OpsMessage(
                session_id=session_id,
                role="tool",
                content=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                metadata_json={"tool_name": tool_run.tool_name, "tool_run_id": tool_run.id},
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
            db.commit()

    def _create_plan(
        self,
        session_id: str,
        actor: str,
        turn: AssistantTurn,
        tool_count: int,
        secrets: _KnownSecrets,
    ) -> OpsResponseResult:
        serialized_steps = [self._serialize_step(step) for step in turn.steps]
        candidate = {"summary": turn.summary, "steps": serialized_steps}
        if secrets.unsafe_short or secrets.contains(candidate):
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
        self, session_id: str, actor: str, reason: str, tool_count: int
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
            db.commit()
            return OpsResponseResult(
                session_id=session_id,
                status=status,
                tool_count=tool_count,
                message_id=message.id,
            )

    def _load_known_secrets(self) -> _KnownSecrets:
        values: set[str] = set()
        unsafe_short = False
        try:
            with self.session_factory() as db:
                providers = db.execute(select(Provider.encrypted_api_key, Provider.headers)).all()
                settings = db.scalars(select(SecretSetting.encrypted_value)).all()
            encrypted_values: list[str] = []
            for encrypted_api_key, headers in providers:
                encrypted_values.append(encrypted_api_key)
                values.add(encrypted_api_key)
                if isinstance(headers, Mapping):
                    for name, value in headers.items():
                        if not isinstance(value, str) or not value:
                            continue
                        scheme, separator, credential = value.partition(" ")
                        if _SENSITIVE_HEADER.search(str(name)) or (
                            separator and scheme.casefold() in {"bearer", "basic"}
                        ):
                            values.add(value)
                            if credential:
                                values.add(credential)
            for encrypted_value in settings:
                encrypted_values.append(encrypted_value)
                values.add(encrypted_value)
            for encrypted in encrypted_values:
                plaintext = self.secret_box.decrypt(encrypted)
                if not plaintext:
                    raise ValueError("empty configured secret")
                values.add(plaintext)
            unsafe_short = any(len(value) < MIN_EMBEDDED_SECRET_CHARS for value in values)
        except Exception:
            raise RuntimeError("configured secrets could not be loaded safely") from None
        if unsafe_short:
            raise RuntimeError("configured secrets are too short to sanitize safely")
        return _KnownSecrets(values, unsafe_short=False)

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
        value = step.model_dump(mode="json")
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
