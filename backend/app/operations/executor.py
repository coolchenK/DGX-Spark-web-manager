from __future__ import annotations

import hashlib
import posixpath
import re
import threading
from datetime import UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.audit import record_audit
from app.models import Deployment, OperationPlan, OpsMessage, OpsSession
from app.security import SecretBox
from app.services.diagnostics import operation_plan_digest
from app.services.ops_secrets import KnownSecrets, load_known_secrets
from app.tasks.engine import TaskCancelled, TaskContext, TaskPaused
from host_agent.dgx_ops_agent.redaction import StreamingRedactor

_TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "timed_out", "cancelled"})
_ACTIVE_PLAN_STATUSES = frozenset({"approved", "executing"})
_MAX_STEP_OUTPUT_CHARS = 100_000
_APPROVAL_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@:-]{0,127}\Z")


class OperationExecutor:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        deployment_service,
        discovery_service,
        agent_client=None,
        secret_box: SecretBox | None = None,
        poll_interval_seconds: float = 0.25,
    ):
        self.session_factory = session_factory
        self.deployment_service = deployment_service
        self.discovery_service = discovery_service
        self.agent_client = agent_client
        self.secret_box = secret_box
        self.poll_interval_seconds = max(0.0, poll_interval_seconds)

    def handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(payload["plan_id"])
        steps, approval_digest, approved_by, approved_at = self._claim_plan(plan_id)
        results: list[dict[str, Any]] = []
        try:
            secrets = load_known_secrets(self.session_factory, self.secret_box)
        except Exception:
            self._save_plan_result(plan_id, "failed", results)
            raise RuntimeError("Operation plan could not load redaction material") from None

        for index, step in enumerate(steps):
            try:
                context.check_control()
            except (TaskCancelled, TaskPaused):
                self._save_plan_result(plan_id, "failed", results)
                raise
            self._verify_approval_digest(plan_id, approval_digest)
            if not step.get("executable"):
                results.append({"index": index, "status": "skipped", "reason": "Not executable"})
                self._save_plan_result(plan_id, "executing", results)
                context.update(progress=(index + 1) / max(len(steps), 1) * 100)
                continue
            try:
                if step.get("operation") == "shell":
                    result = self._execute_shell(
                        context,
                        plan_id=plan_id,
                        index=index,
                        step=step,
                        approved_by=approved_by,
                        approved_at=approved_at,
                        secrets=secrets,
                        prior_results=results,
                    )
                else:
                    result = self._execute_structured(context, step)
            except (TaskCancelled, TaskPaused):
                if step.get("operation") != "shell":
                    self._save_plan_result(plan_id, "failed", results)
                raise
            except Exception:
                if step.get("operation") != "shell":
                    results.append(
                        {
                            "index": index,
                            "status": "failed",
                            "reason": "Operation step failed",
                        }
                    )
                    self._save_plan_result(plan_id, "failed", results)
                raise
            results.append({"index": index, "status": "succeeded", **result})
            self._save_plan_result(plan_id, "executing", results)
            context.update(progress=(index + 1) / max(len(steps), 1) * 100)

        self._save_plan_result(plan_id, "completed", results)
        return {"plan_id": plan_id, "steps": results}

    def _claim_plan(
        self, plan_id: str
    ) -> tuple[list[dict[str, Any]], str, str, str]:
        with self.session_factory() as db:
            plan = db.get(OperationPlan, plan_id)
            if plan is None or plan.status not in _ACTIVE_PLAN_STATUSES:
                raise ValueError("Operation plan is not approved")
            digest = (plan.result_json or {}).get("approval_digest")
            if not isinstance(digest, str) or operation_plan_digest(plan.steps) != digest:
                plan.status = "failed"
                plan.result_json = {
                    **(plan.result_json or {}),
                    "failure": "approval_changed",
                }
                db.commit()
                raise ValueError("Operation plan changed after approval")
            if not plan.approved_by or plan.approved_at is None:
                plan.status = "failed"
                db.commit()
                raise ValueError("Operation plan approval metadata is invalid")
            approved_at = plan.approved_at
            if approved_at.tzinfo is None:
                approved_at = approved_at.replace(tzinfo=UTC)
            plan.status = "executing"
            steps = [dict(step) for step in plan.steps]
            approved_by = plan.approved_by
            db.commit()
            return (
                steps,
                digest,
                approved_by,
                approved_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            )

    def _verify_approval_digest(self, plan_id: str, expected: str) -> None:
        with self.session_factory() as db:
            plan = db.get(OperationPlan, plan_id)
            if plan is None or operation_plan_digest(plan.steps) != expected:
                if plan is not None:
                    plan.status = "failed"
                    plan.result_json = {
                        **(plan.result_json or {}),
                        "failure": "approval_changed",
                    }
                    db.commit()
                raise ValueError("Operation plan changed after approval")

    def _execute_structured(
        self, context: TaskContext, step: dict[str, Any]
    ) -> dict[str, Any]:
        operation = step.get("operation")
        if operation == "rescan_inventory":
            with self.session_factory() as db:
                result = self.discovery_service.scan_all(db)
            return {"result": result}

        action = {
            "start_deployment": "start",
            "stop_deployment": "stop",
            "restart_deployment": "restart",
        }.get(operation)
        if not action:
            return {"status": "skipped", "reason": "Not allowed"}
        deployment_id = str(step["deployment_id"])
        with self.session_factory() as db:
            deployment = db.get(Deployment, deployment_id)
            if deployment is None:
                raise ValueError("Deployment was not found; retry the operation")
            action_payload = {
                "deployment_id": deployment_id,
                "action": action,
                "expected_container_id": deployment.container_id,
                "expected_container_name": deployment.container_name,
            }
        result = self.deployment_service.action_handler(context, action_payload)
        return {"result": result}

    def _execute_shell(
        self,
        context: TaskContext,
        *,
        plan_id: str,
        index: int,
        step: dict[str, Any],
        approved_by: str,
        approved_at: str,
        secrets: KnownSecrets,
        prior_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.agent_client is None:
            self._audit_shell(
                plan_id,
                step,
                actor=approved_by,
                action="ops.shell.fail",
                outcome="failure",
                summary="agent_unavailable",
                index=index,
            )
            self._save_plan_result(
                plan_id,
                "failed",
                [
                    *prior_results,
                    {"index": index, "status": "failed", "reason": "Agent unavailable"},
                ],
            )
            raise RuntimeError("Shell step failed")

        try:
            step_id, parameters = self._shell_parameters(step)
        except ValueError:
            self._audit_shell(
                plan_id,
                step,
                actor=approved_by,
                action="ops.shell.fail",
                outcome="failure",
                summary="invalid_approved_step",
                index=index,
            )
            self._save_plan_result(
                plan_id,
                "failed",
                [
                    *prior_results,
                    {"index": index, "status": "failed", "reason": "Invalid approved step"},
                ],
            )
            raise
        approval = {
            "plan_id": plan_id,
            "step_id": step_id,
            "approved_by": approved_by,
            "approved_at": approved_at,
        }
        try:
            snapshot = self._validate_job_snapshot(
                self.agent_client.call("shell.execute", parameters, approval=approval)
            )
        except Exception:
            self._audit_shell(
                plan_id,
                step,
                actor=approved_by,
                action="ops.shell.fail",
                outcome="failure",
                summary="agent_start_failed",
                index=index,
            )
            self._save_plan_result(
                plan_id,
                "failed",
                [
                    *prior_results,
                    {"index": index, "status": "failed", "reason": "Agent start failed"},
                ],
            )
            raise RuntimeError("Shell step failed") from None

        job_id = snapshot["job_id"]
        self._audit_shell(
            plan_id,
            step,
            actor=approved_by,
            action="ops.shell.start",
            outcome="success",
            summary="started",
            index=index,
            job_id=job_id,
        )
        output_parts: list[str] = []
        output_chars = 0
        redactor = StreamingRedactor(secret_values=secrets.values)
        offset, output_chars, redactor = self._consume_output(
            context,
            snapshot,
            previous_offset=0,
            output_parts=output_parts,
            output_chars=output_chars,
            redactor=redactor,
            secret_values=secrets.values,
        )

        while snapshot["status"] not in _TERMINAL_JOB_STATUSES:
            try:
                context.check_control()
            except (TaskCancelled, TaskPaused):
                try:
                    cancelled = self._validate_job_snapshot(
                        self.agent_client.call("job.cancel", {"job_id": job_id})
                    )
                    if cancelled["job_id"] != job_id:
                        raise RuntimeError("Agent job response is invalid")
                    agent_status = cancelled["status"]
                except Exception:
                    agent_status = "unknown"
                output_chars = self._append_output(
                    context,
                    redactor.finish().decode("utf-8"),
                    output_parts=output_parts,
                    output_chars=output_chars,
                )
                result = {
                    "index": index,
                    "step_id": step_id,
                    "status": "cancelled",
                    "agent_status": agent_status,
                    "output": "".join(output_parts),
                }
                self._save_plan_result(plan_id, "failed", [*prior_results, result])
                self._audit_shell(
                    plan_id,
                    step,
                    actor=approved_by,
                    action="ops.shell.cancel",
                    outcome="failure",
                    summary="task_stopped",
                    index=index,
                    job_id=job_id,
                    agent_status=agent_status,
                )
                raise

            if self.poll_interval_seconds:
                threading.Event().wait(self.poll_interval_seconds)
            try:
                snapshot = self._validate_job_snapshot(
                    self.agent_client.call(
                        "job.get",
                        {"job_id": job_id, "offset": offset},
                    )
                )
                if snapshot["job_id"] != job_id:
                    raise RuntimeError("Agent job response is invalid")
                offset, output_chars, redactor = self._consume_output(
                    context,
                    snapshot,
                    previous_offset=offset,
                    output_parts=output_parts,
                    output_chars=output_chars,
                    redactor=redactor,
                    secret_values=secrets.values,
                )
            except Exception:
                result = {
                    "index": index,
                    "step_id": step_id,
                    "status": "failed",
                    "agent_status": "unknown",
                    "agent_job_id": job_id,
                    "output": "".join(output_parts),
                }
                self._save_plan_result(plan_id, "failed", [*prior_results, result])
                self._audit_shell(
                    plan_id,
                    step,
                    actor=approved_by,
                    action="ops.shell.fail",
                    outcome="failure",
                    summary="agent_poll_failed",
                    index=index,
                    job_id=job_id,
                )
                raise RuntimeError("Shell step failed") from None

        output = "".join(output_parts)
        if snapshot["status"] == "succeeded" and snapshot["exit_code"] == 0:
            self._audit_shell(
                plan_id,
                step,
                actor=approved_by,
                action="ops.shell.succeed",
                outcome="success",
                summary="completed",
                index=index,
                job_id=job_id,
                agent_status=snapshot["status"],
                exit_code=0,
            )
            return {
                "step_id": step_id,
                "agent_job_id": job_id,
                "output": output,
                "exit_code": 0,
            }

        result = {
            "index": index,
            "step_id": step_id,
            "status": "failed",
            "agent_status": snapshot["status"],
            "agent_job_id": job_id,
            "output": output,
            "exit_code": snapshot["exit_code"],
        }
        self._save_plan_result(plan_id, "failed", [*prior_results, result])
        self._audit_shell(
            plan_id,
            step,
            actor=approved_by,
            action="ops.shell.fail",
            outcome="failure",
            summary="agent_job_failed",
            index=index,
            job_id=job_id,
            agent_status=snapshot["status"],
            exit_code=snapshot["exit_code"],
        )
        raise RuntimeError("Shell step failed")

    @staticmethod
    def _shell_parameters(step: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        expected = {
            "id",
            "operation",
            "command",
            "cwd",
            "timeout",
            "reason",
            "impact",
            "rollback",
            "executable",
        }
        if set(step) != expected or step.get("operation") != "shell":
            raise ValueError("Approved Shell step is invalid")
        step_id = step.get("id")
        command = step.get("command")
        cwd = step.get("cwd")
        timeout = step.get("timeout")
        if (
            not isinstance(step_id, str)
            or _APPROVAL_IDENTIFIER.fullmatch(step_id) is None
            or not isinstance(command, str)
            or not command.strip()
            or len(command) > 8000
            or "\x00" in command
            or not isinstance(cwd, str)
            or len(cwd) > 1000
            or not cwd.startswith("/")
            or cwd.startswith("//")
            or "\x00" in cwd
            or posixpath.normpath(cwd) != cwd
            or type(timeout) is not int
            or not 1 <= timeout <= 600
            or step.get("executable") is not True
            or any(
                not isinstance(step.get(key), str)
                or not step[key].strip()
                or len(step[key]) > 2000
                for key in ("reason", "impact", "rollback")
            )
        ):
            raise ValueError("Approved Shell step is invalid")
        return step_id, {"command": command, "cwd": cwd, "timeout": timeout}

    @staticmethod
    def _validate_job_snapshot(value: Any) -> dict[str, Any]:
        expected = {
            "job_id",
            "status",
            "output",
            "output_offset",
            "truncated_before",
            "exit_code",
            "started",
            "finished",
            "error",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RuntimeError("Agent job response is invalid")
        job_id = value.get("job_id")
        status = value.get("status")
        output = value.get("output")
        output_offset = value.get("output_offset")
        truncated_before = value.get("truncated_before")
        exit_code = value.get("exit_code")
        if (
            not isinstance(job_id, str)
            or not job_id
            or len(job_id) > 128
            or status not in {"queued", "running", *_TERMINAL_JOB_STATUSES}
            or not isinstance(output, str)
            or type(output_offset) is not int
            or type(truncated_before) is not int
            or truncated_before < 0
            or output_offset != truncated_before + len(output.encode("utf-8"))
            or (exit_code is not None and type(exit_code) is not int)
        ):
            raise RuntimeError("Agent job response is invalid")
        return value

    @staticmethod
    def _consume_output(
        context: TaskContext,
        snapshot: dict[str, Any],
        *,
        previous_offset: int,
        output_parts: list[str],
        output_chars: int,
        redactor: StreamingRedactor,
        secret_values: tuple[str, ...],
    ) -> tuple[int, int, StreamingRedactor]:
        truncated_before = snapshot["truncated_before"]
        if (
            truncated_before < previous_offset
            or snapshot["output_offset"] < previous_offset
        ):
            raise RuntimeError("Agent output offset is invalid")
        if truncated_before > previous_offset and output_chars < _MAX_STEP_OUTPUT_CHARS:
            output_chars = OperationExecutor._append_output(
                context,
                redactor.finish().decode("utf-8"),
                output_parts=output_parts,
                output_chars=output_chars,
            )
            marker = "[Earlier Agent output was truncated]\n"
            remaining = _MAX_STEP_OUTPUT_CHARS - output_chars
            bounded_marker = marker[:remaining]
            output_parts.append(bounded_marker)
            output_chars += len(bounded_marker)
            context.update(message=bounded_marker)
            redactor = StreamingRedactor(secret_values=secret_values)
        chunk = redactor.feed(snapshot["output"].encode("utf-8")).decode("utf-8")
        if snapshot["status"] in _TERMINAL_JOB_STATUSES:
            chunk += redactor.finish().decode("utf-8")
        output_chars = OperationExecutor._append_output(
            context,
            chunk,
            output_parts=output_parts,
            output_chars=output_chars,
        )
        return snapshot["output_offset"], output_chars, redactor

    @staticmethod
    def _append_output(
        context: TaskContext,
        chunk: str,
        *,
        output_parts: list[str],
        output_chars: int,
    ) -> int:
        if not chunk or output_chars >= _MAX_STEP_OUTPUT_CHARS:
            return output_chars
        remaining = _MAX_STEP_OUTPUT_CHARS - output_chars
        bounded = chunk[:remaining]
        output_parts.append(bounded)
        context.update(message=bounded)
        return output_chars + len(bounded)

    def _save_plan_result(
        self,
        plan_id: str,
        status: str,
        results: list[dict[str, Any]],
    ) -> None:
        with self.session_factory() as db:
            plan = db.get(OperationPlan, plan_id)
            if plan is None:
                return
            plan.status = status
            plan.result_json = {
                **(plan.result_json or {}),
                "steps": results,
            }
            if status in {"completed", "failed"}:
                session_ids = db.scalars(
                    select(OpsMessage.session_id).where(
                        OpsMessage.operation_plan_id == plan_id
                    )
                )
                session_status = "active" if status == "completed" else "failed"
                for session_id in set(session_ids):
                    session = db.get(OpsSession, session_id)
                    if session is not None:
                        session.status = session_status
            db.commit()

    def _audit_shell(
        self,
        plan_id: str,
        step: dict[str, Any],
        *,
        actor: str,
        action: str,
        outcome: str,
        summary: str,
        index: int,
        job_id: str | None = None,
        agent_status: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        command = step.get("command")
        command_digest = (
            hashlib.sha256(command.encode("utf-8")).hexdigest()
            if isinstance(command, str)
            else None
        )
        details = {
            "step_id": step.get("id"),
            "step_index": index,
            "command_digest": command_digest,
            "summary": summary[:128],
        }
        if job_id is not None:
            details["agent_job_id"] = job_id
        if agent_status is not None:
            details["agent_status"] = agent_status
        if exit_code is not None:
            details["exit_code"] = exit_code
        with self.session_factory() as db:
            record_audit(
                db,
                actor=actor,
                action=action,
                resource_type="operation_plan",
                resource_id=plan_id,
                outcome=outcome,
                details=details,
            )
            db.commit()
