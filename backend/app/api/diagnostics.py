from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, select

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import (
    Deployment,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
    TaskRecord,
)
from app.services.diagnostics import ProviderReadinessError, operation_plan_digest

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


class DiagnosticRequest(BaseModel):
    provider_id: str
    deployment_id: str | None = None
    prompt: str = Field(min_length=1, max_length=10_000)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be blank")
        return value


class OpsSessionRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=128)
    deployment_id: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=1, max_length=255)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class OpsMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


def serialize_plan(plan: OperationPlan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "provider_id": plan.provider_id,
        "deployment_id": plan.deployment_id,
        "summary": plan.summary,
        "diagnosis": plan.diagnosis,
        "risk": plan.risk,
        "steps": plan.steps,
        "status": plan.status,
        "requested_by": plan.requested_by,
        "approved_by": plan.approved_by,
        "approved_at": plan.approved_at,
        "result": plan.result_json,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
    }


def serialize_session(session: OpsSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "provider_id": session.provider_id,
        "provider_name": session.provider.name if session.provider is not None else None,
        "deployment_id": session.deployment_id,
        "deployment_name": (
            session.deployment.name if session.deployment is not None else None
        ),
        "status": session.status,
        "requested_by": session.requested_by,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def serialize_message(message: OpsMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "metadata": message.metadata_json,
        "operation_plan_id": message.operation_plan_id,
        "created_at": message.created_at,
        "updated_at": message.updated_at,
    }


def serialize_tool_run(tool_run: OpsToolRun) -> dict[str, Any]:
    return {
        "id": tool_run.id,
        "session_id": tool_run.session_id,
        "tool_name": tool_run.tool_name,
        "risk": tool_run.risk,
        "status": tool_run.status,
        "arguments": tool_run.arguments_json,
        "result": tool_run.result_json,
        "agent_job_id": tool_run.agent_job_id,
        "error": tool_run.error,
        "started_at": tool_run.started_at,
        "finished_at": tool_run.finished_at,
        "created_at": tool_run.created_at,
        "updated_at": tool_run.updated_at,
    }


def _enabled_provider(db: DbSession, provider_id: str) -> Provider:
    provider = db.get(Provider, provider_id)
    if provider is None or not provider.enabled:
        raise HTTPException(status_code=404, detail="Enabled provider not found")
    return provider


def _validate_deployment(db: DbSession, deployment_id: str | None) -> None:
    if deployment_id is not None and db.get(Deployment, deployment_id) is None:
        raise HTTPException(status_code=404, detail="Deployment not found")


def _create_session(
    db: DbSession,
    *,
    provider_id: str,
    deployment_id: str | None,
    title: str,
    actor: str,
) -> OpsSession:
    session = OpsSession(
        provider_id=provider_id,
        deployment_id=deployment_id,
        title=title,
        requested_by=actor,
    )
    db.add(session)
    db.flush()
    record_audit(
        db,
        actor=actor,
        action="ops.session.create",
        resource_type="ops_session",
        resource_id=session.id,
        details={"provider_id": provider_id, "deployment_id": deployment_id},
    )
    return session


def _active_response_task(db: DbSession, session_id: str) -> TaskRecord | None:
    tasks = db.scalars(
        select(TaskRecord).where(
            TaskRecord.type == "ops.respond",
            TaskRecord.status.in_(("queued", "running", "paused")),
        )
    )
    return next(
        (
            task
            for task in tasks
            if (task.input_json or {}).get("session_id") == session_id
        ),
        None,
    )


def _queue_message(
    request: Request,
    db: DbSession,
    *,
    session: OpsSession,
    provider: Provider,
    content: str,
    actor: str,
) -> TaskRecord:
    if session.status not in {"active", "answered", "needs_input", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"Operations session cannot accept a message while {session.status}",
        )
    if _active_response_task(db, session.id) is not None:
        raise HTTPException(
            status_code=409,
            detail="Operations session already has an active response task",
        )
    try:
        request.app.state.diagnostic_service.ensure_provider_ready(db, provider)
    except ProviderReadinessError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    task = request.app.state.task_engine.create_task(
        db,
        task_type="ops.respond",
        title=f"AI 运维响应：{session.title[:80]}",
        input_json={
            "session_id": session.id,
            "prompt": content,
            "actor": actor,
        },
        idempotency_key=f"ops-respond:{session.id}",
        commit=False,
    )
    record_audit(
        db,
        actor=actor,
        action="ops.respond.queue",
        resource_type="ops_session",
        resource_id=session.id,
        details={"task_id": task.id},
    )
    return task


@router.get("")
def list_plans(db: DbSession, _: Admin) -> list[dict[str, Any]]:
    plans = db.scalars(select(OperationPlan).order_by(desc(OperationPlan.created_at)).limit(100))
    return [serialize_plan(plan) for plan in plans]


@router.get("/sessions")
def list_sessions(db: DbSession, _: Admin) -> list[dict[str, Any]]:
    sessions = db.scalars(
        select(OpsSession).order_by(desc(OpsSession.updated_at), desc(OpsSession.id)).limit(100)
    )
    return [serialize_session(session) for session in sessions]


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(
    payload: OpsSessionRequest,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    provider = _enabled_provider(db, payload.provider_id)
    _validate_deployment(db, payload.deployment_id)
    session = _create_session(
        db,
        provider_id=provider.id,
        deployment_id=payload.deployment_id,
        title=payload.title,
        actor=str(admin["username"]),
    )
    db.commit()
    db.refresh(session)
    return serialize_session(session)


@router.get("/sessions/{session_id}")
def get_session(session_id: str, db: DbSession, _: Admin) -> dict[str, Any]:
    session = db.get(OpsSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Operations session not found")
    messages = list(
        db.scalars(
            select(OpsMessage)
            .where(OpsMessage.session_id == session.id)
            .order_by(OpsMessage.created_at, OpsMessage.id)
        )
    )
    tool_runs = list(
        db.scalars(
            select(OpsToolRun)
            .where(OpsToolRun.session_id == session.id)
            .order_by(OpsToolRun.created_at, OpsToolRun.id)
        )
    )
    plan_ids = {
        message.operation_plan_id
        for message in messages
        if message.operation_plan_id is not None
    }
    plans = (
        list(
            db.scalars(
                select(OperationPlan)
                .where(OperationPlan.id.in_(plan_ids))
                .order_by(OperationPlan.created_at, OperationPlan.id)
            )
        )
        if plan_ids
        else []
    )
    return {
        **serialize_session(session),
        "messages": [serialize_message(message) for message in messages],
        "tool_runs": [serialize_tool_run(tool_run) for tool_run in tool_runs],
        "plans": [serialize_plan(plan) for plan in plans],
    }


@router.post(
    "/sessions/{session_id}/messages",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_session_message(
    session_id: str,
    payload: OpsMessageRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    session = db.get(OpsSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Operations session not found")
    if session.provider_id is None:
        raise HTTPException(
            status_code=409,
            detail="Operations session Provider no longer exists",
        )
    provider = _enabled_provider(db, session.provider_id)
    task = _queue_message(
        request,
        db,
        session=session,
        provider=provider,
        content=payload.content,
        actor=str(admin["username"]),
    )
    db.commit()
    db.refresh(task)
    request.app.state.task_engine.notify()
    return serialize_task(task)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_diagnostic(
    payload: DiagnosticRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    provider = _enabled_provider(db, payload.provider_id)
    _validate_deployment(db, payload.deployment_id)
    actor = str(admin["username"])
    try:
        request.app.state.diagnostic_service.ensure_provider_ready(db, provider)
        session = _create_session(
            db,
            provider_id=provider.id,
            deployment_id=payload.deployment_id,
            title=payload.prompt[:255],
            actor=actor,
        )
        task = _queue_message(
            request,
            db,
            session=session,
            provider=provider,
            content=payload.prompt,
            actor=actor,
        )
        db.commit()
    except ProviderReadinessError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from None
    db.refresh(task)
    request.app.state.task_engine.notify()
    return serialize_task(task)


@router.post("/{plan_id}/approve", status_code=status.HTTP_202_ACCEPTED)
def approve_plan(
    plan_id: str,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    plan = db.get(OperationPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Operation plan not found")
    if plan.status != "pending":
        raise HTTPException(status_code=409, detail=f"Plan is already {plan.status}")
    try:
        approval_digest = operation_plan_digest(plan.steps)
    except ValueError:
        raise HTTPException(status_code=409, detail="Operation plan steps are invalid") from None
    plan.status = "approved"
    plan.approved_by = str(admin["username"])
    plan.approved_at = datetime.now(UTC)
    plan.result_json = {
        **(plan.result_json or {}),
        "approval_digest": approval_digest,
    }
    task = request.app.state.task_engine.create_task(
        db,
        task_type="operation.execute",
        title=f"执行诊断方案：{plan.summary[:80]}",
        input_json={"plan_id": plan.id},
        idempotency_key=f"operation-plan:{plan.id}",
        commit=False,
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action="operation_plan.approve",
        resource_type="operation_plan",
        resource_id=plan.id,
    )
    db.commit()
    db.refresh(task)
    request.app.state.task_engine.notify()
    return serialize_task(task)


@router.post("/{plan_id}/reject")
def reject_plan(plan_id: str, db: DbSession, admin: CsrfAdmin) -> dict[str, Any]:
    plan = db.get(OperationPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Operation plan not found")
    if plan.status != "pending":
        raise HTTPException(status_code=409, detail=f"Plan is already {plan.status}")
    plan.status = "rejected"
    session_ids = db.scalars(
        select(OpsMessage.session_id).where(OpsMessage.operation_plan_id == plan.id)
    )
    for session_id in set(session_ids):
        session = db.get(OpsSession, session_id)
        if session is not None and session.status == "approval_required":
            session.status = "active"
    record_audit(
        db,
        actor=str(admin["username"]),
        action="operation_plan.reject",
        resource_type="operation_plan",
        resource_id=plan.id,
    )
    db.commit()
    return serialize_plan(plan)
