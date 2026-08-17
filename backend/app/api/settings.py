from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, or_, select

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import (
    AuditEvent,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    SecretSetting,
    TaskRecord,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class HuggingFaceTokenUpdate(BaseModel):
    token: str | None = Field(default=None, max_length=4096)


class HistoryClearRequest(BaseModel):
    confirmation: Literal["清除历史记录"]


@router.get("")
def get_settings(request: Request, db: DbSession, _: Admin) -> dict[str, Any]:
    stored = db.get(SecretSetting, "huggingface_token")
    settings = request.app.state.settings
    return {
        "huggingface": {
            "token_configured": bool(stored or request.app.state.huggingface_service.token),
            "cache_dir": str(settings.hf_cache_dir),
        },
        "models": {"roots": [str(path) for path in settings.model_root_paths]},
        "runtimes": {
            "vllm": sorted(settings.vllm_images),
            "sglang": sorted(settings.sglang_images),
        },
    }


@router.patch("/huggingface")
def update_huggingface_token(
    payload: HuggingFaceTokenUpdate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    token = payload.token.strip() if payload.token else None
    stored = db.get(SecretSetting, "huggingface_token")
    if token:
        encrypted = request.app.state.secret_box.encrypt(token)
        if stored:
            stored.encrypted_value = encrypted
        else:
            db.add(SecretSetting(key="huggingface_token", encrypted_value=encrypted))
    elif stored:
        db.delete(stored)
    request.app.state.huggingface_service.set_token(token)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="settings.huggingface_token.update",
        resource_type="settings",
        details={"configured": bool(token)},
    )
    db.commit()
    return {"token_configured": bool(token)}


@router.delete("/alerts-diagnostics-history")
def clear_alerts_diagnostics_history(
    payload: HistoryClearRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    del payload  # Validation of the exact confirmation phrase is the authorization boundary.
    active_task_id = db.scalar(
        select(TaskRecord.id)
        .where(
            TaskRecord.type.in_(("ops.respond", "operation.execute")),
            or_(
                TaskRecord.status.in_(("queued", "running", "paused")),
                TaskRecord.cancel_requested.is_(True),
            ),
        )
        .with_for_update()
        .limit(1)
    )
    if active_task_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="存在正在执行或等待处理的运维任务，暂时不能清除历史记录",
        )
    unfinished_plan_id = db.scalar(
        select(OperationPlan.id)
        .where(OperationPlan.status.in_(("approved", "executing")))
        .with_for_update()
        .limit(1)
    )
    if unfinished_plan_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="存在已批准但尚未完成的操作计划，暂时不能清除历史记录",
        )

    failed_task_ids = select(TaskRecord.id).where(TaskRecord.status == "failed")
    related_audits = or_(
        AuditEvent.action.like("ops.%"),
        AuditEvent.action.like("operation_plan.%"),
        AuditEvent.resource_type.in_(
            ("ops_session", "ops_message", "ops_tool_run", "operation_plan")
        ),
        (AuditEvent.resource_type == "task")
        & AuditEvent.resource_id.in_(failed_task_ids),
    )
    audit_count = db.execute(
        delete(AuditEvent)
        .where(related_audits)
        .execution_options(synchronize_session=False)
    ).rowcount
    message_count = db.execute(
        delete(OpsMessage).execution_options(synchronize_session=False)
    ).rowcount
    tool_count = db.execute(
        delete(OpsToolRun).execution_options(synchronize_session=False)
    ).rowcount
    session_count = db.execute(
        delete(OpsSession).execution_options(synchronize_session=False)
    ).rowcount
    plan_count = db.execute(
        delete(OperationPlan).execution_options(synchronize_session=False)
    ).rowcount
    failed_task_count = db.execute(
        delete(TaskRecord)
        .where(TaskRecord.status == "failed")
        .execution_options(synchronize_session=False)
    ).rowcount
    deleted = {
        "failed_tasks": max(failed_task_count or 0, 0),
        "operation_plans": max(plan_count or 0, 0),
        "ops_sessions": max(session_count or 0, 0),
        "ops_messages": max(message_count or 0, 0),
        "ops_tool_runs": max(tool_count or 0, 0),
        "audit_events": max(audit_count or 0, 0),
    }
    record_audit(
        db,
        actor=str(admin["username"]),
        action="maintenance.history.clear",
        resource_type="maintenance",
        source_ip=request.client.host if request.client else None,
        details=deleted,
    )
    db.commit()
    return {"status": "cleared", "deleted": deleted}
