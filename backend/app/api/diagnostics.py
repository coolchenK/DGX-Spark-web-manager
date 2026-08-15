from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import OperationPlan, Provider

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


class DiagnosticRequest(BaseModel):
    provider_id: str
    deployment_id: str | None = None
    prompt: str = Field(min_length=1, max_length=10_000)


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


@router.get("")
def list_plans(db: DbSession, _: Admin) -> list[dict[str, Any]]:
    plans = db.scalars(select(OperationPlan).order_by(desc(OperationPlan.created_at)).limit(100))
    return [serialize_plan(plan) for plan in plans]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_diagnostic(
    payload: DiagnosticRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    provider = db.get(Provider, payload.provider_id)
    if not provider or not provider.enabled:
        raise HTTPException(status_code=404, detail="Enabled provider not found")
    try:
        result = request.app.state.diagnostic_service.diagnose(
            db,
            provider=provider,
            prompt=payload.prompt,
            deployment_id=payload.deployment_id,
        )
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=502, detail=f"Diagnostic provider failed: {exc}") from exc
    plan = OperationPlan(
        provider_id=provider.id,
        deployment_id=payload.deployment_id,
        summary=result["summary"],
        diagnosis=result["diagnosis"],
        risk=result["risk"],
        steps=result["steps"],
        requested_by=str(admin["username"]),
    )
    db.add(plan)
    db.flush()
    record_audit(
        db,
        actor=str(admin["username"]),
        action="diagnostic.create",
        resource_type="operation_plan",
        resource_id=plan.id,
        details={"provider_id": provider.id, "deployment_id": payload.deployment_id},
    )
    db.commit()
    db.refresh(plan)
    return serialize_plan(plan)


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
    plan.status = "approved"
    plan.approved_by = str(admin["username"])
    plan.approved_at = datetime.now(UTC)
    db.commit()
    task = request.app.state.task_engine.create_task(
        db,
        task_type="operation.execute",
        title=f"执行诊断方案：{plan.summary[:80]}",
        input_json={"plan_id": plan.id},
        idempotency_key=f"operation-plan:{plan.id}",
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action="operation_plan.approve",
        resource_type="operation_plan",
        resource_id=plan.id,
    )
    db.commit()
    return serialize_task(task)


@router.post("/{plan_id}/reject")
def reject_plan(plan_id: str, db: DbSession, admin: CsrfAdmin) -> dict[str, Any]:
    plan = db.get(OperationPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Operation plan not found")
    if plan.status != "pending":
        raise HTTPException(status_code=409, detail=f"Plan is already {plan.status}")
    plan.status = "rejected"
    record_audit(
        db,
        actor=str(admin["username"]),
        action="operation_plan.reject",
        resource_type="operation_plan",
        resource_id=plan.id,
    )
    db.commit()
    return serialize_plan(plan)
