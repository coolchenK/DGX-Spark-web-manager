from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import Deployment
from app.runtime.base import DeploymentSpec

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


@router.post("/preview")
def preview_deployment(
    spec: DeploymentSpec,
    request: Request,
    _: Admin,
) -> dict[str, Any]:
    try:
        return request.app.state.deployment_service.preview(spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_deployment(
    spec: DeploymentSpec,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    try:
        request.app.state.deployment_service.preview(spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    task = request.app.state.task_engine.create_task(
        db,
        task_type="deployment.create",
        title=f"部署 {spec.name}",
        input_json=spec.model_dump(),
        idempotency_key=f"deployment:create:{spec.api_model_name}",
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action="deployment.create",
        resource_type="task",
        resource_id=task.id,
        details={"name": spec.name, "runtime": spec.runtime, "port": spec.port},
    )
    db.commit()
    return serialize_task(task)


@router.post("/{deployment_id}/{action}", status_code=status.HTTP_202_ACCEPTED)
def deployment_action(
    deployment_id: str,
    action: Literal["start", "stop", "restart", "delete"],
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    deployment = db.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if action == "delete" and not deployment.managed:
        raise HTTPException(status_code=409, detail="Discovered containers cannot be deleted")
    task = request.app.state.task_engine.create_task(
        db,
        task_type="deployment.action",
        title=f"{action} {deployment.name}",
        input_json={"deployment_id": deployment_id, "action": action},
        idempotency_key=f"deployment:{deployment_id}:{action}",
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action=f"deployment.{action}",
        resource_type="deployment",
        resource_id=deployment_id,
    )
    db.commit()
    return serialize_task(task)


@router.get("/{deployment_id}/logs")
def deployment_logs(
    deployment_id: str,
    request: Request,
    db: DbSession,
    _: Admin,
    tail: int = Query(default=500, ge=1, le=5000),
) -> dict[str, str]:
    deployment = db.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    try:
        return {"logs": request.app.state.deployment_service.logs(deployment, tail)}
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

