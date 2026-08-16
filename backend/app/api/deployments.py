from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import or_, select

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import Deployment, Provider
from app.runtime.base import DeploymentSpec
from app.services.deployment_recommendations import RecommendationRequest

router = APIRouter(prefix="/api/deployments", tags=["deployments"])

DISCOVERED_DELETE_CONFIRMATION_ERROR = (
    "To uninstall a discovered service, confirm_container_name must exactly "
    "match its container name"
)


class DeploymentActionRequest(BaseModel):
    confirm_container_name: str | None = Field(default=None, max_length=255)


@router.post("/recommendations")
def recommend_deployment(
    payload: RecommendationRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
    refresh_ai: bool = Query(default=False),
) -> dict[str, Any]:
    provider = db.get(Provider, payload.provider_id) if payload.provider_id else None
    provider_error: tuple[int, str, str] | None = None
    if payload.provider_id and provider is None:
        provider_error = (404, "AI provider was not found", "provider_not_found")
    elif provider is not None and (not provider.enabled or provider.last_test_status == "failed"):
        provider_error = (409, "AI provider is unavailable", "provider_unavailable")
    if provider_error is not None:
        status_code, detail, audit_status = provider_error
        record_audit(
            db,
            actor=str(admin["username"]),
            action="deployment.recommendation.generate",
            resource_type="model",
            resource_id=payload.model_id,
            outcome="failed",
            details={
                "runtime": payload.runtime,
                "status": audit_status,
                "provider_used": False,
                "refresh": refresh_ai,
            },
        )
        db.commit()
        raise HTTPException(status_code=status_code, detail=detail)
    result = request.app.state.deployment_recommendation_service.recommend(
        db=db,
        model_id=payload.model_id,
        runtime=payload.runtime,
        image=payload.image,
        provider=provider,
        refresh_ai=refresh_ai,
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action="deployment.recommendation.generate",
        resource_type="model",
        resource_id=payload.model_id,
        outcome="failed" if result.status == "unavailable" else "success",
        details={
            "runtime": payload.runtime,
            "status": result.status,
            "provider_used": provider is not None,
            "refresh": refresh_ai,
        },
    )
    db.commit()
    return result.model_dump(mode="json")


@router.post("/preview")
def preview_deployment(
    spec: DeploymentSpec,
    request: Request,
    db: DbSession,
    _: Admin,
    deployment_id: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return request.app.state.deployment_service.preview(
            db,
            spec,
            exclude_deployment_id=deployment_id,
        )
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
        request.app.state.deployment_service.preview(db, spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    conflict = db.scalar(
        select(Deployment).where(
            or_(Deployment.name == spec.name, Deployment.api_model_name == spec.api_model_name)
        )
    )
    if conflict:
        raise HTTPException(
            status_code=409,
            detail=(
                "A deployment already uses this name or API model name; edit or clone it instead"
            ),
        )
    task = request.app.state.task_engine.create_task(
        db,
        task_type="deployment.create",
        title=f"部署 {spec.name}",
        input_json=spec.model_dump(mode="json"),
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


@router.patch("/{deployment_id}", status_code=status.HTTP_202_ACCEPTED)
def update_deployment(
    deployment_id: str,
    spec: DeploymentSpec,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    deployment = db.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if not deployment.managed:
        raise HTTPException(status_code=409, detail="Discovered containers cannot be edited")
    try:
        request.app.state.deployment_service.preview(
            db,
            spec,
            exclude_deployment_id=deployment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    conflict = db.scalar(
        select(Deployment).where(
            Deployment.id != deployment_id,
            or_(Deployment.name == spec.name, Deployment.api_model_name == spec.api_model_name),
        )
    )
    if conflict:
        raise HTTPException(
            status_code=409,
            detail="Another deployment already uses this name or API model name",
        )
    task = request.app.state.task_engine.create_task(
        db,
        task_type="deployment.update",
        title=f"更新部署 {deployment.name}",
        input_json={"deployment_id": deployment_id, "spec": spec.model_dump(mode="json")},
        idempotency_key=f"deployment:{deployment_id}:update",
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action="deployment.update",
        resource_type="deployment",
        resource_id=deployment_id,
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
    payload: DeploymentActionRequest | None = None,
) -> dict[str, Any]:
    deployment = db.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if (
        action == "delete"
        and not deployment.managed
        and (
            payload is None
            or payload.confirm_container_name is None
            or payload.confirm_container_name != deployment.container_name
        )
    ):
        raise HTTPException(status_code=422, detail=DISCOVERED_DELETE_CONFIRMATION_ERROR)
    task = request.app.state.task_engine.create_task(
        db,
        task_type="deployment.action",
        title=(
            f"卸载服务 {deployment.name}"
            if action == "delete"
            else f"{action} {deployment.name}"
        ),
        input_json={
            "deployment_id": deployment_id,
            "action": action,
            "expected_container_id": deployment.container_id,
            "expected_container_name": deployment.container_name,
        },
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
