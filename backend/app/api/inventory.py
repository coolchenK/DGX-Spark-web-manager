from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import Deployment, ModelAsset

router = APIRouter(prefix="/api", tags=["inventory"])


class ModelDeleteRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=255)


@router.get("/models")
def list_models(db: DbSession, _: Admin) -> list[dict[str, Any]]:
    items = db.scalars(select(ModelAsset).order_by(ModelAsset.name))
    return [
        {
            "id": item.id,
            "name": item.name,
            "alias": item.alias,
            "source": item.source,
            "repository_id": item.repository_id,
            "revision": item.revision,
            "commit_hash": item.commit_hash,
            "local_path": item.local_path,
            "format": item.format,
            "quantization": item.quantization,
            "parameter_count": item.parameter_count,
            "size_bytes": item.size_bytes,
            "status": item.status,
            "capabilities": item.capabilities,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        }
        for item in items
    ]


@router.delete("/models/{model_id}", status_code=status.HTTP_202_ACCEPTED)
def delete_model(
    model_id: str,
    payload: ModelDeleteRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    asset = db.get(ModelAsset, model_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if payload.confirmation != asset.name:
        raise HTTPException(
            status_code=422,
            detail="Confirmation does not match model name",
        )

    references = request.app.state.model_lifecycle_service.references(db, model_id)
    if references:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "model_in_use",
                "references": [asdict(reference) for reference in references],
            },
        )
    if asset.status == "deleting":
        raise HTTPException(
            status_code=409,
            detail="Model deletion is already in progress",
        )

    source = asset.source
    repository_id = asset.repository_id
    try:
        task = request.app.state.task_engine.create_task(
            db,
            task_type="model.delete",
            title=f"删除模型 {asset.name}",
            input_json={"model_id": model_id},
            idempotency_key=f"model:{model_id}:delete",
            commit=False,
        )
        record_audit(
            db,
            actor=str(admin["username"]),
            action="model.delete.create",
            resource_type="model",
            resource_id=model_id,
            details={
                "task_id": task.id,
                "source": source,
                "repository_id": repository_id,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(task)
    request.app.state.task_engine.notify()
    return serialize_task(task)


@router.get("/deployments")
def list_deployments(db: DbSession, _: Admin) -> list[dict[str, Any]]:
    items = db.scalars(select(Deployment).order_by(Deployment.name))
    return [
        {
            "id": item.id,
            "name": item.name,
            "model_id": item.model_id,
            "runtime": item.runtime,
            "container_id": item.container_id,
            "container_name": item.container_name,
            "endpoint_url": item.endpoint_url,
            "api_model_name": item.api_model_name,
            "status": item.status,
            "health": item.health,
            "managed": item.managed,
            "image": item.image,
            "port": item.port,
            "benchmark_status": item.benchmark_status,
            "benchmark_tps": item.benchmark_tps,
            "benchmark_completion_tokens": item.benchmark_completion_tokens,
            "benchmark_duration_seconds": item.benchmark_duration_seconds,
            "benchmark_tested_at": item.benchmark_tested_at,
            "benchmark_error": item.benchmark_error,
            "config": item.config,
            "capabilities": item.capabilities,
            "last_checked_at": item.last_checked_at,
        }
        for item in items
    ]


@router.post("/discovery/scan")
def scan_inventory(
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, int | str | None]:
    result = request.app.state.discovery_service.scan_all(db)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="inventory.scan",
        resource_type="system",
        details=result,
    )
    db.commit()
    return result
