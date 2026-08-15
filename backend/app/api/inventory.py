from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import select

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import Deployment, ModelAsset

router = APIRouter(prefix="/api", tags=["inventory"])


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
