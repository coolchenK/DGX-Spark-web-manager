from typing import Any

from fastapi import APIRouter, Query, Request, status
from pydantic import BaseModel, Field

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.tasks.huggingface import validate_repository_id

router = APIRouter(prefix="/api/huggingface", tags=["huggingface"])


class DownloadRequest(BaseModel):
    repository_id: str
    revision: str = "main"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


@router.get("/search")
def search_models(
    request: Request,
    _: Admin,
    query: str = Query(min_length=1, max_length=255),
    limit: int = Query(default=20, ge=1, le=50),
) -> list[dict[str, Any]]:
    return request.app.state.huggingface_service.search(query, limit)


@router.get("/models/{repository_id:path}")
def model_info(
    repository_id: str,
    request: Request,
    _: Admin,
    revision: str = "main",
) -> dict[str, Any]:
    return request.app.state.huggingface_service.info(repository_id, revision)


@router.post("/downloads", status_code=status.HTTP_202_ACCEPTED)
def create_download(
    payload: DownloadRequest,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    repository_id = validate_repository_id(payload.repository_id)
    task = request.app.state.task_engine.create_task(
        db,
        task_type="model.download",
        title=f"下载 {repository_id}",
        input_json={
            "repository_id": repository_id,
            "revision": payload.revision,
            "include": payload.include,
            "exclude": payload.exclude,
        },
        idempotency_key=f"download:{repository_id}:{payload.revision}",
    )
    record_audit(
        db,
        actor=str(admin["username"]),
        action="model.download.create",
        resource_type="task",
        resource_id=task.id,
        details={"repository_id": repository_id, "revision": payload.revision},
    )
    db.commit()
    return serialize_task(task)
