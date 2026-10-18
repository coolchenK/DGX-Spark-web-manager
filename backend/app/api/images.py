import base64
import json
import shutil
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import Field, ValidationError
from sqlalchemy import select

from app.api.tasks import serialize_task
from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.image_schema import MAX_IMAGE_BYTES, MAX_REQUEST_BYTES, ImageRequest
from app.models import Deployment, TaskRecord

router = APIRouter(prefix="/api/images", tags=["images"])


class CreateImageRequest(ImageRequest):
    deployment_id: str = Field(min_length=1, max_length=64)
    source_task_id: str | None = Field(default=None, max_length=64)


@router.get("")
def list_images(db: DbSession, _: Admin) -> list[dict]:
    rows = db.scalars(
        select(TaskRecord)
        .where(TaskRecord.type == "image.generate")
        .order_by(TaskRecord.created_at.desc())
        .limit(50)
    )
    return [
        {
            **serialize_task(task),
            "request": task.input_json.get("request", {}),
            "deployment_id": task.input_json.get("deployment_id"),
            "source_task_id": task.input_json.get("source_task_id"),
        }
        for task in rows
    ]


@router.post("", status_code=202)
async def create_image(request: Request, db: DbSession, admin: CsrfAdmin) -> dict:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "最多上传 3 张参考图，每张不超过 16 MiB")
    try:
        values = json.loads(body)
        if not isinstance(values, dict):
            raise ValueError("图片请求必须是对象")
        source_id = values.get("source_task_id")
        if source_id is not None:
            if not isinstance(source_id, str) or str(UUID(source_id)) != source_id:
                raise ValueError("源图任务 ID 无效")
            if values.get("images") or values.get("operation") not in {
                "refine", "texture", "upscale",
            }:
                raise ValueError("请从历史结果选择一张源图进行放大")
            source = db.get(TaskRecord, source_id)
            if not source or source.type != "image.generate" or source.status != "succeeded":
                raise HTTPException(404, "源图不存在或尚未生成完成")
            path = request.app.state.image_studio_service.directory(source.id) / "output.png"
            if not path.is_file() or path.is_symlink():
                raise HTTPException(404, "源图文件不存在")
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise HTTPException(413, "源图超过 16 MiB，请先缩小图片")
            values["images"] = [
                "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
            ]
        payload = CreateImageRequest.model_validate(values)
    except ValidationError as exc:
        # Do not reflect uploaded Base64 strings back into a validation error.
        message = "; ".join(str(item["msg"]) for item in exc.errors(include_input=False))
        raise HTTPException(422, message[:600]) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)[:200]) from exc
    deployment = db.get(Deployment, payload.deployment_id)
    if not deployment or deployment.runtime != "viggle" or not deployment.managed:
        raise HTTPException(404, "图片模型不存在")
    if deployment.status != "running":
        raise HTTPException(409, "请先启动图片模型")
    pending = db.scalar(
        select(TaskRecord.id)
        .where(TaskRecord.type == "image.generate", TaskRecord.status.in_(["queued", "running"]))
        .limit(4)
        .offset(3)
    )
    if pending is not None:
        raise HTTPException(429, "最多可排队 4 个生成任务，请等待当前任务完成")
    spec = ImageRequest.model_validate(
        payload.model_dump(exclude={"deployment_id", "source_task_id"})
    )
    task = request.app.state.task_engine.create_task(
        db,
        task_type="image.generate",
        title={
            "refine": "2K 精修", "texture": "4K 纹理增强",
            "upscale": "SUPIR 细节重建" if spec.sr_engine == "supir" else "Real-ESRGAN 快速超分",
        }.get(
            spec.operation, "图片编辑" if spec.images else "文生图",
        ),
        input_json={},
        commit=False,
    )
    service = request.app.state.image_studio_service
    try:
        headers = service.save_inputs(task.id, spec)
        task.input_json = {
            "deployment_id": deployment.id,
            "request": spec.model_dump(mode="json", exclude={"images"}),
            "image_headers": headers,
            "source_task_id": payload.source_task_id,
        }
        record_audit(
            db,
            actor=str(admin["username"]),
            action="image.generate",
            resource_type="task",
            resource_id=task.id,
            details={"deployment_id": deployment.id, "reference_count": len(headers)},
        )
        db.commit()
    except Exception:
        shutil.rmtree(service.directory(task.id), ignore_errors=True)
        raise
    request.app.state.task_engine.notify()
    return serialize_task(task)


@router.get("/{task_id}/output")
def image_output(task_id: str, request: Request, db: DbSession, _: Admin, download: bool = False):
    task = db.get(TaskRecord, task_id)
    if not task or task.type != "image.generate" or task.status != "succeeded":
        raise HTTPException(404, "图片不存在")
    path = request.app.state.image_studio_service.directory(task.id) / "output.png"
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "图片文件不存在")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"viggle-{task.id}.png" if download else None,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/{task_id}", status_code=204)
def delete_image(task_id: str, request: Request, db: DbSession, admin: CsrfAdmin):
    task = db.get(TaskRecord, task_id)
    if not task or task.type != "image.generate":
        raise HTTPException(404, "图片任务不存在")
    if task.status in {"running", "queued"}:
        raise HTTPException(409, "请先取消生成任务")
    shutil.rmtree(request.app.state.image_studio_service.directory(task.id), ignore_errors=True)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="image.delete",
        resource_type="task",
        resource_id=task.id,
    )
    db.delete(task)
    db.commit()
