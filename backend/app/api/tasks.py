import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select

from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import TaskRecord

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def calculate_transfer_stats(
    task: TaskRecord,
    *,
    now: datetime | None = None,
) -> dict[str, float | int | None]:
    if not task.started_at or task.completed_bytes <= 0:
        return {"speed_bytes_per_second": None, "eta_seconds": None}
    started_at = task.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    ended_at = task.finished_at or now or datetime.now(UTC)
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=UTC)
    elapsed = max((ended_at - started_at).total_seconds(), 0)
    if elapsed == 0:
        return {"speed_bytes_per_second": None, "eta_seconds": None}
    speed = task.completed_bytes / elapsed
    remaining = max((task.total_bytes or 0) - task.completed_bytes, 0)
    eta = round(remaining / speed) if task.status == "running" and speed > 0 else None
    return {"speed_bytes_per_second": round(speed, 2), "eta_seconds": eta}


def serialize_task(task: TaskRecord) -> dict[str, Any]:
    return {
        "id": task.id,
        "type": task.type,
        "status": task.status,
        "title": task.title,
        "progress": task.progress,
        "completed_bytes": task.completed_bytes,
        "total_bytes": task.total_bytes,
        "result": task.result_json,
        "error": task.error,
        "log": task.log,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        **calculate_transfer_stats(task),
    }


@router.get("")
def list_tasks(
    db: DbSession,
    _: Admin,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    tasks = db.scalars(select(TaskRecord).order_by(desc(TaskRecord.created_at)).limit(limit))
    return [serialize_task(task) for task in tasks]


@router.get("/events")
async def task_events(request: Request, _: Admin) -> StreamingResponse:
    async def stream():
        last_payload = ""
        while not await request.is_disconnected():
            with request.app.state.database.session_factory() as db:
                tasks = db.scalars(
                    select(TaskRecord).order_by(desc(TaskRecord.created_at)).limit(50)
                )
                payload = json.dumps(
                    [serialize_task(task) for task in tasks], default=str, ensure_ascii=True
                )
            if payload != last_payload:
                yield f"event: tasks\ndata: {payload}\n\n"
                last_payload = payload
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/{task_id}")
def get_task(task_id: str, db: DbSession, _: Admin) -> dict[str, Any]:
    task = db.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return serialize_task(task)


@router.post("/{task_id}/pause")
def pause_task(task_id: str, request: Request, db: DbSession, _: CsrfAdmin) -> dict[str, Any]:
    task = db.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        request.app.state.task_engine.request_pause(db, task)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_task(task)


@router.post("/{task_id}/resume")
def resume_task(task_id: str, request: Request, db: DbSession, _: CsrfAdmin) -> dict[str, Any]:
    task = db.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        request.app.state.task_engine.resume(db, task)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_task(task)


@router.delete("/{task_id}", status_code=status.HTTP_202_ACCEPTED)
def cancel_task(task_id: str, request: Request, db: DbSession, _: CsrfAdmin) -> dict[str, Any]:
    task = db.get(TaskRecord, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        request.app.state.task_engine.request_cancel(db, task)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_task(task)

