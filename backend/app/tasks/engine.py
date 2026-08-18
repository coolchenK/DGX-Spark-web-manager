from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import TaskRecord

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "queued": {"running", "paused", "cancelled"},
    "running": {"succeeded", "failed", "paused", "cancelled", "queued"},
    "paused": {"queued", "cancelled"},
    "succeeded": set(),
    "failed": {"queued"},
    "cancelled": {"queued"},
}


class TaskPaused(Exception):
    pass


class TaskCancelled(Exception):
    pass


def transition_task(task: TaskRecord, target: str) -> None:
    allowed = ALLOWED_TRANSITIONS.get(task.status, set())
    if target not in allowed:
        raise ValueError(f"Invalid task transition: {task.status} -> {target}")
    task.status = target
    now = datetime.now(UTC)
    if target == "running":
        task.started_at = now
        task.finished_at = None
        task.error = None
    elif target in TERMINAL_STATES:
        task.finished_at = now


class TaskContext:
    def __init__(self, session_factory: sessionmaker[Session], task_id: str):
        self.session_factory = session_factory
        self.task_id = task_id

    def update(
        self,
        *,
        progress: float | None = None,
        completed_bytes: int | None = None,
        total_bytes: int | None = None,
        message: str | None = None,
    ) -> None:
        with self.session_factory() as db:
            task = db.get(TaskRecord, self.task_id)
            if task is None:
                return
            if progress is not None:
                task.progress = max(0, min(100, progress))
            if total_bytes is not None:
                task.total_bytes = max(0, total_bytes)
            if completed_bytes is not None:
                completed = max(0, completed_bytes)
                if task.total_bytes is not None:
                    completed = min(completed, task.total_bytes)
                task.completed_bytes = completed
            if message:
                task.log = f"{task.log}{message.rstrip()}\n"[-100_000:]
            db.commit()

    def check_control(self) -> None:
        with self.session_factory() as db:
            task = db.get(TaskRecord, self.task_id)
            if task is None:
                raise TaskCancelled()
            control = (task.result_json or {}).get("control")
            if control == "pause":
                raise TaskPaused()
            if task.cancel_requested or control == "cancel":
                raise TaskCancelled()


TaskHandler = Callable[[TaskContext, dict[str, Any]], dict[str, Any] | None]


class TaskEngine:
    # Downloads are intentionally isolated from service mutations. A single
    # download worker prevents competing downloads from saturating storage,
    # while the control worker keeps deployment/uninstall operations ordered.
    DOWNLOAD_TASK_TYPES = frozenset({"model.download"})

    def __init__(self, session_factory: sessionmaker[Session] | Callable[[], Any]):
        self.session_factory = session_factory
        self.handlers: dict[str, TaskHandler] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []

    def register(self, task_type: str, handler: TaskHandler) -> None:
        self.handlers[task_type] = handler

    def notify(self) -> None:
        self._wake.set()

    def create_task(
        self,
        db: Session,
        *,
        task_type: str,
        title: str,
        input_json: dict[str, Any],
        idempotency_key: str | None = None,
        commit: bool = True,
    ) -> TaskRecord:
        if idempotency_key:
            existing = db.scalar(
                select(TaskRecord).where(TaskRecord.idempotency_key == idempotency_key)
            )
            if existing and existing.status not in TERMINAL_STATES:
                return existing
            if existing:
                existing.idempotency_key = None
                db.flush()
        task = TaskRecord(
            type=task_type,
            title=title,
            input_json=input_json,
            idempotency_key=idempotency_key,
        )
        db.add(task)
        try:
            if commit:
                db.commit()
                db.refresh(task)
                self.notify()
            else:
                db.flush()
        except IntegrityError:
            db.rollback()
            if idempotency_key:
                existing = db.scalar(
                    select(TaskRecord).where(
                        TaskRecord.idempotency_key == idempotency_key
                    )
                )
                if existing and existing.status not in TERMINAL_STATES:
                    return existing
            raise
        return task

    def recover_interrupted(self) -> int:
        count = 0
        with self.session_factory() as db:
            tasks = list(db.scalars(select(TaskRecord).where(TaskRecord.status == "running")))
            for task in tasks:
                transition_task(task, "queued")
                task.log = f"{task.log}Recovered after manager restart.\n"
                task.cancel_requested = False
                task.result_json = {}
                count += 1
            db.commit()
        return count

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self.recover_interrupted()
        self._stop.clear()
        self._threads = []
        for lane in ("control", "download"):
            thread = threading.Thread(
                target=self._run,
                args=(lane,),
                name=f"dgx-task-worker-{lane}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        self._thread = self._threads[0]

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=5)

    def _claim(self, *, lane: str) -> tuple[str, str, dict[str, Any]] | None:
        with self.session_factory() as db:
            statement = select(TaskRecord).where(TaskRecord.status == "queued")
            if lane == "download":
                statement = statement.where(
                    TaskRecord.type.in_(self.DOWNLOAD_TASK_TYPES)
                )
            else:
                statement = statement.where(
                    ~TaskRecord.type.in_(self.DOWNLOAD_TASK_TYPES)
                )
            task = db.scalar(statement.order_by(TaskRecord.created_at))
            if task is None:
                return None
            transition_task(task, "running")
            task.log = f"{task.log}Task started.\n"
            db.commit()
            return task.id, task.type, dict(task.input_json)

    def _finish(
        self,
        task_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.session_factory() as db:
            task = db.get(TaskRecord, task_id)
            if task is None:
                return
            transition_task(task, status)
            task.result_json = result or {}
            task.error = error
            task.progress = 100 if status == "succeeded" else task.progress
            task.log = f"{task.log}Task {status}.\n"
            db.commit()

    def _run(self, lane: str) -> None:
        while not self._stop.is_set():
            claimed = self._claim(lane=lane)
            if claimed is None:
                self._wake.wait(timeout=1)
                self._wake.clear()
                continue
            task_id, task_type, payload = claimed
            handler = self.handlers.get(task_type)
            if handler is None:
                self._finish(task_id, "failed", error=f"No handler registered for {task_type}")
                continue
            context = TaskContext(self.session_factory, task_id)
            try:
                result = handler(context, payload)
            except TaskPaused:
                self._finish(task_id, "paused")
            except TaskCancelled:
                self._finish(task_id, "cancelled")
            except Exception as exc:  # worker boundary records sanitized exception text
                self._finish(task_id, "failed", error=str(exc)[:4000])
            else:
                self._finish(task_id, "succeeded", result=result)

    def request_pause(self, db: Session, task: TaskRecord) -> None:
        if task.status == "queued":
            transition_task(task, "paused")
        elif task.status == "running":
            task.result_json = {"control": "pause"}
        else:
            raise ValueError(f"Task in {task.status} cannot be paused")
        db.commit()

    def resume(self, db: Session, task: TaskRecord) -> None:
        transition_task(task, "queued")
        task.cancel_requested = False
        task.result_json = {}
        db.commit()
        self.notify()

    def request_cancel(self, db: Session, task: TaskRecord) -> None:
        if task.status in {"queued", "paused"}:
            transition_task(task, "cancelled")
        elif task.status == "running":
            task.cancel_requested = True
            task.result_json = {"control": "cancel"}
        else:
            raise ValueError(f"Task in {task.status} cannot be cancelled")
        db.commit()
