import threading
from datetime import UTC, datetime, timedelta
from types import MethodType

from app.api import tasks as task_api
from app.models import TaskRecord
from app.tasks.engine import TaskContext, TaskEngine, transition_task
from sqlalchemy import select


def test_task_creation_is_idempotent(settings):
    engine = TaskEngine(lambda: None)

    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        first = engine.create_task(
            db,
            task_type="model.download",
            title="Download org/model",
            input_json={"repository_id": "org/model"},
            idempotency_key="download:org/model:main",
        )
        second = engine.create_task(
            db,
            task_type="model.download",
            title="Download org/model again",
            input_json={"repository_id": "org/model"},
            idempotency_key="download:org/model:main",
        )

        assert first.id == second.id
        assert len(list(db.scalars(select(TaskRecord)))) == 1


def test_terminal_task_releases_idempotency_key_for_a_new_run(settings):
    engine = TaskEngine(lambda: None)

    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        first = engine.create_task(
            db,
            task_type="model.download",
            title="Download org/model",
            input_json={"repository_id": "org/model"},
            idempotency_key="download:org/model:main",
        )
        first.status = "succeeded"
        db.commit()

        second = engine.create_task(
            db,
            task_type="model.download",
            title="Download org/model again",
            input_json={"repository_id": "org/model"},
            idempotency_key="download:org/model:main",
        )

        assert first.id != second.id
        assert first.idempotency_key is None
        assert second.idempotency_key == "download:org/model:main"
        assert len(list(db.scalars(select(TaskRecord)))) == 2


def test_task_creation_can_defer_commit(settings):
    engine = TaskEngine(lambda: None)

    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        task = engine.create_task(
            db,
            task_type="model.delete",
            title="Delete model",
            input_json={"model_id": "model-1"},
            idempotency_key="model:model-1:delete",
            commit=False,
        )

        assert task.id is not None
        with database.session_factory() as other_db:
            assert other_db.get(TaskRecord, task.id) is None

        db.rollback()

    with database.session_factory() as db:
        assert list(db.scalars(select(TaskRecord))) == []


def test_concurrent_task_creation_reuses_winning_idempotency_key(settings):
    engine = TaskEngine(lambda: None)

    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    both_lookups_finished = threading.Barrier(2)
    winner_committed = threading.Event()
    task_ids: list[str | None] = [None, None]
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        with database.session_factory() as db:
            original_scalar = db.scalar
            first_lookup = True

            def synchronized_scalar(self, statement, *args, **kwargs):
                nonlocal first_lookup
                result = original_scalar(statement, *args, **kwargs)
                if first_lookup:
                    first_lookup = False
                    assert result is None
                    self.rollback()
                    both_lookups_finished.wait(timeout=5)
                    if index == 1:
                        assert winner_committed.wait(timeout=5)
                return result

            db.scalar = MethodType(synchronized_scalar, db)
            try:
                task = engine.create_task(
                    db,
                    task_type="model.delete",
                    title="Delete model",
                    input_json={"model_id": "model-1"},
                    idempotency_key="model:model-1:delete",
                )
                task_ids[index] = task.id
            except BaseException as exc:
                errors.append(exc)
            finally:
                if index == 0:
                    winner_committed.set()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert task_ids[0] is not None
    assert task_ids[0] == task_ids[1]
    with database.session_factory() as db:
        tasks = list(
            db.scalars(
                select(TaskRecord).where(
                    TaskRecord.idempotency_key == "model:model-1:delete"
                )
            )
        )
        assert [task.id for task in tasks] == [task_ids[0]]


def test_running_tasks_are_recovered_to_queue(settings):
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        task = TaskRecord(type="scan", title="Scan", status="running")
        db.add(task)
        db.commit()

    engine = TaskEngine(database.session_factory)
    assert engine.recover_interrupted() == 1

    with database.session_factory() as db:
        recovered = db.get(TaskRecord, task.id)
        assert recovered.status == "queued"
        assert "Recovered after manager restart" in recovered.log


def test_task_transition_rejects_invalid_state_change():
    task = TaskRecord(type="scan", title="Scan", status="succeeded")

    try:
        transition_task(task, "running")
    except ValueError as exc:
        assert "succeeded -> running" in str(exc)
    else:
        raise AssertionError("Expected invalid transition to raise")


def test_transfer_stats_report_speed_and_remaining_time():
    started = datetime(2026, 8, 16, tzinfo=UTC)
    task = TaskRecord(
        type="model.download",
        title="Download",
        status="running",
        started_at=started,
        completed_bytes=1_000,
        total_bytes=2_000,
    )
    calculate = getattr(task_api, "calculate_transfer_stats", lambda *_args, **_kwargs: {})

    assert calculate(task, now=started + timedelta(seconds=10)) == {
        "speed_bytes_per_second": 100.0,
        "eta_seconds": 10,
    }


def test_task_progress_bytes_are_capped_at_known_total(settings):
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        task = TaskRecord(type="model.download", title="Download", total_bytes=100)
        db.add(task)
        db.commit()
        task_id = task.id

    TaskContext(database.session_factory, task_id).update(completed_bytes=129)

    with database.session_factory() as db:
        assert db.get(TaskRecord, task_id).completed_bytes == 100
