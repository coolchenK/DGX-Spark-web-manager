from datetime import UTC, datetime, timedelta

from app.api import tasks as task_api
from app.models import TaskRecord
from app.tasks.engine import TaskEngine, transition_task
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
