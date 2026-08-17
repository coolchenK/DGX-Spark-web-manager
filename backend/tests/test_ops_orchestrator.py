from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from alembic import command
from alembic.config import Config
from app.db import Database
from app.models import (
    Deployment,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
)
from sqlalchemy import delete, event, inspect, select, text

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _database(path) -> Database:
    database = Database(f"sqlite:///{path}")

    @event.listens_for(database.engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return database


def _provider(name: str) -> Provider:
    return Provider(
        name=name,
        base_url=f"https://{name}.example/v1",
        default_model="test-model",
        encrypted_api_key="encrypted",
    )


def _deployment(name: str) -> Deployment:
    return Deployment(
        name=name,
        runtime="vllm",
        endpoint_url=f"http://{name}:8000/v1",
        api_model_name=f"{name}-model",
    )


def test_ops_session_persists_ordered_messages_and_tool_runs(tmp_path):
    database = _database(tmp_path / "ops.db")
    database.create_schema()
    base_time = datetime(2026, 8, 17, tzinfo=UTC)

    with database.session_factory() as db:
        session = OpsSession(title="Repair gateway", requested_by="admin")
        db.add(session)
        db.flush()
        session_id = session.id
        db.add_all(
            [
                OpsMessage(
                    session_id=session_id,
                    role="assistant",
                    content="Second",
                    created_at=base_time + timedelta(seconds=2),
                ),
                OpsMessage(
                    session_id=session_id,
                    role="user",
                    content="First",
                    created_at=base_time,
                ),
                OpsToolRun(
                    session_id=session_id,
                    tool_name="host.memory",
                    risk="read_only",
                    status="succeeded",
                    arguments_json={},
                    result_json={"available": 1},
                    created_at=base_time + timedelta(seconds=2),
                ),
                OpsToolRun(
                    session_id=session_id,
                    tool_name="host.health",
                    risk="read_only",
                    status="succeeded",
                    arguments_json={},
                    result_json={"healthy": True},
                    created_at=base_time,
                ),
            ]
        )
        db.commit()

    with database.session_factory() as db:
        persisted = db.get(OpsSession, session_id)
        assert persisted is not None
        assert [message.content for message in persisted.messages] == ["First", "Second"]
        assert [tool.tool_name for tool in persisted.tool_runs] == [
            "host.health",
            "host.memory",
        ]
        db.delete(persisted)
        db.commit()

    with database.session_factory() as db:
        assert db.scalars(select(OpsMessage)).all() == []
        assert db.scalars(select(OpsToolRun)).all() == []

    database.dispose()


def test_ops_foreign_keys_set_null_when_optional_parents_are_deleted(tmp_path):
    database = _database(tmp_path / "foreign-keys.db")
    database.create_schema()

    with database.session_factory() as db:
        provider = _provider("provider")
        deployment = _deployment("deployment")
        plan = OperationPlan(summary="Repair", diagnosis="Broken")
        db.add_all([provider, deployment, plan])
        db.flush()
        session = OpsSession(
            title="Repair gateway",
            provider_id=provider.id,
            deployment_id=deployment.id,
            requested_by="admin",
        )
        db.add(session)
        db.flush()
        message = OpsMessage(
            session_id=session.id,
            role="assistant",
            content="Approval required",
            operation_plan_id=plan.id,
        )
        db.add(message)
        db.commit()
        session_id = session.id
        message_id = message.id

        db.execute(delete(Provider).where(Provider.id == provider.id))
        db.execute(delete(Deployment).where(Deployment.id == deployment.id))
        db.execute(delete(OperationPlan).where(OperationPlan.id == plan.id))
        db.commit()

    with database.session_factory() as db:
        persisted_session = db.get(OpsSession, session_id)
        persisted_message = db.get(OpsMessage, message_id)
        assert persisted_session is not None
        assert persisted_session.provider_id is None
        assert persisted_session.deployment_id is None
        assert persisted_message is not None
        assert persisted_message.operation_plan_id is None

    database.dispose()


def test_json_defaults_are_isolated_between_rows(tmp_path):
    database = _database(tmp_path / "defaults.db")
    database.create_schema()

    with database.session_factory() as db:
        first_provider = _provider("first")
        second_provider = _provider("second")
        session = OpsSession(title="Defaults", requested_by="admin")
        db.add_all([first_provider, second_provider, session])
        db.flush()
        first_message = OpsMessage(session_id=session.id, role="user", content="One")
        second_message = OpsMessage(session_id=session.id, role="user", content="Two")
        first_tool = OpsToolRun(session_id=session.id, tool_name="host.health")
        second_tool = OpsToolRun(session_id=session.id, tool_name="host.memory")
        db.add_all([first_message, second_message, first_tool, second_tool])
        db.flush()

        assert first_provider.last_test_result is not second_provider.last_test_result
        assert first_message.metadata_json is not second_message.metadata_json
        assert first_tool.arguments_json is not second_tool.arguments_json
        assert first_tool.result_json is not second_tool.result_json

        first_provider.last_test_result["ok"] = True
        first_message.metadata_json["source"] = "test"
        first_tool.arguments_json["tail"] = 10
        first_tool.result_json["ok"] = True

        assert second_provider.last_test_result == {}
        assert second_message.metadata_json == {}
        assert second_tool.arguments_json == {}
        assert second_tool.result_json == {}

    database.dispose()


def _alembic_config(database_url: str) -> Config:
    config = Config(str(REPOSITORY_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location", str(REPOSITORY_ROOT / "backend" / "migrations")
    )
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_fresh_database_upgrades_directly_to_head(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'fresh-migration.db'}"
    monkeypatch.setenv("DGX_DATABASE_URL", database_url)
    command.upgrade(_alembic_config(database_url), "head")

    database = _database(tmp_path / "fresh-migration.db")
    with database.engine.connect() as connection:
        inspector = inspect(connection)
        assert {"ops_sessions", "ops_messages", "ops_tool_runs"} <= set(
            inspector.get_table_names()
        )
        assert "last_test_result" in {
            column["name"] for column in inspector.get_columns("providers")
        }
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            "20260817_0002"
        )
    database.dispose()


def test_ops_session_migration_preserves_existing_provider_data(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("DGX_DATABASE_URL", database_url)
    config = _alembic_config(database_url)

    command.upgrade(config, "20260816_0001")
    database = _database(tmp_path / "migration.db")
    with database.engine.connect() as connection:
        inspector = inspect(connection)
        assert not {"ops_sessions", "ops_messages", "ops_tool_runs"} & set(
            inspector.get_table_names()
        )
        assert "last_test_result" not in {
            column["name"] for column in inspector.get_columns("providers")
        }
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO providers "
                "(id, name, base_url, default_model, encrypted_api_key, timeout_seconds, "
                "headers, enabled, created_at, updated_at) "
                "VALUES (:id, :name, :base_url, :model, :key, 60, '{}', 1, :now, :now)"
            ),
            {
                "id": "existing-provider",
                "name": "Existing",
                "base_url": "https://existing.example/v1",
                "model": "existing-model",
                "key": "encrypted",
                "now": "2026-08-17 00:00:00",
            },
        )

    command.upgrade(config, "head")
    with database.engine.connect() as connection:
        inspector = inspect(connection)
        assert {"ops_sessions", "ops_messages", "ops_tool_runs"} <= set(
            inspector.get_table_names()
        )
        assert "last_test_result" in {
            column["name"] for column in inspector.get_columns("providers")
        }
        row = connection.execute(
            text("SELECT name, last_test_result FROM providers WHERE id='existing-provider'")
        ).one()
        assert row.name == "Existing"
        assert row.last_test_result == "{}"

    command.downgrade(config, "20260816_0001")
    with database.engine.connect() as connection:
        inspector = inspect(connection)
        assert not {"ops_sessions", "ops_messages", "ops_tool_runs"} & set(
            inspector.get_table_names()
        )
        assert "last_test_result" not in {
            column["name"] for column in inspector.get_columns("providers")
        }
        assert connection.execute(
            text("SELECT name FROM providers WHERE id='existing-provider'")
        ).scalar_one() == "Existing"

    command.upgrade(config, "head")
    with database.engine.connect() as connection:
        inspector = inspect(connection)
        assert {"ops_sessions", "ops_messages", "ops_tool_runs"} <= set(
            inspector.get_table_names()
        )
        assert connection.execute(
            text("SELECT last_test_result FROM providers WHERE id='existing-provider'")
        ).scalar_one() == "{}"

    database.dispose()
