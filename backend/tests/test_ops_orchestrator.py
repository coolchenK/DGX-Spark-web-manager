from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from app.db import Database
from app.models import (
    Deployment,
    ModelAsset,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
)
from sqlalchemy import delete, inspect, select, text

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _database(path) -> Database:
    return Database(f"sqlite:///{path}")


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


def _model(name: str, path: str) -> ModelAsset:
    return ModelAsset(name=name, local_path=path)


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

    with database.session_factory() as db:
        persisted = db.get(OpsSession, session_id)
        assert persisted is not None
        assert "messages" in inspect(persisted).unloaded
        assert "tool_runs" in inspect(persisted).unloaded
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
        model = _model("model", str(tmp_path / "model"))
        model_deployment = _deployment("model-deployment")
        model_deployment.model = model
        deployment = _deployment("deployment")
        plan = OperationPlan(
            provider_id=provider.id,
            deployment_id=deployment.id,
            summary="Repair",
            diagnosis="Broken",
        )
        db.add_all([provider, model_deployment, deployment])
        db.flush()
        plan.provider_id = provider.id
        plan.deployment_id = deployment.id
        db.add(plan)
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
        model_deployment_id = model_deployment.id
        plan_id = plan.id

        db.execute(delete(ModelAsset).where(ModelAsset.id == model.id))
        db.execute(delete(Provider).where(Provider.id == provider.id))
        db.execute(delete(Deployment).where(Deployment.id == deployment.id))
        db.commit()

    with database.session_factory() as db:
        persisted_session = db.get(OpsSession, session_id)
        persisted_message = db.get(OpsMessage, message_id)
        persisted_deployment = db.get(Deployment, model_deployment_id)
        persisted_plan = db.get(OperationPlan, plan_id)
        assert persisted_session is not None
        assert persisted_session.provider_id is None
        assert persisted_session.deployment_id is None
        assert persisted_deployment is not None
        assert persisted_deployment.model_id is None
        assert persisted_plan is not None
        assert persisted_plan.provider_id is None
        assert persisted_plan.deployment_id is None
        assert persisted_message is not None

        db.execute(delete(OperationPlan).where(OperationPlan.id == plan_id))
        db.commit()

    with database.session_factory() as db:
        persisted_message = db.get(OpsMessage, message_id)
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


def test_ops_dict_json_top_level_mutations_are_persisted(tmp_path):
    database = _database(tmp_path / "mutable-json.db")
    database.create_schema()

    with database.session_factory() as db:
        provider = _provider("mutable")
        session = OpsSession(title="Mutable JSON", requested_by="admin")
        db.add_all([provider, session])
        db.flush()
        message = OpsMessage(session_id=session.id, role="user", content="Inspect")
        tool = OpsToolRun(session_id=session.id, tool_name="host.health")
        db.add_all([message, tool])
        db.commit()
        provider_id, message_id, tool_id = provider.id, message.id, tool.id

    with database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        message = db.get(OpsMessage, message_id)
        tool = db.get(OpsToolRun, tool_id)
        assert provider is not None
        assert message is not None
        assert tool is not None

        provider.last_test_result["models"] = "healthy"
        message.metadata_json["source"] = "assistant"
        tool.arguments_json["tail"] = 100
        tool.result_json["healthy"] = True

        assert all(item in db.dirty for item in (provider, message, tool))
        db.commit()

    with database.session_factory() as db:
        assert db.get(Provider, provider_id).last_test_result == {"models": "healthy"}
        assert db.get(OpsMessage, message_id).metadata_json == {"source": "assistant"}
        tool = db.get(OpsToolRun, tool_id)
        assert tool.arguments_json == {"tail": 100}
        assert tool.result_json == {"healthy": True}

    database.dispose()


def _alembic_config(database_url: str) -> Config:
    config = Config(str(REPOSITORY_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location", str(REPOSITORY_ROOT / "backend" / "migrations")
    )
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _foreign_key_ondelete(inspector, table_name: str, column_name: str) -> str | None:
    for foreign_key in inspector.get_foreign_keys(table_name):
        if foreign_key["constrained_columns"] == [column_name]:
            return foreign_key["options"].get("ondelete")
    raise AssertionError(f"Missing foreign key for {table_name}.{column_name}")


def _insert_legacy_model(connection, model_id: str) -> None:
    connection.execute(
        text(
            "INSERT INTO model_assets "
            "(id, name, source, local_path, size_bytes, status, capabilities, metadata_json, "
            "created_at, updated_at) VALUES (:id, :id, 'local', :path, 0, 'available', "
            "'[]', '{}', :now, :now)"
        ),
        {"id": model_id, "path": f"/models/{model_id}", "now": "2026-08-17 00:00:00"},
    )


def _insert_legacy_provider(connection, provider_id: str) -> None:
    connection.execute(
        text(
            "INSERT INTO providers "
            "(id, name, base_url, default_model, encrypted_api_key, timeout_seconds, "
            "headers, enabled, created_at, updated_at) VALUES "
            "(:id, :id, 'https://example.com/v1', 'model', 'encrypted', 60, '{}', 1, "
            ":now, :now)"
        ),
        {"id": provider_id, "now": "2026-08-17 00:00:00"},
    )


def _insert_legacy_deployment(connection, deployment_id: str, model_id: str | None) -> None:
    connection.execute(
        text(
            "INSERT INTO deployments "
            "(id, name, model_id, runtime, endpoint_url, api_model_name, status, health, "
            "managed, config, capabilities, created_at, updated_at) VALUES "
            "(:id, :id, :model_id, 'vllm', :endpoint, :api_model, 'running', 'healthy', "
            "1, '{}', '[]', :now, :now)"
        ),
        {
            "id": deployment_id,
            "model_id": model_id,
            "endpoint": f"http://{deployment_id}:8000/v1",
            "api_model": f"{deployment_id}-model",
            "now": "2026-08-17 00:00:00",
        },
    )


def _insert_legacy_plan(
    connection, plan_id: str, provider_id: str | None, deployment_id: str | None
) -> None:
    connection.execute(
        text(
            "INSERT INTO operation_plans "
            "(id, provider_id, deployment_id, summary, diagnosis, risk, steps, status, "
            "requested_by, result_json, created_at, updated_at) VALUES "
            "(:id, :provider_id, :deployment_id, 'Repair', 'Broken', 'low', '[]', "
            "'pending', 'admin', '{}', :now, :now)"
        ),
        {
            "id": plan_id,
            "provider_id": provider_id,
            "deployment_id": deployment_id,
            "now": "2026-08-17 00:00:00",
        },
    )


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
        provider_columns = {
            column["name"] for column in inspector.get_columns("providers")
        }
        assert {"last_test_result", "config_revision"} <= provider_columns
        assert _foreign_key_ondelete(inspector, "deployments", "model_id") == "SET NULL"
        assert _foreign_key_ondelete(inspector, "operation_plans", "provider_id") == "SET NULL"
        assert (
            _foreign_key_ondelete(inspector, "operation_plans", "deployment_id") == "SET NULL"
        )
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            "20260817_0002"
        )
    database.dispose()


def test_legacy_references_are_set_null_after_upgrade(tmp_path, monkeypatch):
    database_path = tmp_path / "legacy-references.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DGX_DATABASE_URL", database_url)
    config = _alembic_config(database_url)
    command.upgrade(config, "20260816_0001")

    database = _database(database_path)
    with database.engine.begin() as connection:
        _insert_legacy_model(connection, "legacy-model")
        _insert_legacy_provider(connection, "legacy-provider")
        _insert_legacy_deployment(connection, "model-deployment", "legacy-model")
        _insert_legacy_deployment(connection, "parent-deployment", None)
        _insert_legacy_plan(
            connection, "legacy-plan", "legacy-provider", "parent-deployment"
        )

    command.upgrade(config, "head")
    with database.engine.begin() as connection:
        connection.execute(text("DELETE FROM model_assets WHERE id='legacy-model'"))
        connection.execute(text("DELETE FROM providers WHERE id='legacy-provider'"))
        connection.execute(text("DELETE FROM deployments WHERE id='parent-deployment'"))

    with database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT model_id FROM deployments WHERE id='model-deployment'")
        ).scalar_one() is None
        plan = connection.execute(
            text(
                "SELECT provider_id, deployment_id FROM operation_plans "
                "WHERE id='legacy-plan'"
            )
        ).one()
        assert plan.provider_id is None
        assert plan.deployment_id is None
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
    database.dispose()


def test_upgrade_repairs_legacy_nullable_foreign_key_orphans(tmp_path, monkeypatch):
    database_path = tmp_path / "legacy-orphans.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DGX_DATABASE_URL", database_url)
    config = _alembic_config(database_url)
    command.upgrade(config, "20260816_0001")

    engine = Database(database_url).engine
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        _insert_legacy_deployment(connection, "orphan-deployment", "missing-model")
        _insert_legacy_plan(
            connection, "orphan-plan", "missing-provider", "missing-deployment"
        )
        connection.commit()
    engine.dispose()

    command.upgrade(config, "head")
    database = _database(database_path)
    with database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT model_id FROM deployments WHERE id='orphan-deployment'")
        ).scalar_one() is None
        plan = connection.execute(
            text(
                "SELECT provider_id, deployment_id FROM operation_plans "
                "WHERE id='orphan-plan'"
            )
        ).one()
        assert plan.provider_id is None
        assert plan.deployment_id is None
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
    database.dispose()


def test_upgrade_fails_closed_for_unexpected_foreign_key_violations(tmp_path, monkeypatch):
    database_path = tmp_path / "unexpected-orphan.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DGX_DATABASE_URL", database_url)
    config = _alembic_config(database_url)
    command.upgrade(config, "20260816_0001")

    database = _database(database_path)
    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.exec_driver_sql(
            "CREATE TABLE unexpected_children ("
            "id VARCHAR(36) PRIMARY KEY, "
            "model_id VARCHAR(36) NOT NULL REFERENCES model_assets(id))"
        )
        connection.exec_driver_sql(
            "INSERT INTO unexpected_children (id, model_id) VALUES ('child', 'missing')"
        )
        connection.commit()
    database.dispose()

    with pytest.raises(RuntimeError, match="foreign key check failed"):
        command.upgrade(config, "head")


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
        provider_columns = {
            column["name"] for column in inspector.get_columns("providers")
        }
        assert not {"last_test_result", "config_revision"} & provider_columns
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
        provider_columns = {
            column["name"] for column in inspector.get_columns("providers")
        }
        assert {"last_test_result", "config_revision"} <= provider_columns
        row = connection.execute(
            text(
                "SELECT name, last_test_result, config_revision FROM providers "
                "WHERE id='existing-provider'"
            )
        ).one()
        assert row.name == "Existing"
        assert row.last_test_result == "{}"
        assert row.config_revision == 0

    command.downgrade(config, "20260816_0001")
    with database.engine.connect() as connection:
        inspector = inspect(connection)
        assert not {"ops_sessions", "ops_messages", "ops_tool_runs"} & set(
            inspector.get_table_names()
        )
        provider_columns = {
            column["name"] for column in inspector.get_columns("providers")
        }
        assert not {"last_test_result", "config_revision"} & provider_columns
        assert _foreign_key_ondelete(inspector, "deployments", "model_id") is None
        assert _foreign_key_ondelete(inspector, "operation_plans", "provider_id") is None
        assert _foreign_key_ondelete(inspector, "operation_plans", "deployment_id") is None
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
            text(
                "SELECT last_test_result, config_revision FROM providers "
                "WHERE id='existing-provider'"
            )
        ).one() == ("{}", 0)
        assert _foreign_key_ondelete(inspector, "deployments", "model_id") == "SET NULL"
        assert _foreign_key_ondelete(inspector, "operation_plans", "provider_id") == "SET NULL"
        assert (
            _foreign_key_ondelete(inspector, "operation_plans", "deployment_id") == "SET NULL"
        )

    database.dispose()
