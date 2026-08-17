from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from app.db import Database
from app.models import (
    AuditEvent,
    Deployment,
    ModelAsset,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
    RequestMetric,
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
        assert "ix_request_metrics_created_at" in {
            index["name"] for index in inspector.get_indexes("request_metrics")
        }
        assert _foreign_key_ondelete(inspector, "deployments", "model_id") == "SET NULL"
        assert _foreign_key_ondelete(inspector, "operation_plans", "provider_id") == "SET NULL"
        assert (
            _foreign_key_ondelete(inspector, "operation_plans", "deployment_id") == "SET NULL"
        )
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            "20260817_0002"
        )
    database.dispose()


def test_request_metric_created_at_index_upgrade_downgrade_cycle(
    tmp_path, monkeypatch
):
    database_url = f"sqlite:///{tmp_path / 'metric-index.db'}"
    monkeypatch.setenv("DGX_DATABASE_URL", database_url)
    config = _alembic_config(database_url)

    command.upgrade(config, "20260816_0001")
    database = _database(tmp_path / "metric-index.db")
    with database.engine.connect() as connection:
        assert "ix_request_metrics_created_at" not in {
            index["name"] for index in inspect(connection).get_indexes("request_metrics")
        }

    command.upgrade(config, "head")
    with database.engine.connect() as connection:
        assert "ix_request_metrics_created_at" in {
            index["name"] for index in inspect(connection).get_indexes("request_metrics")
        }

    command.downgrade(config, "20260816_0001")
    with database.engine.connect() as connection:
        assert "ix_request_metrics_created_at" not in {
            index["name"] for index in inspect(connection).get_indexes("request_metrics")
        }

    command.upgrade(config, "head")
    with database.engine.connect() as connection:
        assert "ix_request_metrics_created_at" in {
            index["name"] for index in inspect(connection).get_indexes("request_metrics")
        }
    database.dispose()


def test_request_metric_created_at_orm_column_is_indexed():
    assert RequestMetric.__table__.c.created_at.index is True


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


class _ScriptedOpsProvider:
    def __init__(self, turns, *, before_complete=None):
        self.turns = list(turns)
        self.calls = []
        self.before_complete = before_complete

    def complete(self, provider, messages):
        self.calls.append((provider.id, messages))
        if self.before_complete is not None:
            self.before_complete(len(self.calls), messages)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class _ScriptedOpsTools:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _ops_runtime(tmp_path, turns, tool_results=(), **kwargs):
    from app.security import SecretBox
    from app.services.ops_orchestrator import OpsOrchestrator

    database = _database(tmp_path / "orchestrator.db")
    database.create_schema()
    secret_box = SecretBox("test-secret-key-with-at-least-32-characters")
    with database.session_factory() as db:
        provider = Provider(
            name="ops-provider",
            base_url="https://provider.example/v1",
            default_model="ops-model",
            encrypted_api_key=secret_box.encrypt("known-secret-value"),
            enabled=True,
        )
        deployment = _deployment("ops-deployment")
        db.add_all([provider, deployment])
        db.flush()
        session = OpsSession(
            title="Repair",
            provider_id=provider.id,
            deployment_id=deployment.id,
            requested_by="admin",
        )
        db.add(session)
        db.commit()
        session_id = session.id
        provider_id = provider.id
        deployment_id = deployment.id
    scripted_provider = _ScriptedOpsProvider(turns)
    tools = _ScriptedOpsTools(tool_results)
    orchestrator = OpsOrchestrator(
        session_factory=database.session_factory,
        provider_client=scripted_provider,
        tools=tools,
        secret_box=secret_box,
        **kwargs,
    )
    return (
        database,
        orchestrator,
        scripted_provider,
        tools,
        session_id,
        provider_id,
        deployment_id,
        secret_box,
    )


def test_runtime_orchestrator_runs_read_tool_then_answers(tmp_path):
    from app.services.ops_orchestrator import OpsResponseResult
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect memory",
                tool={"name": "host.memory", "arguments": {}},
            ),
            AssistantTurn(action="answer", summary="Memory is healthy"),
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"free": 42})],
    )
    database, orchestrator, provider, tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="Check memory", actor="admin")

    assert isinstance(result, OpsResponseResult)
    assert result.status == "answered"
    assert result.tool_count == 1
    assert len(provider.calls) == 2
    assert [call.name for call in tools.calls] == ["host.memory"]
    with database.session_factory() as db:
        session = db.get(OpsSession, session_id)
        assert session.status == "answered"
        assert [(message.role, message.content) for message in session.messages] == [
            ("user", "Check memory"),
            ("assistant", "Inspect memory"),
            (
                "tool",
                next(message.content for message in session.messages if message.role == "tool"),
            ),
            ("assistant", "Memory is healthy"),
        ]
        tool_run = db.scalar(select(OpsToolRun).where(OpsToolRun.session_id == session_id))
        assert tool_run.status == "succeeded"
        assert tool_run.result_json["output"] == {"free": 42}
        assert tool_run.started_at is not None
        assert tool_run.finished_at is not None


def test_runtime_orchestrator_creates_pending_plan_without_agent_execution(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="plan",
                summary="Repair service",
                steps=[
                    {
                        "operation": "shell",
                        "command": "systemctl restart demo",
                        "cwd": "/",
                        "timeout": 60,
                        "reason": "Recover",
                        "impact": "Brief outage",
                        "rollback": "systemctl restart demo",
                    }
                ],
            )
        ],
    )
    database, orchestrator, _provider, tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="Repair demo", actor="admin")

    assert result.status == "approval_required"
    assert result.plan_id is not None
    assert tools.calls == []
    with database.session_factory() as db:
        plan = db.get(OperationPlan, result.plan_id)
        assert plan.status == "pending"
        assert plan.risk == "high"
        assert plan.steps[0]["operation"] == "shell"
        assert plan.steps[0]["executable"] is True
        message = db.scalar(
            select(OpsMessage).where(OpsMessage.operation_plan_id == result.plan_id)
        )
        assert message is not None
        assert message.content == "Repair service"


def test_runtime_orchestrator_rejects_secret_plan_without_persisting_secret(tmp_path):
    from app.services.ops_provider import AssistantTurn

    secret = "known-secret-value"
    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="plan",
                summary="Unsafe",
                steps=[
                    {
                        "operation": "shell",
                        "command": f"curl -H 'Authorization: Bearer {secret}' example.invalid",
                        "cwd": "/",
                        "timeout": 30,
                        "reason": "test",
                        "impact": "network",
                        "rollback": "none",
                    }
                ],
            )
        ],
    )
    database, orchestrator, *_rest, session_id, _, _, _ = runtime

    with pytest.raises(ValueError, match="secret material") as caught:
        orchestrator.respond(session_id=session_id, prompt="test", actor="admin")

    assert secret not in str(caught.value)
    with database.session_factory() as db:
        assert db.scalars(select(OperationPlan)).all() == []
        serialized = "\n".join(
            [
                *(
                    message.content + str(message.metadata_json)
                    for message in db.scalars(select(OpsMessage))
                ),
                *(str(event.details) for event in db.scalars(select(AuditEvent))),
            ]
        )
        assert secret not in serialized


def test_runtime_orchestrator_tool_audit_omits_output_and_raw_arguments(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect logs",
                tool={"name": "docker.logs", "arguments": {"container": "demo", "tail": 20}},
            ),
            AssistantTurn(action="answer", summary="Done"),
        ],
        [ToolResult(name="docker.logs", status="succeeded", output={"log": "private"})],
    )
    database, orchestrator, *_rest, session_id, _, _, _ = runtime

    orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    with database.session_factory() as db:
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.tool.execute"))
        assert event.details["tool_name"] == "docker.logs"
        assert event.details["status"] == "succeeded"
        assert event.details["argument_keys"] == ["container", "tail"]
        assert "output" not in event.details
        assert "demo" not in str(event.details)
        assert "private" not in str(event.details)


def test_runtime_orchestrator_stops_before_seventh_tool_execution(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    turns = [
        AssistantTurn(
            action="tool",
            summary=f"Inspect {index}",
            tool={"name": "host.memory", "arguments": {}},
        )
        for index in range(7)
    ]
    runtime = _ops_runtime(
        tmp_path,
        turns,
        [
            ToolResult(name="host.memory", status="succeeded", output={"index": index})
            for index in range(6)
        ],
    )
    database, orchestrator, provider, tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    assert result.tool_count == 6
    assert len(provider.calls) == 7
    assert len(tools.calls) == 6
    with database.session_factory() as db:
        assert len(db.scalars(select(OpsToolRun)).all()) == 6
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.limit"))
        assert event.details == {
            "reason": "tool_turn_limit",
            "message_id": result.message_id,
        }


def test_runtime_orchestrator_time_limit_is_persisted_after_one_tool(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    ticks = iter([0.0, 0.1, 0.2, 1.0])
    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect",
                tool={"name": "host.memory", "arguments": {}},
            )
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"ok": True})],
        max_total_seconds=0.5,
        monotonic=lambda: next(ticks),
    )
    database, orchestrator, provider, tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    assert result.tool_count == 1
    assert len(provider.calls) == 1
    assert len(tools.calls) == 1
    with database.session_factory() as db:
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.limit"))
        assert event.details["reason"] == "time_limit"


def test_runtime_orchestrator_cumulative_output_limit_is_bounded_and_audited(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect",
                tool={"name": "host.memory", "arguments": {}},
            )
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"data": "x" * 500})],
        max_total_tool_chars=120,
    )
    database, orchestrator, provider, _tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    assert len(provider.calls) == 1
    with database.session_factory() as db:
        tool_run = db.scalar(select(OpsToolRun))
        assert len(json.dumps(tool_run.result_json, separators=(",", ":"))) <= 120
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.limit"))
        assert event.details["reason"] == "tool_output_limit"


def test_runtime_orchestrator_returns_failed_tool_once_then_stops_duplicate(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    repeated = AssistantTurn(
        action="tool",
        summary="Inspect service",
        tool={"name": "systemd.status", "arguments": {"service": "demo.service"}},
    )
    runtime = _ops_runtime(
        tmp_path,
        [repeated, repeated],
        [
            ToolResult(
                name="systemd.status",
                status="failed",
                output={},
                error="service unavailable",
            )
        ],
    )
    database, orchestrator, provider, tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    assert result.tool_count == 1
    assert len(provider.calls) == 2
    assert len(tools.calls) == 1
    assert "service unavailable" in str(provider.calls[1][1])
    with database.session_factory() as db:
        assert len(db.scalars(select(OpsToolRun)).all()) == 1
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.limit"))
        assert event.details["reason"] == "repeated_failed_tool"


def test_runtime_orchestrator_commits_each_turn_before_next_provider_call(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect memory",
                tool={"name": "host.memory", "arguments": {}},
            ),
            AssistantTurn(action="answer", summary="Done"),
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"ok": True})],
    )
    database, orchestrator, provider, _tools, session_id, *_ = runtime
    observed = []

    def observe(call_number, messages):
        with database.session_factory() as db:
            persisted = list(
                db.scalars(
                    select(OpsMessage)
                    .where(OpsMessage.session_id == session_id)
                    .order_by(OpsMessage.created_at, OpsMessage.id)
                )
            )
            observed.append([(message.role, message.content) for message in persisted])

    provider.before_complete = observe
    orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert observed[0] == [("user", "inspect")]
    assert [role for role, _content in observed[1]] == ["user", "assistant", "tool"]


def test_runtime_orchestrator_saves_question_and_needs_input(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="question", summary="Which service should I inspect?")],
    )
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "needs_input"
    with database.session_factory() as db:
        session = db.get(OpsSession, session_id)
        assert session.status == "needs_input"
        assert session.messages[-1].metadata_json["action"] == "question"


@pytest.mark.parametrize("provider_state", ["missing", "disabled"])
def test_runtime_orchestrator_rejects_missing_or_disabled_provider(tmp_path, provider_state):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="unused")],
    )
    database, orchestrator, provider, _tools, session_id, provider_id, *_ = runtime
    with database.session_factory() as db:
        if provider_state == "missing":
            db.execute(delete(Provider).where(Provider.id == provider_id))
        else:
            db.get(Provider, provider_id).enabled = False
        db.commit()

    with pytest.raises(ValueError, match="provider"):
        orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert provider.calls == []
    with database.session_factory() as db:
        session = db.get(OpsSession, session_id)
        assert session.messages == []


def test_runtime_orchestrator_rejects_missing_session(tmp_path):
    from app.services.ops_orchestrator import OpsSessionNotFound
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="unused")],
    )
    _database_ref, orchestrator, provider, *_ = runtime

    with pytest.raises(OpsSessionNotFound):
        orchestrator.respond(session_id="missing", prompt="inspect", actor="admin")

    assert provider.calls == []


def test_runtime_orchestrator_provider_bug_bubbles_after_generic_failure(tmp_path):
    runtime = _ops_runtime(tmp_path, [RuntimeError("known-secret-value")])
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    with pytest.raises(RuntimeError, match="operations orchestration failed") as caught:
        orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert "known-secret-value" not in str(caught.value)
    with database.session_factory() as db:
        session = db.get(OpsSession, session_id)
        assert session.status == "failed"
        serialized = "\n".join(message.content for message in session.messages)
        assert "known-secret-value" not in serialized
        assert "internal error" in serialized


def test_runtime_orchestrator_tool_bug_marks_run_failed_then_bubbles(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect",
                tool={"name": "host.memory", "arguments": {}},
            )
        ],
        [RuntimeError("known-secret-value")],
    )
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    with pytest.raises(RuntimeError, match="operations orchestration failed") as caught:
        orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert "known-secret-value" not in str(caught.value)
    with database.session_factory() as db:
        tool_run = db.scalar(select(OpsToolRun))
        assert tool_run.status == "failed"
        assert tool_run.finished_at is not None
        assert "known-secret-value" not in str(tool_run.result_json)
        assert "known-secret-value" not in (tool_run.error or "")


def test_runtime_orchestrator_rejects_concurrent_response_for_same_session(tmp_path):
    import threading

    from app.services.ops_orchestrator import OpsSessionConflict
    from app.services.ops_provider import AssistantTurn

    entered = threading.Event()
    release = threading.Event()
    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="Done")],
    )
    _database_ref, orchestrator, provider, _tools, session_id, *_ = runtime

    def block_provider(call_number, messages):
        entered.set()
        assert release.wait(timeout=5)

    provider.before_complete = block_provider
    errors = []

    def first_response():
        try:
            orchestrator.respond(session_id=session_id, prompt="first", actor="admin")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=first_response)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(OpsSessionConflict):
            orchestrator.respond(session_id=session_id, prompt="second", actor="admin")
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []


def test_runtime_orchestrator_task_handler_validates_payload_and_returns_result(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="Done")],
    )
    _database_ref, orchestrator, _provider, _tools, session_id, *_ = runtime

    class Context:
        def __init__(self):
            self.control_checks = 0

        def check_control(self):
            self.control_checks += 1

    context = Context()
    result = orchestrator.handler(
        context,
        {"session_id": session_id, "prompt": "inspect", "actor": "admin"},
    )

    assert result == {
        "session_id": session_id,
        "status": "answered",
        "tool_count": 0,
        "message_id": result["message_id"],
        "plan_id": None,
    }
    assert context.control_checks == 3
    with pytest.raises(ValueError, match="payload"):
        orchestrator.handler(context, {"session_id": session_id, "prompt": "again"})


def test_runtime_orchestrator_provider_call_cannot_finish_after_time_limit(tmp_path):
    from app.services.ops_provider import AssistantTurn

    ticks = iter([0.0, 0.1, 1.0])
    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="Late answer")],
        max_total_seconds=0.5,
        monotonic=lambda: next(ticks),
    )
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    with database.session_factory() as db:
        assert all(message.content != "Late answer" for message in db.scalars(select(OpsMessage)))
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.limit"))
        assert event.details["reason"] == "time_limit"


def test_runtime_orchestrator_fails_closed_for_too_short_configured_secret(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="Contains x in ordinary text")],
    )
    database, orchestrator, provider, _tools, session_id, provider_id, *_ = runtime
    with database.session_factory() as db:
        configured = db.get(Provider, provider_id)
        configured.headers = {"X-Auth-Token": "x"}
        db.commit()

    with pytest.raises(RuntimeError, match="too short"):
        orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert provider.calls == []
    with database.session_factory() as db:
        assert db.get(OpsSession, session_id).messages == []


def test_runtime_orchestrator_expected_provider_failure_is_session_visible(tmp_path):
    from app.services.provider_errors import OpsProviderError

    runtime = _ops_runtime(tmp_path, [OpsProviderError("known-secret-value")])
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    with database.session_factory() as db:
        session = db.get(OpsSession, session_id)
        assert session.status == "failed"
        assert "known-secret-value" not in "\n".join(
            message.content for message in session.messages
        )
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.failure"))
        assert event.details["reason"] == "provider_failure"


def test_create_app_registers_single_ops_orchestrator_and_task_handler(settings):
    from app.main import create_app
    from app.services.ops_orchestrator import OpsOrchestrator
    from app.services.ops_tools import OpsToolRegistry

    app = create_app(settings)

    assert isinstance(app.state.ops_tool_registry, OpsToolRegistry)
    assert isinstance(app.state.ops_orchestrator, OpsOrchestrator)
    assert app.state.task_engine.handlers["ops.respond"] == app.state.ops_orchestrator.handler


def test_runtime_orchestrator_rejects_plan_for_unknown_deployment(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="plan",
                summary="Restart unknown deployment",
                steps=[
                    {
                        "operation": "restart_deployment",
                        "deployment_id": "missing-deployment",
                        "reason": "Recover",
                        "impact": "Brief outage",
                        "rollback": "Start the previous deployment",
                    }
                ],
            )
        ],
    )
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    with pytest.raises(ValueError, match="deployment"):
        orchestrator.respond(session_id=session_id, prompt="repair", actor="admin")

    with database.session_factory() as db:
        assert db.scalars(select(OperationPlan)).all() == []


@pytest.mark.parametrize("secret_source", ["setting", "header"])
def test_runtime_orchestrator_rejects_secrets_from_all_config_sources(tmp_path, secret_source):
    from app.models import SecretSetting
    from app.services.ops_provider import AssistantTurn

    secret = f"{secret_source}-secret-value"
    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="plan",
                summary="Unsafe",
                steps=[
                    {
                        "operation": "shell",
                        "command": f"printf '%s' '{secret}'",
                        "cwd": "/",
                        "reason": "test",
                        "impact": "none",
                        "rollback": "none",
                    }
                ],
            )
        ],
    )
    database, orchestrator, _provider, _tools, session_id, provider_id, _, secret_box = runtime
    with database.session_factory() as db:
        if secret_source == "setting":
            db.add(
                SecretSetting(
                    key="huggingface_token",
                    encrypted_value=secret_box.encrypt(secret),
                )
            )
        else:
            db.get(Provider, provider_id).headers = {"X-Authorization": f"Bearer {secret}"}
        db.commit()

    with pytest.raises(ValueError, match="secret material") as caught:
        orchestrator.respond(session_id=session_id, prompt="repair", actor="admin")

    assert secret not in str(caught.value)
    with database.session_factory() as db:
        serialized = "\n".join(
            [
                *(message.content for message in db.scalars(select(OpsMessage))),
                *(str(event.details) for event in db.scalars(select(AuditEvent))),
            ]
        )
        assert secret not in serialized


def test_runtime_orchestrator_bounds_each_tool_result_to_thirty_thousand_chars(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect",
                tool={"name": "host.memory", "arguments": {}},
            ),
            AssistantTurn(action="answer", summary="Done"),
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"data": "x" * 50_000})],
    )
    database, orchestrator, _provider, _tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "answered"
    with database.session_factory() as db:
        tool_run = db.scalar(select(OpsToolRun))
        assert len(json.dumps(tool_run.result_json, separators=(",", ":"))) <= 30_000


def test_runtime_orchestrator_preserves_task_cancellation_and_releases_session(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.tasks.engine import TaskCancelled

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="unused")],
    )
    database, orchestrator, provider, _tools, session_id, *_ = runtime

    class Context:
        def __init__(self):
            self.calls = 0

        def check_control(self):
            self.calls += 1
            if self.calls == 2:
                raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        orchestrator.handler(
            Context(),
            {"session_id": session_id, "prompt": "inspect", "actor": "admin"},
        )

    assert provider.calls == []
    with database.session_factory() as db:
        assert db.get(OpsSession, session_id).status == "active"


def test_runtime_orchestrator_preserves_tool_count_on_expected_provider_failure(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult
    from app.services.provider_errors import OpsProviderError

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect",
                tool={"name": "host.memory", "arguments": {}},
            ),
            OpsProviderError("provider unavailable"),
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"ok": True})],
    )
    _database_ref, orchestrator, _provider, _tools, session_id, *_ = runtime

    result = orchestrator.respond(session_id=session_id, prompt="inspect", actor="admin")

    assert result.status == "failed"
    assert result.tool_count == 1


def test_runtime_orchestrator_recovers_interrupted_processing_sessions(tmp_path):
    from app.services.ops_provider import AssistantTurn

    runtime = _ops_runtime(
        tmp_path,
        [AssistantTurn(action="answer", summary="unused")],
    )
    database, orchestrator, _provider, _tools, session_id, *_ = runtime
    with database.session_factory() as db:
        db.get(OpsSession, session_id).status = "processing"
        db.commit()

    assert orchestrator.recover_interrupted() == 1

    with database.session_factory() as db:
        assert db.get(OpsSession, session_id).status == "active"
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.failure"))
        assert event.actor == "system"
        assert event.details == {"reason": "manager_restart"}


def test_runtime_orchestrator_finishes_running_tool_when_cancelled_before_call(tmp_path):
    from app.services.ops_provider import AssistantTurn
    from app.services.ops_tools import ToolResult
    from app.tasks.engine import TaskCancelled

    runtime = _ops_runtime(
        tmp_path,
        [
            AssistantTurn(
                action="tool",
                summary="Inspect",
                tool={"name": "host.memory", "arguments": {}},
            )
        ],
        [ToolResult(name="host.memory", status="succeeded", output={"unused": True})],
    )
    database, orchestrator, _provider, tools, session_id, *_ = runtime

    class Context:
        def __init__(self):
            self.calls = 0

        def check_control(self):
            self.calls += 1
            if self.calls == 3:
                raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        orchestrator.handler(
            Context(),
            {"session_id": session_id, "prompt": "inspect", "actor": "admin"},
        )

    assert tools.calls == []
    with database.session_factory() as db:
        tool_run = db.scalar(select(OpsToolRun))
        assert tool_run.status == "failed"
        assert tool_run.started_at is not None
        assert tool_run.finished_at is not None
        assert db.get(OpsSession, session_id).status == "active"
