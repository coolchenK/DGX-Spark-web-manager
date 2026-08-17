from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from app.db import Database
from app.models import (
    Deployment,
    ModelAsset,
    Provider,
    RequestMetric,
    SecretSetting,
    TaskRecord,
)
from app.security import SecretBox
from app.services.ops_agent import OpsAgentUnavailable
from app.services.ops_provider import ReadOnlyToolRequest
from app.services.ops_tools import OpsToolRegistry, ToolExecutionError
from pydantic import ValidationError


class FakeAgent:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        self.result = result or {"status": "succeeded", "output": "ok"}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def database(tmp_path) -> Database:
    value = Database(f"sqlite:///{tmp_path / 'ops-tools.db'}")
    value.create_schema()
    yield value
    value.dispose()


@pytest.fixture
def secret_box() -> SecretBox:
    return SecretBox("test-secret-key-that-is-at-least-32-bytes")


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("host.memory", {}),
        ("host.disk", {}),
        ("host.gpu", {}),
        ("host.ports", {}),
        ("host.processes", {}),
        ("docker.list", {}),
        ("docker.inspect", {"container": "model-server_1"}),
        ("docker.logs", {"container": "model-server_1", "tail": 5000}),
        ("docker.stats", {"container": "a" * 128}),
        ("systemd.status", {"service": "docker.service"}),
        ("systemd.journal", {"service": "model@one.service", "tail": 1}),
        ("manager.summary", {}),
        ("manager.tasks", {"limit": 7}),
        ("manager.gateway", {"minutes": 15, "limit": 3}),
    ],
)
def test_registry_validates_and_dispatches_all_read_only_tools(
    database: Database,
    secret_box: SecretBox,
    name: str,
    arguments: dict[str, Any],
) -> None:
    agent = FakeAgent()
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)
    request = ReadOnlyToolRequest.model_validate({"name": name, "arguments": arguments})

    result = registry.execute(request)

    assert result.name == name
    assert result.risk == "read_only"
    assert result.status == "succeeded"
    if not name.startswith("manager."):
        assert agent.calls == [(name, arguments)]
    else:
        assert agent.calls == []


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("host.memory", {"unexpected": True}),
        ("host.ports", {"protocol": "tcp"}),
        ("host.processes", {"limit": 10}),
        ("docker.inspect", {"container": ""}),
        ("docker.inspect", {"container": "-option"}),
        ("docker.inspect", {"container": "a" * 129}),
        ("docker.inspect", {"container": "bad/name"}),
        ("docker.logs", {"container": "valid", "tail": 0}),
        ("docker.logs", {"container": "valid", "tail": 5001}),
        ("docker.logs", {"container": "valid", "tail": True}),
        ("docker.stats", {"container": "valid", "extra": 1}),
        ("systemd.status", {"service": "-option"}),
        ("systemd.status", {"service": "123"}),
        ("systemd.status", {"service": "bad/name"}),
        ("systemd.status", {"service": "a" * 257}),
        ("systemd.journal", {"service": "docker.service", "tail": "10"}),
        ("manager.summary", {"details": True}),
        ("manager.tasks", {"limit": 0}),
        ("manager.tasks", {"limit": 51}),
        ("manager.tasks", {"limit": True}),
        ("manager.gateway", {"minutes": 0}),
        ("manager.gateway", {"minutes": 1441}),
        ("manager.gateway", {"limit": 2, "extra": "no"}),
    ],
)
def test_tool_arguments_are_strictly_validated_by_name(
    name: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        ReadOnlyToolRequest.model_validate({"name": name, "arguments": arguments})


def test_shell_is_never_an_automatic_tool(database: Database, secret_box: SecretBox) -> None:
    agent = FakeAgent()
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)
    with pytest.raises(ValidationError):
        ReadOnlyToolRequest.model_validate(
            {"name": "shell.execute", "arguments": {"command": "id"}}
        )

    bypassed = ReadOnlyToolRequest.model_construct(
        name="shell.execute", arguments={"command": "id"}
    )
    with pytest.raises(ValueError, match="not an automatic read-only tool"):
        registry.execute(bypassed)
    assert agent.calls == []


def _seed_manager_data(database: Database, secret_box: SecretBox) -> dict[str, str]:
    provider_secret = "provider-plain-secret-123"
    setting_secret = "hf-plain-secret-456"
    encrypted_provider = secret_box.encrypt(provider_secret)
    encrypted_setting = secret_box.encrypt(setting_secret)
    with database.session_factory() as db:
        model = ModelAsset(
            name="model", local_path="/models/model", status="available", source="hf"
        )
        deployment = Deployment(
            name="deployment",
            model=model,
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8000/v1",
            api_model_name="model",
            status="running",
            health="healthy",
            managed=True,
        )
        provider = Provider(
            name="provider",
            base_url="https://provider.example/v1",
            default_model="online-model",
            encrypted_api_key=encrypted_provider,
            headers={"Authorization": f"Bearer {provider_secret}"},
            enabled=True,
            last_test_status="healthy",
        )
        task = TaskRecord(
            type="model.download",
            status="failed",
            title="Download",
            progress=0.5,
            input_json={"token": setting_secret},
            result_json={"api_key": provider_secret},
            error=f"Authorization: Bearer {provider_secret}",
            log=("x" * 40_000) + setting_secret,
        )
        metric = RequestMetric(
            model="model",
            endpoint="chat",
            status_code=500,
            latency_ms=123.5,
            prompt_tokens=10,
            completion_tokens=4,
            created_at=datetime.now(UTC),
        )
        db.add_all(
            [
                model,
                deployment,
                provider,
                SecretSetting(key="huggingface_token", encrypted_value=encrypted_setting),
                task,
                metric,
            ]
        )
        db.commit()
    return {
        "provider": provider_secret,
        "setting": setting_secret,
        "encrypted_provider": encrypted_provider,
        "encrypted_setting": encrypted_setting,
    }


@pytest.mark.parametrize("tool_name", ["manager.summary", "manager.tasks", "manager.gateway"])
def test_manager_tools_use_public_bounded_fields_and_never_leak_secrets(
    database: Database,
    secret_box: SecretBox,
    tool_name: str,
) -> None:
    secrets = _seed_manager_data(database, secret_box)
    registry = OpsToolRegistry(FakeAgent(), database.session_factory, secret_box)
    arguments = {"limit": 10} if tool_name == "manager.tasks" else {}
    request = ReadOnlyToolRequest.model_validate({"name": tool_name, "arguments": arguments})

    result = registry.execute(request)
    dumped = json.dumps(result.output, ensure_ascii=True)

    assert result.risk == "read_only"
    assert result.status == "succeeded"
    assert len(dumped) <= 30_000
    for secret in secrets.values():
        assert secret not in dumped
    assert "encrypted_api_key" not in dumped
    assert "Authorization" not in dumped
    if tool_name == "manager.summary":
        assert result.output["models"]["total"] == 1
        assert result.output["deployments"]["healthy"] == 1
        assert result.output["providers"]["healthy"] == 1
    elif tool_name == "manager.tasks":
        assert result.output["tasks"][0]["title"] == "Download"
        assert "input_json" not in result.output["tasks"][0]
    else:
        assert result.output["total_requests"] == 1
        assert result.output["failed_requests"] == 1


def test_agent_result_is_recursively_redacted_and_bounded(
    database: Database, secret_box: SecretBox
) -> None:
    secrets = _seed_manager_data(database, secret_box)
    agent = FakeAgent(
        {
            "status": "succeeded",
            "output": ("z" * 50_000) + secrets["provider"],
            "headers": {"Authorization": f"Bearer {secrets['setting']}"},
            "opaque": secrets["encrypted_provider"],
            "note": "api-key=opaque-value",
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = json.dumps(result.output, ensure_ascii=True)

    assert len(dumped) <= 30_000
    assert "Authorization" not in dumped
    assert "api-key" not in dumped
    for secret in secrets.values():
        assert secret not in dumped


@pytest.mark.parametrize(
    ("credential", "opaque"),
    [
        ("Authorization: Bearer opaque-auth-value", "opaque-auth-value"),
        ("authorization = Basic opaque-basic-value", "opaque-basic-value"),
        ("api_key=opaque-api-value", "opaque-api-value"),
        ("api-key: opaque-hyphen-value", "opaque-hyphen-value"),
        ('"apikey": "opaque-json-value"', "opaque-json-value"),
        ('"token": "opaque-token-value"', "opaque-token-value"),
        ("access_token='opaque access value'", "opaque access value"),
        ("password: 'opaque password value'", "opaque password value"),
        ("secret=opaque-secret-value", "opaque-secret-value"),
    ],
)
def test_unknown_credential_assignments_redact_the_entire_value(
    database: Database,
    secret_box: SecretBox,
    credential: str,
    opaque: str,
) -> None:
    agent = FakeAgent(
        {
            "status": "failed",
            "output": f"prefix {credential} suffix",
            "error": credential,
            "normal": "token count is 8 and password rotation is scheduled",
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert opaque not in dumped
    assert "[REDACTED]" in dumped
    assert result.output["normal"] == "token count is 8 and password rotation is scheduled"
    assert result.error is not None
    assert opaque not in result.error


def test_sensitive_key_matching_preserves_public_token_metrics(
    database: Database, secret_box: SecretBox
) -> None:
    agent = FakeAgent(
        {
            "status": "succeeded",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "token": "opaque-token",
            "access_token": "opaque-access-token",
            "nested": {"client_secret": "opaque-client-secret"},
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert result.output["prompt_tokens"] == 11
    assert result.output["completion_tokens"] == 7
    assert "opaque-token" not in dumped
    assert "opaque-access-token" not in dumped
    assert "opaque-client-secret" not in dumped


def test_credential_redaction_preserves_following_non_secret_query_parameters(
    database: Database, secret_box: SecretBox
) -> None:
    agent = FakeAgent(
        {"status": "succeeded", "output": "token=opaque-query&mode=read"}
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))

    assert "opaque-query" not in result.output["output"]
    assert "mode=read" in result.output["output"]


def test_gateway_aggregates_use_the_requested_time_window(
    database: Database, secret_box: SecretBox
) -> None:
    now = datetime.now(UTC)
    with database.session_factory() as db:
        db.add_all(
            [
                RequestMetric(
                    model="old",
                    endpoint="chat",
                    status_code=500,
                    latency_ms=1000,
                    created_at=now - timedelta(hours=2),
                ),
                RequestMetric(
                    model="recent",
                    endpoint="chat",
                    status_code=200,
                    latency_ms=100,
                    prompt_tokens=12,
                    completion_tokens=5,
                    created_at=now - timedelta(minutes=5),
                ),
            ]
        )
        db.commit()
    registry = OpsToolRegistry(FakeAgent(), database.session_factory, secret_box)

    result = registry.execute(
        ReadOnlyToolRequest(
            name="manager.gateway", arguments={"minutes": 60, "limit": 10}
        )
    )

    assert result.output["total_requests"] == 1
    assert result.output["failed_requests"] == 0
    assert result.output["error_rate"] == 0
    assert result.output["average_latency_ms"] == 100
    assert result.output["recent"][0]["prompt_tokens"] == 12
    assert result.output["recent"][0]["completion_tokens"] == 5


@pytest.mark.parametrize("agent_status", ["succeeded", "failed"])
def test_final_tool_result_including_wrapper_stays_within_30000_characters(
    database: Database,
    secret_box: SecretBox,
    agent_status: str,
) -> None:
    agent = FakeAgent(
        {
            "status": agent_status,
            "output": ('quote=" slash=\\ 中文' * 5000)
            + " api_key=opaque-boundary-secret",
            "error": (
                "password=opaque-error-secret " + ("错误" * 1000)
                if agent_status == "failed"
                else None
            ),
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert len(dumped) <= 30_000
    assert "opaque-boundary-secret" not in dumped
    assert "opaque-error-secret" not in dumped


def test_expected_agent_unavailability_is_a_persistable_failure(
    database: Database, secret_box: SecretBox
) -> None:
    registry = OpsToolRegistry(
        FakeAgent(error=OpsAgentUnavailable()), database.session_factory, secret_box
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.gpu", arguments={}))

    assert result.status == "failed"
    assert result.risk == "read_only"
    assert result.output == {}
    assert result.error == "Host operations agent is unavailable"


@pytest.mark.parametrize("agent_status", ["failed", "timed_out", "cancelled"])
def test_terminal_agent_job_failure_is_not_reported_as_tool_success(
    database: Database,
    secret_box: SecretBox,
    agent_status: str,
) -> None:
    agent = FakeAgent(
        {
            "status": agent_status,
            "output": "command output",
            "error": "operation failed",
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.disk", arguments={}))

    assert result.status == "failed"
    assert result.error == "operation failed"
    assert result.output["status"] == agent_status


def test_unexpected_agent_bug_crosses_the_tool_execution_boundary(
    database: Database, secret_box: SecretBox
) -> None:
    registry = OpsToolRegistry(
        FakeAgent(error=RuntimeError("raw internal detail")),
        database.session_factory,
        secret_box,
    )

    with pytest.raises(ToolExecutionError, match="unexpected Agent failure"):
        registry.execute(ReadOnlyToolRequest(name="host.disk", arguments={}))


def test_secret_decryption_failure_returns_no_tool_output(
    database: Database, secret_box: SecretBox
) -> None:
    agent = FakeAgent({"output": "otherwise-safe"})
    with database.session_factory() as db:
        db.add(
            Provider(
                name="broken",
                base_url="https://example.com/v1",
                default_model="model",
                encrypted_api_key="not-a-valid-ciphertext",
            )
        )
        db.commit()
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    with pytest.raises(ToolExecutionError, match="secrets could not be loaded safely"):
        registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    assert agent.calls == []
