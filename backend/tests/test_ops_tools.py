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
from sqlalchemy import event


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


@pytest.mark.parametrize(
    ("credential", "opaque"),
    [
        ("AZURE_OPENAI_API_KEY=multi-prefix-secret", "multi-prefix-secret"),
        ("HUGGINGFACE_ACCESS_TOKEN=hf-secret", "hf-secret"),
        ("MY_CUSTOM_CLIENT_SECRET=client-secret-value", "client-secret-value"),
        ('"AZURE_OPENAI_API_KEY": "double-json-secret"', "double-json-secret"),
        ("'MY_CUSTOM_CLIENT_SECRET': 'single-json-secret'", "single-json-secret"),
        ("Authorization: Bearer multi-auth-secret", "multi-auth-secret"),
    ],
)
def test_multi_segment_credential_labels_share_structured_key_semantics(
    database: Database,
    secret_box: SecretBox,
    credential: str,
    opaque: str,
) -> None:
    registry = OpsToolRegistry(
        FakeAgent({"status": "succeeded", "output": credential}),
        database.session_factory,
        secret_box,
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert opaque not in dumped
    assert result.output["output"] == "[REDACTED]"


def test_bounded_credential_label_and_url_redaction_preserve_non_secret_text(
    database: Database, secret_box: SecretBox
) -> None:
    bounded_label = f"{'A' * 120}_API_KEY"
    assert len(bounded_label) == 128
    output = (
        f"{bounded_label}=bounded-secret "
        "...?AZURE_OPENAI_API_KEY=query-secret&mode=read "
        "token count is 7; password rotation pending"
    )
    registry = OpsToolRegistry(
        FakeAgent({"status": "succeeded", "output": output}),
        database.session_factory,
        secret_box,
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    sanitized = result.output["output"]

    assert "bounded-secret" not in sanitized
    assert "query-secret" not in sanitized
    assert "mode=read" in sanitized
    assert "token count is 7" in sanitized
    assert "password rotation pending" in sanitized


@pytest.mark.parametrize(
    ("assignment", "secret", "sensitive_tail"),
    [
        ("api_key=abc;def", "abc;def", "def"),
        ("Authorization: Bearer abc def", "abc def", "def"),
        ("token=abc&def", "abc&def", "def"),
    ],
)
def test_known_secrets_are_removed_before_assignment_parsing(
    database: Database,
    secret_box: SecretBox,
    assignment: str,
    secret: str,
    sensitive_tail: str,
) -> None:
    with database.session_factory() as db:
        db.add_all(
            [
                Provider(
                    name="contained-secret",
                    base_url="https://contained.example/v1",
                    default_model="model",
                    encrypted_api_key=secret_box.encrypt("abc"),
                ),
                Provider(
                    name="full-secret",
                    base_url="https://full.example/v1",
                    default_model="model",
                    encrypted_api_key=secret_box.encrypt(secret),
                ),
            ]
        )
        db.commit()
    agent = FakeAgent(
        {
            "status": "failed",
            "output": assignment,
            "nested": {"again": assignment},
            "large": ("x" * 50_000) + assignment,
            "error": assignment,
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert secret not in dumped
    assert sensitive_tail not in dumped
    assert "[REDACTED]" in dumped
    assert "[REDACTED]]" not in dumped
    assert result.error == "[REDACTED]"


@pytest.mark.parametrize(
    ("assignment", "forbidden"),
    [
        ("api_key=prefixabcSuffix", "Suffix"),
        ("token=abcSuffix", "Suffix"),
        ("password=prefixabc", "prefix"),
        ("api_key=[REDACTED]actual-secret", "actual-secret"),
    ],
)
def test_unquoted_credentials_consume_marker_and_adjacent_value_fragments(
    database: Database,
    secret_box: SecretBox,
    assignment: str,
    forbidden: str,
) -> None:
    with database.session_factory() as db:
        db.add(
            Provider(
                name="known-secret",
                base_url="https://known.example/v1",
                default_model="model",
                encrypted_api_key=secret_box.encrypt("abc"),
            )
        )
        db.commit()
    registry = OpsToolRegistry(
        FakeAgent(
            {
                "status": "failed",
                "output": assignment,
                "nested": {"assignment": assignment},
                "error": assignment,
            }
        ),
        database.session_factory,
        secret_box,
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert forbidden not in dumped
    assert result.output["output"] == "[REDACTED]"
    assert result.output["nested"]["assignment"] == "[REDACTED]"
    assert result.error == "[REDACTED]"
    assert "]]" not in dumped


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


def test_camel_case_credentials_are_redacted_but_public_metrics_are_preserved(
    database: Database, secret_box: SecretBox
) -> None:
    agent = FakeAgent(
        {
            "status": "succeeded",
            "accessToken": "opaque-access",
            "clientSecret": "opaque-client",
            "apiToken": "opaque-api",
            "secretKey": "opaque-key",
            "promptTokens": 13,
            "completionTokens": 8,
            "tokenCount": 21,
            "passwordRotation": "pending",
            "log": (
                "accessToken=log-access "
                '"clientSecret": "log-client" '
                "apiToken='log-api' secretKey=log-key"
            ),
        }
    )
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    for opaque in (
        "opaque-access",
        "opaque-client",
        "opaque-api",
        "opaque-key",
        "log-access",
        "log-client",
        "log-api",
        "log-key",
    ):
        assert opaque not in dumped
    assert result.output["promptTokens"] == 13
    assert result.output["completionTokens"] == 8
    assert result.output["tokenCount"] == 21
    assert result.output["passwordRotation"] == "pending"


@pytest.mark.parametrize("short_secret", ["a", "ab"])
def test_short_provider_secret_fails_closed_without_calling_agent(
    database: Database, secret_box: SecretBox, short_secret: str
) -> None:
    with database.session_factory() as db:
        db.add(
            Provider(
                name="short",
                base_url="https://short.example/v1",
                default_model="model",
                encrypted_api_key=secret_box.encrypt(short_secret),
            )
        )
        db.commit()
    agent = FakeAgent({"status": "succeeded", "output": "database available"})
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    with pytest.raises(
        ToolExecutionError, match="configured secrets are too short to sanitize safely"
    ):
        registry.execute(ReadOnlyToolRequest(name="manager.summary", arguments={}))
    assert agent.calls == []


@pytest.mark.parametrize(
    "credential_key",
    [
        "X-Auth",
        "X-Authorization",
        "xAuth",
        "xAuthorization",
        "MY_CUSTOM_AUTH",
        "MY_CUSTOM_AUTHORIZATION",
    ],
)
def test_auth_suffix_credentials_are_redacted_in_keys_and_assignments(
    database: Database,
    secret_box: SecretBox,
    credential_key: str,
) -> None:
    registry = OpsToolRegistry(
        FakeAgent(
            {
                "status": "succeeded",
                credential_key: "structured-auth-secret",
                "log": f"{credential_key}=logged-auth-secret",
                "author": "kept",
                "authors": "kept",
            }
        ),
        database.session_factory,
        secret_box,
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = result.model_dump_json()

    assert "structured-auth-secret" not in dumped
    assert "logged-auth-secret" not in dumped
    assert result.output["author"] == "kept"
    assert result.output["authors"] == "kept"


def test_short_known_secret_boundaries_preserve_words_and_redact_tokens_and_keys(
    database: Database, secret_box: SecretBox
) -> None:
    for index, secret in enumerate(("ata", "model", "active", "healthy")):
        with database.session_factory() as db:
            db.add(
                Provider(
                    name=f"boundary-{index}",
                    base_url=f"https://boundary-{index}.example/v1",
                    default_model="online-model",
                    encrypted_api_key=secret_box.encrypt(secret),
                )
            )
            db.commit()
    registry = OpsToolRegistry(FakeAgent(), database.session_factory, secret_box)

    summary = registry.execute(ReadOnlyToolRequest(name="manager.summary", arguments={}))
    assert "models" in summary.output
    assert "database" in summary.output["system"]
    assert "active_tasks" in summary.output["system"]
    assert "healthy" in summary.output["providers"]

    registry.agent = FakeAgent(
        {
            "status": "succeeded",
            "scalar": "model",
            "log": "ata model active healthy",
            "ata": "secret-key-name",
            "database": "preserved",
            "models": "preserved",
            "active_tasks": "preserved",
            "healthy_count": "preserved",
        }
    )
    agent_result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    dumped = agent_result.model_dump_json()
    assert agent_result.output["scalar"] == "[REDACTED]"
    assert agent_result.output["log"] == "[REDACTED] [REDACTED] [REDACTED] [REDACTED]"
    assert "ata" not in agent_result.output
    for key in ("database", "models", "active_tasks", "healthy_count"):
        assert key in agent_result.output
    assert "secret-key-name" in dumped


def test_single_pass_known_secret_replacement_does_not_mutate_redaction_marker(
    database: Database, secret_box: SecretBox
) -> None:
    with database.session_factory() as db:
        db.add_all(
            [
                Provider(
                    name="red",
                    base_url="https://red.example/v1",
                    default_model="model",
                    encrypted_api_key=secret_box.encrypt("RED"),
                ),
                Provider(
                    name="act",
                    base_url="https://act.example/v1",
                    default_model="model",
                    encrypted_api_key=secret_box.encrypt("ACT"),
                ),
            ]
        )
        db.commit()
    registry = OpsToolRegistry(
        FakeAgent(
            {
                "status": "failed",
                "output": "api_key=RED",
                "error": "token=ACT",
            }
        ),
        database.session_factory,
        secret_box,
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))

    assert result.output["output"] == "[REDACTED]"
    assert result.output["error"] == "[REDACTED]"
    assert result.error == "[REDACTED]"
    assert result.model_dump_json().count("[REDACTED]") == 3


def test_non_credential_custom_header_is_not_collected_as_a_secret(
    database: Database, secret_box: SecretBox
) -> None:
    with database.session_factory() as db:
        db.add(
            Provider(
                name="tenant",
                base_url="https://tenant.example/v1",
                default_model="model",
                encrypted_api_key=secret_box.encrypt("long-provider-secret"),
                headers={"X-Tenant": "a"},
            )
        )
        db.commit()
    agent = FakeAgent({"status": "succeeded", "output": "database status available for tenant a"})
    registry = OpsToolRegistry(agent, database.session_factory, secret_box)

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))

    assert result.output["output"] == "database status available for tenant a"


def test_bearer_custom_header_value_is_collected_even_with_a_custom_name(
    database: Database, secret_box: SecretBox
) -> None:
    with database.session_factory() as db:
        db.add(
            Provider(
                name="custom-auth",
                base_url="https://custom-auth.example/v1",
                default_model="model",
                encrypted_api_key=secret_box.encrypt("long-provider-secret"),
                headers={"X-Custom-Auth": "Bearer header-credential-secret"},
            )
        )
        db.commit()
    registry = OpsToolRegistry(
        FakeAgent(
            {
                "status": "succeeded",
                "output": "credential header-credential-secret",
            }
        ),
        database.session_factory,
        secret_box,
    )

    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))

    assert "header-credential-secret" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("manager.summary", {}),
        ("manager.tasks", {"limit": 5}),
        ("manager.gateway", {"minutes": 60, "limit": 5}),
    ],
)
def test_manager_database_failures_use_a_generic_tool_execution_error(
    database: Database,
    secret_box: SecretBox,
    name: str,
    arguments: dict[str, Any],
) -> None:
    raw_detail = "postgresql://admin:raw-secret@host/db SELECT private"

    class FailAfterSecretLoad:
        calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls == 1:
                return database.session_factory()
            raise RuntimeError(raw_detail)

    registry = OpsToolRegistry(FakeAgent(), FailAfterSecretLoad(), secret_box)

    with pytest.raises(ToolExecutionError) as captured:
        registry.execute(ReadOnlyToolRequest(name=name, arguments=arguments))

    assert str(captured.value) == "Manager read-only tool failed"
    assert raw_detail not in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_credential_redaction_preserves_following_non_secret_query_parameters(
    database: Database, secret_box: SecretBox
) -> None:
    agent = FakeAgent({"status": "succeeded", "output": "token=opaque-query&mode=read"})
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
        ReadOnlyToolRequest(name="manager.gateway", arguments={"minutes": 60, "limit": 10})
    )

    assert result.output["total_requests"] == 1
    assert result.output["failed_requests"] == 0
    assert result.output["error_rate"] == 0
    assert result.output["average_latency_ms"] == 100
    assert result.output["recent"][0]["prompt_tokens"] == 12
    assert result.output["recent"][0]["completion_tokens"] == 5


def test_gateway_uses_one_aggregate_and_one_recent_metric_query(
    database: Database, secret_box: SecretBox
) -> None:
    with database.session_factory() as db:
        db.add(
            RequestMetric(
                model="model",
                endpoint="chat",
                status_code=200,
                latency_ms=50,
                created_at=datetime.now(UTC),
            )
        )
        db.commit()
    statements: list[str] = []

    def capture_statement(_conn, _cursor, statement, _parameters, _context, _many):
        if "request_metrics" in statement.casefold():
            statements.append(statement.casefold())

    event.listen(database.engine, "before_cursor_execute", capture_statement)
    try:
        registry = OpsToolRegistry(FakeAgent(), database.session_factory, secret_box)
        result = registry.execute(
            ReadOnlyToolRequest(name="manager.gateway", arguments={"minutes": 60, "limit": 5})
        )
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_statement)

    assert result.output["total_requests"] == 1
    assert len(statements) == 2
    assert "case" in statements[0]
    assert "avg(" in statements[0]
    assert "order by" in statements[1]


def test_manager_tasks_truncates_large_lobs_in_sql_and_keeps_recent_log_tail(
    database: Database, secret_box: SecretBox
) -> None:
    with database.session_factory() as db:
        db.add(
            TaskRecord(
                type="model.download",
                status="failed",
                title="Large task",
                log="old-log-prefix" + ("x" * 100_000) + "recent-log-tail",
                error="error-head" + ("y" * 100_000) + "error-tail",
            )
        )
        db.commit()
    statements: list[str] = []

    def capture_statement(_conn, _cursor, statement, _parameters, _context, _many):
        if "from tasks" in statement.casefold():
            statements.append(statement.casefold())

    event.listen(database.engine, "before_cursor_execute", capture_statement)
    try:
        registry = OpsToolRegistry(FakeAgent(), database.session_factory, secret_box)
        result = registry.execute(ReadOnlyToolRequest(name="manager.tasks", arguments={"limit": 5}))
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_statement)

    task = result.output["tasks"][0]
    assert task["log"].startswith("[truncated]\n")
    assert task["log"].endswith("recent-log-tail")
    assert len(task["log"]) < 5_000
    assert task["log_truncated"] is True
    assert task["error"].startswith("error-head")
    assert len(task["error"]) < 3_000
    assert task["error_truncated"] is True
    assert len(statements) == 1
    assert "substr(tasks.log" in statements[0]
    assert "substr(tasks.error" in statements[0]


@pytest.mark.parametrize("agent_status", ["succeeded", "failed"])
def test_final_tool_result_including_wrapper_stays_within_30000_characters(
    database: Database,
    secret_box: SecretBox,
    agent_status: str,
) -> None:
    agent = FakeAgent(
        {
            "status": agent_status,
            "output": ('quote=" slash=\\ 中文' * 5000) + " api_key=opaque-boundary-secret",
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
