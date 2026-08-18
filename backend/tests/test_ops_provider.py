from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from app.models import Provider
from app.security import SecretBox
from app.services.ops_provider import (
    MAX_PROVIDER_RESPONSE_BYTES,
    MAX_REPAIR_MESSAGE_CHARS,
    MAX_REPAIR_MESSAGES,
    MAX_REPAIR_TOTAL_CHARS,
    AssistantTurn,
    OpsProviderClient,
    OpsProviderError,
)
from app.services.providers import PinnedProviderEndpoint, ProviderService
from pydantic import ValidationError

SECRET_KEY = "test-secret-key-with-at-least-32-characters"
API_KEY = "sk-test-provider-secret"


def _provider() -> Provider:
    box = SecretBox(SECRET_KEY)
    return Provider(
        id="provider-1",
        name="Operations AI",
        base_url="https://provider.invalid/v1",
        default_model="reasoning-model",
        encrypted_api_key=box.encrypt(API_KEY),
        timeout_seconds=30,
        headers={"X-Tenant": "spark"},
    )


def _endpoint(url: str) -> PinnedProviderEndpoint:
    return PinnedProviderEndpoint(
        url=url.replace("provider.invalid", "93.184.216.34"),
        host_header="provider.invalid",
        sni_hostname="provider.invalid",
    )


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Check the DGX Spark"}]


def _success(summary: str = "healthy", *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"action": "answer", "summary": summary}),
                },
            }
        ]
    }


def _client(handler) -> OpsProviderClient:
    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        return httpx.Client(transport=transport, **kwargs)

    return OpsProviderClient(
        SecretBox(SECRET_KEY),
        endpoint_resolver=_endpoint,
        http_client_factory=factory,
    )


def test_structured_provider_response_budgets_match_long_context_contract():
    assert MAX_PROVIDER_RESPONSE_BYTES == 4 * 1024 * 1024
    assert MAX_REPAIR_MESSAGE_CHARS == 12_000
    assert MAX_REPAIR_TOTAL_CHARS == 128_000


def test_assistant_turn_contract_is_strict_and_action_specific() -> None:
    turn = AssistantTurn.model_validate(
        {
            "action": "tool",
            "summary": "Inspect memory",
            "tool": {"name": "host.memory", "arguments": {}},
        }
    )
    assert turn.tool is not None

    with pytest.raises(ValidationError):
        AssistantTurn.model_validate(
            {"action": "answer", "summary": "ok", "unexpected": True}
        )
    with pytest.raises(ValidationError):
        AssistantTurn.model_validate({"action": "tool", "summary": "missing tool"})
    with pytest.raises(ValidationError):
        AssistantTurn.model_validate(
            {
                "action": "plan",
                "summary": "too many",
                "steps": [
                    {
                        "operation": "shell",
                        "command": "true",
                        "cwd": "/",
                        "timeout": 30,
                        "reason": "repair",
                        "impact": "none",
                        "rollback": "none",
                    }
                ]
                * 21,
            }
        )


def test_reasons_then_repairs_with_larger_budget_without_using_reasoning_content() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_content": (
                                    '{"action":"plan","summary":"run it","steps":'
                                    '[{"operation":"shell","command":"rm -rf /"}]}'
                                ),
                            },
                        }
                    ]
                },
            )
        return httpx.Response(200, json=_success("repaired"))

    result = _client(handler).complete(_provider(), _messages())

    assert result.summary == "repaired"
    assert [request["max_tokens"] for request in requests] == [16384, 32768]
    assert requests[0]["messages"] != requests[1]["messages"]
    assert any(
        message.get("content") == "Check the DGX Spark"
        for message in requests[1]["messages"]
    )


def test_finish_reason_length_repairs_even_when_content_is_valid() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=_success("truncated", finish_reason="length"))
        return httpx.Response(200, json=_success("complete"))

    result = _client(handler).complete(_provider(), _messages())

    assert result.summary == "complete"
    assert calls == 2


def test_complete_shares_one_timeout_across_fallback_and_repair() -> None:
    requests = 0
    configured_timeouts: list[float] = []
    ticks = iter([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7])

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                400, json={"error": {"message": "response_format unsupported"}}
            )
        if requests == 2:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": ""}}
                    ]
                },
            )
        return httpx.Response(200, json=_success("repaired in time"))

    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        configured_timeouts.append(float(kwargs["timeout"]))
        return httpx.Client(transport=transport, **kwargs)

    client = OpsProviderClient(
        SecretBox(SECRET_KEY),
        endpoint_resolver=_endpoint,
        http_client_factory=factory,
        _monotonic=lambda: next(ticks),
    )

    result = client.complete(_provider(), _messages(), timeout_seconds=1.0)

    assert result.summary == "repaired in time"
    assert len(configured_timeouts) == 3
    assert configured_timeouts == sorted(configured_timeouts, reverse=True)
    assert configured_timeouts[0] <= 1.0


def test_complete_enforces_absolute_deadline_while_reading_trickle_response() -> None:
    class Clock:
        def __init__(self) -> None:
            self.now = 10.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    encoded = json.dumps(_success("too late")).encode()

    class TrickleStream(httpx.SyncByteStream):
        def __iter__(self):
            for offset in range(0, len(encoded), 8):
                clock.now += 0.4
                yield encoded[offset : offset + 8]

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=TrickleStream())
    )

    def factory(**kwargs):
        return httpx.Client(transport=transport, **kwargs)

    client = OpsProviderClient(
        SecretBox(SECRET_KEY),
        endpoint_resolver=_endpoint,
        http_client_factory=factory,
        _monotonic=clock,
    )

    with pytest.raises(OpsProviderError, match="deadline exceeded"):
        client.complete(_provider(), _messages(), timeout_seconds=1.0)


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"action": "answer", "summary": "", "extra": "forbidden"}),
    ],
)
def test_malformed_or_schema_invalid_content_gets_one_repair(content: str) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": content},
                        }
                    ]
                },
            )
        return httpx.Response(200, json=_success("valid"))

    assert _client(handler).complete(_provider(), _messages()).summary == "valid"
    assert calls == 2


def test_repair_failure_is_bounded_and_does_not_start_a_third_attempt() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "{"}}]},
        )

    with pytest.raises(OpsProviderError, match="invalid structured response"):
        _client(handler).complete(_provider(), _messages())
    assert calls == 2


def test_invalid_openai_envelope_is_not_treated_as_repairable_content() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(OpsProviderError, match="response shape is invalid"):
        _client(handler).complete(_provider(), _messages())
    assert calls == 1


def test_response_format_named_400_retries_same_attempt_without_parameter() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is unsupported"}},
            )
        return httpx.Response(200, json=_success("fallback"))

    result = _client(handler).complete(_provider(), _messages())

    assert result.summary == "fallback"
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]
    assert requests[0]["max_tokens"] == requests[1]["max_tokens"] == 16384


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (500, {"error": {"message": "response_format exploded"}}),
        (400, {"error": {"message": "model is unavailable"}}),
    ],
)
def test_response_format_fallback_is_not_used_for_server_or_unrelated_client_errors(
    status_code: int, body: dict[str, Any]
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json=body)

    with pytest.raises(OpsProviderError):
        _client(handler).complete(_provider(), _messages())
    assert calls == 1


def test_provider_errors_are_redacted_bounded_and_never_include_credentials() -> None:
    dangerous = (
        f"Authorization: Bearer {API_KEY}\r\napi_key={API_KEY}\x00"
        + "x" * 2_000
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=dangerous)

    with pytest.raises(OpsProviderError) as captured:
        _client(handler).complete(_provider(), _messages())

    message = str(captured.value)
    assert API_KEY not in message
    assert "Authorization" not in message
    assert "\r" not in message and "\n" not in message and "\x00" not in message
    assert len(captured.value.detail) <= 500


@pytest.mark.parametrize("error_type", ["connect", "os"])
def test_opaque_provider_key_is_removed_from_transport_error_and_persisted_probe(
    settings, error_type: str
) -> None:
    from app.db import Database

    opaque_key = "credential.with.an-opaque-format.987654321"
    box = SecretBox(SECRET_KEY)
    provider = _provider()
    provider.encrypted_api_key = box.encrypt(opaque_key)

    def handler(request: httpx.Request) -> httpx.Response:
        message = f"socket rejected Authorization credential {opaque_key}"
        if error_type == "connect":
            raise httpx.ConnectError(message, request=request)
        raise OSError(message)

    client = _client(handler)
    with pytest.raises(OpsProviderError) as captured:
        client.list_models(provider)
    assert opaque_key not in str(captured.value)
    assert opaque_key not in captured.value.detail

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        db.add(provider)
        db.commit()
        provider_id = provider.id
    service = ProviderService(box, client)
    with database.session_factory() as db:
        result = service.test(db, db.get(Provider, provider_id))
        serialized = service.serialize(db.get(Provider, provider_id))

    assert opaque_key not in json.dumps(result)
    assert opaque_key not in json.dumps(serialized, default=str)
    assert result["connection"]["error"] == serialized["last_test_result"]["connection"][
        "error"
    ]

    class LeakyProbe:
        def list_models(self, _provider: Provider) -> list[str]:
            raise OpsProviderError(f"probe wrapper retained {opaque_key}")

    service.ops_provider_client = LeakyProbe()
    with database.session_factory() as db:
        second_result = service.test(db, db.get(Provider, provider_id))
        second_serialized = service.serialize(db.get(Provider, provider_id))
    assert opaque_key not in json.dumps(second_result)
    assert opaque_key not in json.dumps(second_serialized, default=str)
    database.dispose()


def test_chat_request_uses_pinned_endpoint_safe_headers_and_authentication() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success())

    _client(handler).complete(_provider(), _messages())

    request = seen[0]
    assert request.url.host == "93.184.216.34"
    assert request.headers["Host"] == "provider.invalid"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["X-Tenant"] == "spark"
    assert request.headers["Accept-Encoding"] == "identity"
    assert request.extensions["sni_hostname"] == "provider.invalid"


def test_models_request_has_no_json_body_and_legacy_trailing_slash_is_normalized() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "reasoning-model"}]})

    provider = _provider()
    provider.base_url += "/"
    assert _client(handler).list_models(provider) == ["reasoning-model"]
    assert seen[0].url.path == "/v1/models"
    assert seen[0].content == b""


def test_repair_context_is_compact_bounded_and_drops_non_content_reasoning_fields() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_content": "provider-private-reasoning",
                            },
                        }
                    ]
                },
            )
        return httpx.Response(200, json=_success("bounded repair"))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "CORE-SYSTEM-CONSTRAINT " + "s" * 30_000,
            "reasoning_content": "SYSTEM-REASONING-MUST-NOT-COPY",
        }
    ]
    for index in range(20):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"old-assistant-{index} " + "a" * 20_000,
                    "reasoning_content": f"assistant-reasoning-{index}",
                },
                {
                    "role": "tool",
                    "content": f"old-tool-{index} " + "t" * 20_000,
                    "reasoning_content": f"tool-reasoning-{index}",
                },
                {"role": "user", "content": f"old-user-{index} " + "u" * 20_000},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "CURRENT-USER-REQUEST diagnose the gateway " + "c" * 40_000,
            "reasoning_content": "USER-REASONING-MUST-NOT-COPY",
        }
    )

    assert _client(handler).complete(_provider(), messages).summary == "bounded repair"
    repair_messages = requests[1]["messages"]
    repair_dump = json.dumps(repair_messages)

    assert len(repair_messages) <= MAX_REPAIR_MESSAGES
    assert sum(len(message["content"]) for message in repair_messages) <= (
        MAX_REPAIR_TOTAL_CHARS
    )
    assert all(len(message["content"]) <= MAX_REPAIR_MESSAGE_CHARS for message in repair_messages)
    assert "CORE-SYSTEM-CONSTRAINT" in repair_dump
    assert "CURRENT-USER-REQUEST" in repair_dump
    assert "old-tool-0" not in repair_dump
    assert "reasoning_content" not in repair_dump
    assert "MUST-NOT-COPY" not in repair_dump
    assert "provider-private-reasoning" not in repair_dump


def test_assistant_summary_must_contain_non_whitespace_text() -> None:
    with pytest.raises(ValidationError):
        AssistantTurn(action="answer", summary="   ")


class _ProbeClient:
    def __init__(self, *, chat_error: Exception | None = None) -> None:
        self.chat_error = chat_error
        self.models_calls: list[str] = []
        self.chat_calls: list[tuple[str, list[dict[str, str]]]] = []

    def list_models(self, provider: Provider) -> list[str]:
        self.models_calls.append(provider.id)
        return ["model-a", provider.default_model]

    def complete(
        self, provider: Provider, messages: list[dict[str, str]]
    ) -> AssistantTurn:
        self.chat_calls.append((provider.default_model, messages))
        if self.chat_error:
            raise self.chat_error
        return AssistantTurn(action="answer", summary="probe ok")


def _persisted_provider(database) -> tuple[str, SecretBox]:
    box = SecretBox(SECRET_KEY)
    with database.session_factory() as db:
        provider = _provider()
        db.add(provider)
        db.commit()
        return provider.id, box


def test_provider_test_checks_models_and_default_model_and_persists_structured_result(
    settings,
) -> None:
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    provider_id, box = _persisted_provider(database)
    probe = _ProbeClient()
    service = ProviderService(box, probe)

    with database.session_factory() as db:
        result = service.test(db, db.get(Provider, provider_id))

    assert result == {
        "status": "healthy",
        "connection": {"status": "healthy", "models_seen": 2},
        "default_model": {"status": "healthy", "model": "reasoning-model"},
    }
    with database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.last_test_status == "healthy"
        assert provider.last_test_result == result
        serialized = service.serialize(provider)
        assert serialized["last_test_result"] == result
        assert serialized["last_tested_at"] is not None
        assert "encrypted_api_key" not in serialized
    database.dispose()


def test_provider_test_keeps_connection_healthy_when_default_model_probe_fails(settings) -> None:
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    provider_id, box = _persisted_provider(database)
    probe = _ProbeClient(chat_error=OpsProviderError("default model rejected"))
    service = ProviderService(box, probe)

    with database.session_factory() as db:
        result = service.test(db, db.get(Provider, provider_id))

    assert result["status"] == "failed"
    assert result["connection"] == {"status": "healthy", "models_seen": 2}
    assert result["default_model"]["status"] == "failed"
    assert result["default_model"]["model"] == "reasoning-model"
    assert "default model rejected" in result["default_model"]["error"]
    with database.session_factory() as db:
        assert db.get(Provider, provider_id).last_test_result == result
    database.dispose()


def test_provider_test_stops_after_connection_failure(settings) -> None:
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    provider_id, box = _persisted_provider(database)
    probe = _ProbeClient()

    def fail_models(_provider: Provider) -> list[str]:
        raise OpsProviderError("connection refused")

    probe.list_models = fail_models  # type: ignore[method-assign]
    service = ProviderService(box, probe)

    with database.session_factory() as db:
        result = service.test(db, db.get(Provider, provider_id))

    assert result["status"] == "failed"
    assert result["connection"] == {"status": "failed", "error": "connection refused"}
    assert result["default_model"] == {
        "status": "not_tested",
        "model": "reasoning-model",
    }
    assert probe.chat_calls == []
    database.dispose()


def test_create_app_shares_one_ops_provider_client_with_provider_service(settings) -> None:
    from app.main import create_app

    app = create_app(settings)

    assert app.state.provider_service.ops_provider_client is app.state.ops_provider_client
