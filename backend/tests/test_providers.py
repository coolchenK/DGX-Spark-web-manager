import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from app.models import Provider
from app.services.ops_provider import AssistantTurn, OpsProviderError
from app.services.providers import (
    normalize_openai_base_url,
    resolve_provider_endpoint,
    validate_provider_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1:8000/v1",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost:9000/v1",
        "https://user:password@example.com/v1",
    ],
)
def test_provider_url_rejects_ssrf_targets(url):
    with pytest.raises(ValueError):
        validate_provider_url(url)


def test_base_url_normalization_adds_v1():
    assert normalize_openai_base_url("https://api.example.com") == "https://api.example.com/v1"
    assert normalize_openai_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"


def test_provider_endpoint_is_resolved_once_and_pinned_with_host_and_sni() -> None:
    calls = 0

    def resolver(hostname: str, port: int, *_args):
        nonlocal calls
        calls += 1
        assert hostname == "api.example.com"
        assert port == 8443
        address = "93.184.216.34" if calls == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    endpoint = resolve_provider_endpoint(
        "https://api.example.com:8443/v1/chat/completions", resolver=resolver
    )

    assert calls == 1
    assert endpoint.url == "https://93.184.216.34:8443/v1/chat/completions"
    assert endpoint.host_header == "api.example.com:8443"
    assert endpoint.sni_hostname == "api.example.com"


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.1.1"],
        ["93.184.216.34", "192.168.1.2"],
    ],
)
def test_provider_endpoint_resolution_fails_closed_for_restricted_addresses(addresses):
    def resolver(_hostname: str, port: int, *_args):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses
        ]

    with pytest.raises(ValueError, match="restricted network"):
        resolve_provider_endpoint("https://api.example.com/v1/chat/completions", resolver=resolver)


def test_provider_api_encrypts_secret_and_returns_masked_value(authenticated_client, monkeypatch):
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)

    response = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Operations AI",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-provider-secret-123456",
            "default_model": "ops-model",
            "timeout_seconds": 45,
        },
    )

    assert response.status_code == 201
    assert response.json()["api_key_masked"] == "sk-p...3456"
    assert "api_key" not in response.json()
    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import Provider

        provider = db.get(Provider, response.json()["id"])
        assert "sk-provider-secret-123456" not in provider.encrypted_api_key


def test_provider_api_delete_sets_operation_plan_reference_null(
    authenticated_client, monkeypatch
):
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    response = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Deletable provider",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-provider-secret-123456",
            "default_model": "ops-model",
        },
    )
    assert response.status_code == 201
    provider_id = response.json()["id"]

    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import OperationPlan

        plan = OperationPlan(
            provider_id=provider_id,
            summary="Referenced provider",
            diagnosis="Delete should preserve plan",
        )
        db.add(plan)
        db.commit()
        plan_id = plan.id

    response = authenticated_client.delete(f"/api/providers/{provider_id}")

    assert response.status_code == 204
    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import OperationPlan

        assert db.get(OperationPlan, plan_id).provider_id is None


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Test\r\nInjected": "value"},
        {"X-Test": "value\r\nInjected: true"},
        {"Host": "attacker.example"},
        {"Content-Length": "999"},
        {"Authorization": "Bearer override"},
    ],
)
def test_provider_rejects_unsafe_custom_headers(authenticated_client, monkeypatch, headers):
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)

    response = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Unsafe headers",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-provider-secret-123456",
            "default_model": "ops-model",
            "headers": headers,
        },
    )

    assert response.status_code == 422


def test_provider_test_api_persists_connection_and_default_model_probe(
    authenticated_client, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Probe provider",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-provider-secret-123456",
            "default_model": "ops-model",
        },
    )
    assert created.status_code == 201

    class ProbeClient:
        def list_models(self, _provider: Provider) -> list[str]:
            return ["ops-model", "other-model"]

        def complete(self, provider: Provider, messages) -> AssistantTurn:
            assert provider.default_model == "ops-model"
            assert messages[-1]["content"] == "Confirm this model can respond."
            return AssistantTurn(action="answer", summary="ok")

    authenticated_client.app.state.provider_service.ops_provider_client = ProbeClient()
    response = authenticated_client.post(f"/api/providers/{created.json()['id']}/test")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "connection": {"status": "healthy", "models_seen": 2},
        "default_model": {"status": "healthy", "model": "ops-model"},
    }
    listed = authenticated_client.get("/api/providers").json()[0]
    assert listed["last_test_result"] == response.json()
    assert listed["last_tested_at"] is not None
    assert "encrypted_api_key" not in listed


def test_provider_test_api_reports_chat_failure_without_losing_connection_status(
    authenticated_client, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Failed model probe",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-provider-secret-123456",
            "default_model": "missing-model",
        },
    )

    class ProbeClient:
        def list_models(self, _provider: Provider) -> list[str]:
            return ["other-model"]

        def complete(self, _provider: Provider, _messages) -> AssistantTurn:
            raise OpsProviderError("configured model was rejected")

    authenticated_client.app.state.provider_service.ops_provider_client = ProbeClient()
    response = authenticated_client.post(f"/api/providers/{created.json()['id']}/test")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["connection"] == {"status": "healthy", "models_seen": 1}
    assert response.json()["default_model"] == {
        "status": "failed",
        "model": "missing-model",
        "error": "configured model was rejected",
    }


def test_provider_serialization_redacts_bounded_historical_result_without_mutating_database(
    authenticated_client, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    opaque_key = "credential.with.opaque.history.123456789"
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Historical provider",
            "base_url": "https://api.example.com/v1",
            "api_key": opaque_key,
            "default_model": "ops-model",
        },
    )
    provider_id = created.json()["id"]
    historical_result = {
        f"secret-key-{opaque_key}": {
            "error": f"upstream returned {opaque_key}\x00" + "x" * 2_000,
            "nested": [{"message": f"still {opaque_key}"}],
        },
        "items": [f"item-{index}-{opaque_key}" for index in range(80)],
    }
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        provider.last_test_result = historical_result
        provider.last_test_status = "failed"
        provider.last_tested_at = datetime.now(UTC)
        db.commit()

    listed = authenticated_client.get("/api/providers").json()[0]
    serialized_result = listed["last_test_result"]
    serialized_dump = json.dumps(serialized_result)

    assert opaque_key not in serialized_dump
    assert len(serialized_result["items"]) <= 50
    assert all(len(value) <= 500 for value in serialized_result["items"])
    assert all(len(key) <= 500 for key in serialized_result)
    with authenticated_client.app.state.database.session_factory() as db:
        persisted = db.get(Provider, provider_id)
        assert persisted.last_test_result == historical_result
        assert opaque_key in json.dumps(persisted.last_test_result)


def test_rotating_provider_key_clears_probe_state_and_never_echoes_secrets(
    authenticated_client, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    old_key = "old.opaque.provider.credential.123456"
    new_key = "new.opaque.provider.credential.987654"
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Rotating provider",
            "base_url": "https://api.example.com/v1",
            "api_key": old_key,
            "default_model": "ops-model",
        },
    )
    provider_id = created.json()["id"]
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        provider.last_test_result = {
            "status": "failed",
            "connection": {"status": "failed", "error": f"rejected {old_key}"},
        }
        provider.last_test_status = "failed"
        provider.last_tested_at = datetime.now(UTC)
        db.commit()

    response = authenticated_client.patch(
        f"/api/providers/{provider_id}", json={"api_key": new_key}
    )

    assert response.status_code == 200
    response_dump = json.dumps(response.json())
    assert old_key not in response_dump
    assert new_key not in response_dump
    assert response.json()["last_test_result"] == {}
    assert response.json()["last_test_status"] is None
    assert response.json()["last_tested_at"] is None
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.last_test_result == {}
        assert provider.last_test_status is None
        assert provider.last_tested_at is None
        assert authenticated_client.app.state.secret_box.decrypt(provider.encrypted_api_key) == (
            new_key
        )

    listed_dump = json.dumps(authenticated_client.get("/api/providers").json())
    audit_dump = json.dumps(authenticated_client.get("/api/audit").json())
    assert old_key not in listed_dump and new_key not in listed_dump
    assert old_key not in audit_dump and new_key not in audit_dump


@pytest.mark.parametrize(
    "patch",
    [
        {"base_url": "https://changed.example/v1"},
        {"default_model": "changed-model"},
        {"headers": {"X-Tenant": "changed"}},
        {"timeout_seconds": 90},
        {"enabled": False},
    ],
)
def test_probe_affecting_provider_updates_invalidate_health(
    authenticated_client, monkeypatch, patch
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": f"Invalidated {next(iter(patch))}",
            "base_url": "https://api.example.com/v1",
            "api_key": "opaque.provider.key.123456",
            "default_model": "ops-model",
        },
    )
    provider_id = created.json()["id"]
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        provider.last_test_result = {"status": "healthy"}
        provider.last_test_status = "healthy"
        provider.last_tested_at = datetime.now(UTC)
        initial_revision = provider.config_revision
        db.commit()

    response = authenticated_client.patch(f"/api/providers/{provider_id}", json=patch)

    assert response.status_code == 200
    assert response.json()["last_test_result"] == {}
    assert response.json()["last_test_status"] is None
    assert response.json()["last_tested_at"] is None
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.config_revision == initial_revision + 1
        assert provider.last_test_result == {}
        assert provider.last_test_status is None
        assert provider.last_tested_at is None


def test_same_probe_configuration_patch_keeps_health_and_revision(
    authenticated_client, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": "Unchanged provider",
            "base_url": "https://api.example.com/v1",
            "api_key": "opaque.provider.key.123456",
            "default_model": "ops-model",
            "headers": {"X-Tenant": "same"},
            "timeout_seconds": 60,
        },
    )
    provider_id = created.json()["id"]
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        provider.last_test_result = {"status": "healthy"}
        provider.last_test_status = "healthy"
        provider.last_tested_at = datetime.now(UTC)
        initial_revision = provider.config_revision
        db.commit()

    response = authenticated_client.patch(
        f"/api/providers/{provider_id}",
        json={
            "base_url": "https://api.example.com/v1/",
            "default_model": "ops-model",
            "headers": {"X-Tenant": "same"},
            "timeout_seconds": 60,
            "enabled": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["last_test_status"] == "healthy"
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.config_revision == initial_revision
        assert provider.last_test_result == {"status": "healthy"}


@pytest.mark.parametrize(
    ("patch", "expected_field", "expected_value"),
    [
        ({"api_key": "new.invalid.opaque.key.654321"}, "encrypted_api_key", None),
        (
            {"base_url": "https://concurrent.example/v1"},
            "base_url",
            "https://concurrent.example/v1",
        ),
        ({"default_model": "concurrent-model"}, "default_model", "concurrent-model"),
    ],
)
def test_stale_probe_cannot_overwrite_concurrent_provider_configuration(
    authenticated_client, monkeypatch, patch, expected_field, expected_value
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    old_key = "old.concurrent.opaque.key.123456"
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": f"Concurrent {expected_field}",
            "base_url": "https://api.example.com/v1",
            "api_key": old_key,
            "default_model": "ops-model",
        },
    )
    provider_id = created.json()["id"]
    started = threading.Event()
    release = threading.Event()

    class BlockingProbe:
        def list_models(self, provider: Provider) -> list[str]:
            assert provider.default_model == "ops-model"
            started.set()
            assert release.wait(10)
            return [provider.default_model]

        def complete(self, _provider: Provider, _messages) -> AssistantTurn:
            return AssistantTurn(action="answer", summary="stale healthy")

    authenticated_client.app.state.provider_service.ops_provider_client = BlockingProbe()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            authenticated_client.post, f"/api/providers/{provider_id}/test"
        )
        assert started.wait(10)
        updated = authenticated_client.patch(f"/api/providers/{provider_id}", json=patch)
        assert updated.status_code == 200
        release.set()
        stale_response = future.result(timeout=10)

    assert stale_response.status_code == 409
    assert stale_response.json()["detail"] == "Provider configuration changed during test"
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.last_test_result == {}
        assert provider.last_test_status is None
        assert provider.last_tested_at is None
        assert provider.config_revision == 1
        if expected_field == "encrypted_api_key":
            decrypted = authenticated_client.app.state.secret_box.decrypt(
                provider.encrypted_api_key
            )
            assert decrypted == patch["api_key"]
        else:
            assert getattr(provider, expected_field) == expected_value


@pytest.mark.parametrize("stage", ["models", "chat"])
def test_unexpected_provider_client_errors_bubble_without_persisting_failed_state(
    authenticated_client, monkeypatch, stage
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": f"Unexpected {stage}",
            "base_url": "https://api.example.com/v1",
            "api_key": "opaque.provider.key.123456",
            "default_model": "ops-model",
        },
    )
    provider_id = created.json()["id"]

    class UnexpectedProbe:
        def list_models(self, _provider: Provider) -> list[str]:
            if stage == "models":
                raise AttributeError("programming failure")
            return ["ops-model"]

        def complete(self, _provider: Provider, _messages) -> AssistantTurn:
            raise TypeError("programming failure")

    authenticated_client.app.state.provider_service.ops_provider_client = UnexpectedProbe()
    transport = authenticated_client._transport
    raise_server_exceptions = transport.raise_server_exceptions
    transport.raise_server_exceptions = False
    try:
        response = authenticated_client.post(f"/api/providers/{provider_id}/test")
    finally:
        transport.raise_server_exceptions = raise_server_exceptions
    assert response.status_code == 500

    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.last_test_result == {}
        assert provider.last_test_status is None
        assert provider.last_tested_at is None


@pytest.mark.parametrize("stage", ["models", "chat"])
def test_expected_provider_errors_return_structured_failure_and_persist(
    authenticated_client, monkeypatch, stage
) -> None:
    monkeypatch.setattr("app.services.providers.validate_provider_url", lambda value: value)
    created = authenticated_client.post(
        "/api/providers",
        json={
            "name": f"Expected {stage}",
            "base_url": "https://api.example.com/v1",
            "api_key": "opaque.provider.key.123456",
            "default_model": "ops-model",
        },
    )
    provider_id = created.json()["id"]

    class ExpectedFailureProbe:
        def list_models(self, _provider: Provider) -> list[str]:
            if stage == "models":
                raise OpsProviderError("expected connection failure")
            return ["ops-model"]

        def complete(self, _provider: Provider, _messages) -> AssistantTurn:
            raise OpsProviderError("expected chat failure")

    authenticated_client.app.state.provider_service.ops_provider_client = (
        ExpectedFailureProbe()
    )
    response = authenticated_client.post(f"/api/providers/{provider_id}/test")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    expected_section = "connection" if stage == "models" else "default_model"
    assert response.json()[expected_section]["status"] == "failed"
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.last_test_status == "failed"
        assert provider.last_test_result == response.json()
        assert provider.last_tested_at is not None
