import socket

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
