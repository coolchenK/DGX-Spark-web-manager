import socket

import pytest
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
