import pytest
from app.services.providers import normalize_openai_base_url, validate_provider_url


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

