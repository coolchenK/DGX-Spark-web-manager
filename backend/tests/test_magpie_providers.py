from unittest.mock import Mock

import httpx
import pytest
from app.models import OperationPlan, Provider
from app.services.magpie_providers import (
    GATEWAY_BASE,
    MagpieProviderSync,
    resolve_magpie_provider_endpoint,
)
from app.services.providers import validate_provider_url
from sqlalchemy import select


def snapshot(*, name="DeepSeek", models=("deepseek-v4-flash",), off=False):
    return [
        {
            "id": "deepseek",
            "name": name,
            "ready": True,
            "off": off,
            "chat": "https://api.deepseek.com/v1/chat/completions",
            "key": {"set": True, "masked": "sk-***"},
            "models": [{"id": model, "on": True} for model in models],
        }
    ]


def test_sync_preserves_existing_provider_id_and_conversation_reference(client):
    box = client.app.state.secret_box
    sync = MagpieProviderSync(box, "web-key", "server-only-gateway-key")
    with client.app.state.database.session_factory() as db:
        original = Provider(
            id="original-provider",
            name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            default_model="deepseek-v4-flash",
            encrypted_api_key=box.encrypt("vendor-secret"),
            timeout_seconds=120,
            enabled=True,
            last_test_status="healthy",
        )
        db.add(original)
        db.flush()
        db.add(
            OperationPlan(
                id="existing-plan", provider_id=original.id, summary="old", diagnosis="old"
            )
        )
        db.commit()
        sync._apply(db, snapshot())
        provider = db.get(Provider, "original-provider")
        assert provider.base_url == GATEWAY_BASE
        assert provider.default_model == "deepseek/deepseek-v4-flash"
        assert provider.timeout_seconds == 120
        assert box.decrypt(provider.encrypted_api_key) == "server-only-gateway-key"
        assert provider.last_test_status is None
        assert db.get(OperationPlan, "existing-plan").provider_id == provider.id
        revision = provider.config_revision
        provider.last_test_status = "healthy"
        db.commit()
        sync._apply(db, snapshot(name="Renamed DeepSeek"))
        assert provider.name == "Renamed DeepSeek"
        assert provider.config_revision == revision
        assert provider.last_test_status == "healthy"
        assert len(list(db.scalars(select(Provider)))) == 1


def test_sync_retains_default_then_changes_it_when_removed_and_disables_deleted_provider(client):
    sync = MagpieProviderSync(client.app.state.secret_box, "web-key", "gateway-key")
    with client.app.state.database.session_factory() as db:
        sync._apply(db, snapshot(models=("first", "second")))
        provider = db.scalar(select(Provider))
        provider_id = provider.id
        sync._apply(db, snapshot(models=("second", "first")))
        assert provider.default_model == "deepseek/first"
        sync._apply(db, snapshot(models=("second",)))
        assert provider.default_model == "deepseek/second"
        sync._apply(db, [])
        assert db.get(Provider, provider_id).enabled is False
        sync._apply(db, snapshot())
        assert db.get(Provider, provider_id).enabled is True
        assert len(list(db.scalars(select(Provider)))) == 1


def test_sync_excludes_local_and_non_chat_models(client):
    sync = MagpieProviderSync(client.app.state.secret_box, "web-key", "gateway-key")
    data = snapshot()
    data[0]["models"] = [
        {"id": "image", "on": True, "kind": "image"},
        {"id": "video", "on": True, "kind": "video"},
        {"id": "unroutable", "on": True, "noAPI": True},
        {"id": "hidden", "on": False},
        {"id": "org/text", "on": True},
    ]
    data.append({**snapshot()[0], "id": "dgx-local"})
    with client.app.state.database.session_factory() as db:
        sync._apply(db, data)
        providers = list(db.scalars(select(Provider)))
        assert len(providers) == 1
        assert providers[0].default_model == "deepseek/org/text"


def test_failed_snapshot_retains_previous_enabled_provider(client, monkeypatch):
    sync = MagpieProviderSync(client.app.state.secret_box, "web-key", "gateway-key")
    monkeypatch.setattr(sync, "_fetch", Mock(side_effect=httpx.ConnectError("offline")))
    with client.app.state.database.session_factory() as db:
        sync._apply(db, snapshot())
        assert sync.sync(db) is False
        assert db.scalar(select(Provider)).enabled is True


def test_fetch_uses_server_cookie_and_rejects_malformed_snapshot(respx_mock, client):
    sync = MagpieProviderSync(client.app.state.secret_box, "server-web-key", "gateway-key")
    route = respx_mock.get("http://127.0.0.1:3430/api/providers").mock(
        return_value=httpx.Response(200, json={"providers": snapshot()}),
    )
    assert sync._fetch()[0]["id"] == "deepseek"
    assert route.calls.last.request.headers["cookie"] == "magpie_web_3430=server-web-key"
    route.mock(return_value=httpx.Response(200, json={"providers": [{"id": "bad"}]}))
    with pytest.raises(ValueError, match="snapshot"):
        sync._fetch()


def test_magpie_resolver_allows_only_two_fixed_internal_endpoints():
    for path in ("models", "chat/completions"):
        endpoint = resolve_magpie_provider_endpoint(f"{GATEWAY_BASE}/{path}")
        assert endpoint.url == f"{GATEWAY_BASE}/{path}"
        assert endpoint.host_header == "127.0.0.1:3425"
    for url in (
        GATEWAY_BASE + "/models?target=http://localhost",
        GATEWAY_BASE + "/other",
        "http://127.0.0.1:3430/api/providers",
        "http://localhost:3425/v1/models",
        "http://169.254.169.254/latest/meta-data",
    ):
        with pytest.raises(ValueError):
            resolve_magpie_provider_endpoint(url)
    with pytest.raises(ValueError):
        validate_provider_url(GATEWAY_BASE)


def test_magpie_diagnosis_tests_default_model_on_first_use(client, monkeypatch):
    service = client.app.state.diagnostic_service
    sync = MagpieProviderSync(client.app.state.secret_box, "web-key", "gateway-key")
    probe = Mock(return_value={})
    monkeypatch.setattr(
        service.provider_service.ops_provider_client,
        "list_models",
        Mock(return_value=["deepseek/deepseek-v4-flash"]),
    )
    monkeypatch.setattr(service.provider_service.ops_provider_client, "complete", probe)
    with client.app.state.database.session_factory() as db:
        sync._apply(db, snapshot())
        provider = db.scalar(select(Provider))
        service.ensure_provider_ready(db, provider)
        service.ensure_provider_ready(db, provider)
        assert provider.last_test_status == "healthy"
    assert probe.call_count == 1


def test_provider_list_exposes_sync_status_without_secrets(authenticated_client):
    sync = Mock()
    sync.sync.return_value = False
    authenticated_client.app.state.magpie_provider_sync = sync
    result = authenticated_client.get("/api/providers")
    assert result.status_code == 200
    assert result.headers["X-Magpie-Sync"] == "unavailable"
    assert result.json() == []
