import httpx
import respx
from app.api import gateway
from app.models import Deployment


def enable(client):
    client.app.state.settings.magpie_enabled = True
    client.app.state.settings.magpie_bridge_key = "internal-local-key-not-a-client-key"


def test_native_keys_and_streaming_go_to_magpie_without_panel_secrets(client):
    enable(client)
    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:3425/v1/chat/completions").respond(
            200,
            content=b'data: {"ok":true}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "dgx/local", "stream": True},
            headers={
                "Authorization": "Bearer sk-magpie-key-example",
                "Cookie": "dgx_session=private",
                "X-CSRF-Token": "private",
                "X-Forwarded-For": "127.0.0.1",
            },
        )
    assert response.status_code == 200 and "[DONE]" in response.text
    forwarded = route.calls[0].request
    assert forwarded.headers["authorization"] == "Bearer sk-magpie-key-example"
    assert forwarded.headers["x-forwarded-for"] == "192.0.2.1"
    assert "cookie" not in forwarded.headers and "x-csrf-token" not in forwarded.headers
    assert response.headers["x-accel-buffering"] == "no"


def test_invalid_native_key_is_not_trusted_as_local(client):
    enable(client)
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:3425/v1/models").respond(401, json={"error": "invalid key"})
        assert (
            client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )


def test_bridge_requires_secret_and_never_discovers_upstream(client, monkeypatch):
    enable(client)
    monkeypatch.setattr(
        gateway,
        "resolve_upstream_gateway",
        lambda *a: (_ for _ in ()).throw(AssertionError("loop")),
    )
    assert client.get("/_magpie/local/v1/models").status_code == 401
    headers = {"Authorization": "Bearer " + client.app.state.settings.magpie_bridge_key}
    with client.app.state.database.session_factory() as db:
        db.add(
            Deployment(
                name="Local",
                runtime="vllm",
                endpoint_url="http://127.0.0.1:8003",
                api_model_name="local",
                status="running",
                health="healthy",
                managed=False,
                config={},
                capabilities=["chat"],
            )
        )
        db.commit()
    response = client.get("/_magpie/local/v1/models", headers=headers)
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == ["local"]
    unknown = client.post(
        "/_magpie/local/v1/chat/completions",
        json={"model": "stopped", "messages": []},
        headers=headers,
    )
    assert unknown.status_code == 404
    assert client.get("/_magpie/local/api/providers", headers=headers).status_code == 404


def test_legacy_keys_keep_existing_local_authentication(authenticated_client):
    client = authenticated_client
    enable(client)
    key = client.post("/api/keys", json={"name": "legacy"}).json()["key"]
    assert client.get("/v1/models", headers={"Authorization": "Bearer dgx_bad"}).status_code == 401
    with respx.mock() as mock:
        response = client.get("/v1/models", headers={"Authorization": "Bearer " + key})
        assert response.status_code == 200
        assert not mock.calls


def test_unavailable_magpie_returns_bounded_error(client):
    enable(client)
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:3425/v1/models").mock(side_effect=httpx.ConnectError("private"))
        response = client.get("/v1/models")
    assert response.status_code == 502 and "private" not in response.text


def test_legacy_inference_is_authenticated_before_magpie(authenticated_client):
    client = authenticated_client
    enable(client)
    client.app.state.settings.magpie_gateway_key = "sk-magpie-key-internal"
    key = client.post("/api/keys", json={"name": "legacy"}).json()["key"]
    with respx.mock() as mock:
        upstream = mock.post("http://127.0.0.1:3425/v1/chat/completions").respond(
            200, json={"choices": []}
        )
        assert (
            client.post(
                "/v1/chat/completions", json={}, headers={"Authorization": "Bearer dgx_bad"}
            ).status_code
            == 401
        )
        assert not mock.calls
        response = client.post(
            "/v1/chat/completions",
            json={"model": "local", "messages": []},
            headers={"Authorization": "Bearer " + key, "X-API-Key": "wrong"},
        )
    assert response.status_code == 200
    assert upstream.calls[0].request.headers["authorization"] == "Bearer sk-magpie-key-internal"
    assert "x-api-key" not in upstream.calls[0].request.headers


def test_live_local_route_works_before_magpie_catalog_refresh(client):
    enable(client)
    with client.app.state.database.session_factory() as db:
        db.add(
            Deployment(
                name="New local",
                runtime="vllm",
                endpoint_url="http://127.0.0.1:8011",
                api_model_name="new-local",
                status="running",
                health="healthy",
                managed=False,
                config={},
                capabilities=["chat"],
            )
        )
        db.commit()
    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:3425/v1/chat/completions").respond(
            200, json={"choices": []}
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "new-local", "messages": []},
            headers={"Authorization": "Bearer sk-magpie-key-example"},
        )
    assert response.status_code == 200
    import json

    assert json.loads(route.calls[0].request.content)["model"] == "dgx-local/new-local"
