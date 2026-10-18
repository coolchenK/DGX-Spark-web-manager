from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from app.api.magpie import WEB_URL, _rewrite_script, router
from app.security import SessionManager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

WEB_KEY = "test-magpie-web-key-kept-on-server"


@pytest.fixture
def magpie_client():
    app = FastAPI()
    app.state.settings = SimpleNamespace(magpie_enabled=True, magpie_web_key=WEB_KEY)
    app.state.session_manager = SessionManager("test-secret-for-magpie", 3600)
    app.include_router(router)
    with TestClient(app) as client:
        client.cookies.set("dgx_session", app.state.session_manager.create({"role": "admin"}))
        yield client


def test_every_magpie_endpoint_requires_panel_login(magpie_client):
    magpie_client.cookies.clear()
    for path in (
        "/api/magpie/status",
        "/magpie",
        "/magpie/",
        "/magpie/api/providers",
        "/magpie/app.js",
    ):
        assert magpie_client.get(path).status_code == 401


@respx.mock
def test_proxy_preserves_query_and_hides_internal_credentials(magpie_client):
    upstream = respx.get(f"{WEB_URL}/?view=gateway").respond(
        200,
        content='<script src="app.js"></script>',
        headers={"content-type": "text/html", "set-cookie": f"magpie_web_3430={WEB_KEY}"},
    )
    response = magpie_client.get(
        "/magpie/?view=gateway&k=client-supplied-key",
        headers={"Authorization": "Bearer panel-secret", "X-Forwarded-For": "203.0.113.8"},
    )
    assert response.status_code == 200
    sent = upstream.calls.last.request.headers
    assert sent["cookie"] == f"magpie_web_3430={WEB_KEY}"
    assert "authorization" not in sent
    assert "x-forwarded-for" not in sent
    assert "set-cookie" not in response.headers
    assert WEB_KEY not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "SAMEORIGIN"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "null"},
        {"Origin": "https://evil.test"},
        {"Origin": "http://testserver", "Sec-Fetch-Site": "cross-site"},
        {
            "Origin": "https://evil.test",
            "X-Forwarded-Host": "evil.test",
            "X-Forwarded-Proto": "https",
        },
    ],
)
@respx.mock
def test_mutations_reject_missing_or_cross_origin(magpie_client, headers):
    response = magpie_client.post(
        "/magpie/api/provider/save", json={"name": "Bad"}, headers=headers
    )
    assert response.status_code == 403
    assert not respx.calls


@respx.mock
def test_same_origin_provider_save_is_forwarded(magpie_client):
    upstream = respx.post(f"{WEB_URL}/api/provider/save").respond(200, json={"providers": []})
    response = magpie_client.post(
        "/magpie/api/provider/save", json={"name": "Local"}, headers={"Origin": "http://testserver"}
    )
    assert response.status_code == 200
    assert b'"name":"Local"' in upstream.calls.last.request.content


def test_loopback_proxy_origin_is_supported(magpie_client):
    with TestClient(magpie_client.app, client=("127.0.0.1", 1234)) as client, respx.mock:
        client.cookies.update(magpie_client.cookies)
        respx.post(f"{WEB_URL}/api/provider/save").respond(200, json={})
        response = client.post(
            "/magpie/api/provider/save",
            json={},
            headers={
                "Origin": "https://panel.test",
                "X-Forwarded-Host": "panel.test",
                "X-Forwarded-Proto": "https",
            },
        )
        assert response.status_code == 200


def test_script_rewrite_keeps_provider_api_path_semantics():
    source = b"""fetch("/api/providers"); fetch(`/api/gateway/trace?after=${seq}`);
img.src = "/api/icons/" + name; a.href = '/api/archive/file?';
if (path === "/api/user/self") return 1;
return site + "/api/usage/token";
const example = "https://vendor.test/api/foo";"""
    output = _rewrite_script(source)
    assert b'fetch("/magpie/api/providers")' in output
    assert b"fetch(`/magpie/api/gateway/trace?" in output
    assert b'img.src = "/magpie/api/icons/' in output
    assert b"a.href = '/magpie/api/archive/file?" in output
    assert b'path === "/api/user/self"' in output
    assert b'site + "/api/usage/token"' in output
    assert b"https://vendor.test/api/foo" in output


@respx.mock
def test_javascript_rewritten_on_response(magpie_client):
    respx.get(f"{WEB_URL}/app.js").respond(
        200,
        content='fetch("/api/providers")',
        headers={"content-type": "text/javascript; charset=utf-8"},
    )
    response = magpie_client.get("/magpie/app.js")
    assert response.text == 'fetch("/magpie/api/providers")'
    assert int(response.headers["content-length"]) == len(response.content)


@respx.mock
def test_status_reports_availability_without_secrets(magpie_client):
    route = respx.get(f"{WEB_URL}/boot.js").respond(200, text="window.bootPrefs = {};")
    assert magpie_client.get("/api/magpie/status").json() == {"status": "ok"}
    route.mock(side_effect=httpx.ConnectError("internal details"))
    response = magpie_client.get("/api/magpie/status")
    assert response.json()["status"] == "unavailable"
    assert "internal details" not in response.text


@respx.mock
def test_proxy_failure_does_not_expose_internal_key(magpie_client):
    respx.get(f"{WEB_URL}/").mock(side_effect=httpx.ConnectError(WEB_KEY))
    response = magpie_client.get("/magpie/")
    assert response.status_code == 502
    assert WEB_KEY not in response.text


@respx.mock
def test_redirect_is_confined_to_protected_prefix(magpie_client):
    upstream = respx.get(f"{WEB_URL}/icons").respond(301, headers={"location": "/icons/"})
    response = magpie_client.get("/magpie/icons", follow_redirects=False)
    assert response.headers["location"] == "/magpie/icons/"
    upstream.respond(302, headers={"location": "https://elsewhere.test"})
    assert magpie_client.get("/magpie/icons", follow_redirects=False).status_code == 502


@respx.mock
def test_disabled_service_does_not_proxy(magpie_client):
    magpie_client.app.state.settings.magpie_enabled = False
    assert magpie_client.get("/magpie/").status_code == 503
    assert magpie_client.get("/api/magpie/status").json()["status"] == "unavailable"
    assert not respx.calls


def test_disconnect_while_sending_request_closes_upstream(magpie_client, monkeypatch):
    send = AsyncMock(side_effect=ClientDisconnect())
    close = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    response = magpie_client.get("/magpie/")
    assert response.status_code == 499
    close.assert_awaited_once()
