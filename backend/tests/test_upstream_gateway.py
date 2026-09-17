import respx
from httpx import Response


def _create_gateway_key(client):
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
    return client.post("/api/keys", json={"name": "Upstream test"}).json()["key"]


@respx.mock
def test_fallback_records_prompt_and_completion_tokens(client, settings):
    settings.fallback_base_url = "https://upstream.test/v1"
    settings.fallback_api_key = "upstream-secret"
    key = _create_gateway_key(client)
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 13},
            },
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "remote-only-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 200

    with client.app.state.database.session_factory() as db:
        from app.models import RequestMetric

        metric = db.query(RequestMetric).order_by(RequestMetric.created_at.desc()).first()
        assert metric.prompt_tokens == 21
        assert metric.completion_tokens == 13


def test_upstream_api_root_accepts_both_base_url_forms():
    from app.services.upstream_gateway import upstream_api_root

    assert upstream_api_root("https://up.test/v1") == "https://up.test"
    assert upstream_api_root("https://up.test") == "https://up.test"
    assert upstream_api_root("https://up.test/v1/") == "https://up.test"
    assert upstream_api_root("https://up.test/openai/v1") == "https://up.test/openai"


def test_upstream_request_url_never_doubles_the_v1_prefix():
    from app.services.upstream_gateway import upstream_request_url

    for base in ("https://up.test", "https://up.test/v1", "https://up.test/v1/"):
        assert (
            upstream_request_url(base, "/v1/chat/completions")
            == "https://up.test/v1/chat/completions"
        )
    assert upstream_request_url("https://up.test", "/v1/models") == "https://up.test/v1/models"


@respx.mock
def test_fallback_stream_records_token_usage(client, settings):
    settings.fallback_base_url = "https://upstream.test"
    settings.fallback_api_key = "upstream-secret"
    key = _create_gateway_key(client)
    sse = (
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "remote-only-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 200
    assert b'"content":"he"' in response.content

    with client.app.state.database.session_factory() as db:
        from app.models import RequestMetric

        metric = db.query(RequestMetric).order_by(RequestMetric.created_at.desc()).first()
        assert metric.prompt_tokens == 5
        assert metric.completion_tokens == 2
