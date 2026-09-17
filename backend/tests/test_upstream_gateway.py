import pytest
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


def test_valid_base_url_is_normalised():
    from app.services.upstream_gateway import validate_upstream_base_url

    assert (
        validate_upstream_base_url("  https://upstream.test/v1/  ")
        == "https://upstream.test/v1"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "ftp://upstream.test/v1",
        "https://",
        "https://user:pw@upstream.test/v1",
        "https://upstream.test/v1#fragment",
        "https://" + "a" * 600,
    ],
)
def test_invalid_base_urls_are_rejected(value):
    from app.services.upstream_gateway import validate_upstream_base_url

    with pytest.raises(ValueError):
        validate_upstream_base_url(value)

def test_upstream_endpoints_require_admin(client):
    assert client.get("/api/gateway/upstream").status_code == 401
    assert client.put("/api/gateway/upstream", json={"base_url": None}).status_code == 401
    assert client.post("/api/gateway/upstream/test").status_code == 401


def test_put_requires_csrf(client):
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    assert login.status_code == 200

    response = client.put("/api/gateway/upstream", json={"base_url": "https://x.test/v1"})

    assert response.status_code == 403


def test_put_rejects_an_invalid_base_url(authenticated_client):
    response = authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "ftp://bad/v1"}
    )

    assert response.status_code == 422


def test_get_reports_unset_when_nothing_is_configured(authenticated_client):
    assert authenticated_client.get("/api/gateway/upstream").json() == {
        "base_url": None,
        "api_key_configured": False,
        "source": "unset",
        "enabled": False,
        # Offering every discovered upstream model is the default so an upgrade
        # never silently stops serving a model a client already uses.
        "expose_all": True,
        "selected_models": [],
    }


def test_database_configuration_wins_over_environment(authenticated_client, settings):
    settings.fallback_base_url = "https://env.test/v1"

    authenticated_client.put(
        "/api/gateway/upstream",
        json={"base_url": "https://db.test/v1", "api_key": "db-key"},
    )

    assert authenticated_client.get("/api/gateway/upstream").json() == {
        "base_url": "https://db.test/v1",
        "api_key_configured": True,
        "source": "database",
        "enabled": True,
        "expose_all": True,
        "selected_models": [],
    }


def test_clearing_database_configuration_falls_back_to_environment(
    authenticated_client, settings
):
    settings.fallback_base_url = "https://env.test/v1"
    settings.fallback_api_key = "env-key"
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://db.test/v1", "api_key": "db-key"}
    )

    authenticated_client.put("/api/gateway/upstream", json={"base_url": None})

    assert authenticated_client.get("/api/gateway/upstream").json() == {
        "base_url": "https://env.test/v1",
        "api_key_configured": True,
        "source": "environment",
        "enabled": True,
        "expose_all": True,
        "selected_models": [],
    }


def test_environment_configuration_is_reported_when_no_database_row_exists(
    authenticated_client, settings
):
    settings.fallback_base_url = "https://env-only.test/v1"

    body = authenticated_client.get("/api/gateway/upstream").json()

    assert body["source"] == "environment"
    assert body["api_key_configured"] is False


def test_upstream_never_returns_the_stored_key(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream",
        json={"base_url": "https://db.test/v1", "api_key": "super-secret-value"},
    )

    assert "super-secret-value" not in authenticated_client.get("/api/gateway/upstream").text


def test_omitting_api_key_keeps_the_stored_key(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://db.test/v1", "api_key": "first"}
    )

    authenticated_client.put("/api/gateway/upstream", json={"base_url": "https://db2.test/v1"})

    assert authenticated_client.get("/api/gateway/upstream").json()["api_key_configured"] is True


def test_empty_api_key_clears_only_the_key(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://db.test/v1", "api_key": "first"}
    )

    authenticated_client.put("/api/gateway/upstream", json={"api_key": ""})

    body = authenticated_client.get("/api/gateway/upstream").json()
    assert body["base_url"] == "https://db.test/v1"
    assert body["api_key_configured"] is False


def test_test_endpoint_reports_unset_when_not_configured(authenticated_client):
    assert authenticated_client.post("/api/gateway/upstream/test").json()["status"] == "unset"


@respx.mock
def test_test_endpoint_reports_model_count(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "a"}, {"id": "b"}]})
    )

    body = authenticated_client.post("/api/gateway/upstream/test").json()

    assert body["status"] == "ok"
    assert body["model_count"] == 2
    assert isinstance(body["latency_ms"], int)


@respx.mock
def test_test_endpoint_reports_unavailable_without_leaking_the_key(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "secret-key"}
    )
    respx.get("https://up.test/v1/models").mock(return_value=Response(503, text="boom"))

    body = authenticated_client.post("/api/gateway/upstream/test").json()

    assert body["status"] == "unavailable"
    assert "secret-key" not in str(body)


def test_config_change_is_audited(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://audit.test/v1", "api_key": "abc"}
    )

    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import AuditEvent

        actions = {row.action for row in db.query(AuditEvent).all()}

    assert "gateway.upstream.update" in actions


def test_config_change_invalidates_the_model_cache(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://cache.test/v1", "api_key": "k"}
    )
    cache = authenticated_client.app.state.upstream_model_cache
    cache.set("whatever", [{"id": "stale"}])

    authenticated_client.put("/api/gateway/upstream", json={"base_url": "https://cache2.test/v1"})

    assert cache.get("whatever") is None

def _seed_healthy_deployment(client, *, api_model_name="local-chat"):
    from app.models import Deployment

    with client.app.state.database.session_factory() as db:
        db.add(
            Deployment(
                name=api_model_name,
                # vLLM keeps the models endpoint free of the SGLang runtime
                # capacity probe; the merge logic is runtime independent.
                runtime="vllm",
                endpoint_url="http://127.0.0.1:8001",
                api_model_name=api_model_name,
                status="running",
                health="healthy",
                managed=False,
                config={},
                capabilities=["chat", "completion"],
            )
        )
        db.commit()
    return api_model_name


def _models(authenticated_client, key):
    return authenticated_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {key}"}
    ).json()


@respx.mock
def test_models_include_upstream_entries(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _seed_healthy_deployment(authenticated_client)
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(
            200, json={"object": "list", "data": [{"id": "remote-a"}, {"id": "remote-b"}]}
        )
    )

    data = _models(authenticated_client, key)
    ids = [item["id"] for item in data["data"]]
    upstream = [item for item in data["data"] if item.get("dgx_source") == "upstream"]

    assert {"local-chat", "remote-a", "remote-b"} <= set(ids)
    assert len(upstream) == 2
    assert all(item["owned_by"] == "upstream" for item in upstream)
    assert all(item["capabilities"] == [] for item in upstream)
    assert all(item["input_modalities"] == [] for item in upstream)
    assert all(item["context_window"] is None for item in upstream)
    assert data["upstream"] == {"status": "ok", "detail": None}


@respx.mock
def test_local_route_wins_over_a_same_named_upstream_model(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _seed_healthy_deployment(authenticated_client, api_model_name="shared-name")
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "shared-name"}]})
    )

    data = _models(authenticated_client, key)
    matches = [item for item in data["data"] if item["id"] == "shared-name"]

    assert len(matches) == 1
    assert matches[0].get("dgx_source") != "upstream"
    assert matches[0]["runtime"] == "vllm"


@respx.mock
def test_upstream_failure_keeps_local_models_available(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _seed_healthy_deployment(authenticated_client)
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(return_value=Response(503, text="boom"))

    data = _models(authenticated_client, key)

    assert [item["id"] for item in data["data"]] == ["local-chat"]
    assert data["upstream"]["status"] == "unavailable"


def test_models_report_unset_upstream(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _seed_healthy_deployment(authenticated_client)

    data = _models(authenticated_client, key)

    assert data["upstream"] == {"status": "unset", "detail": None}


@respx.mock
def test_upstream_models_come_from_cache_within_the_ttl(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _seed_healthy_deployment(authenticated_client)
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    route = respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "remote-a"}]})
    )

    _models(authenticated_client, key)
    _models(authenticated_client, key)

    assert route.call_count == 1


def _configure_upstream(authenticated_client, base_url="https://up.test/v1"):
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": base_url, "api_key": "k"}
    )


def _mock_models(ids, base_url="https://up.test/v1"):
    respx.get(f"{base_url}/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})
    )


def test_exposure_defaults_to_offering_every_upstream_model(authenticated_client):
    _configure_upstream(authenticated_client)
    _mock_models(["a", "b"])

    body = authenticated_client.get("/api/gateway/upstream").json()

    assert body["expose_all"] is True
    assert body["selected_models"] == []


@respx.mock
def test_upstream_models_endpoint_reports_exposure_state(authenticated_client):
    _configure_upstream(authenticated_client)
    _mock_models(["a", "b"])

    body = authenticated_client.get("/api/gateway/upstream/models").json()

    assert body["status"] == "ok"
    assert body["models"] == [
        {"id": "a", "exposed": True},
        {"id": "b", "exposed": True},
    ]


def test_upstream_models_endpoint_reports_unset_without_configuration(authenticated_client):
    assert authenticated_client.get("/api/gateway/upstream/models").json() == {
        "status": "unset",
        "models": [],
        "detail": None,
    }


@respx.mock
def test_upstream_models_endpoint_reports_unavailable_upstream(authenticated_client):
    _configure_upstream(authenticated_client)
    respx.get("https://up.test/v1/models").mock(return_value=Response(503))

    body = authenticated_client.get("/api/gateway/upstream/models").json()

    assert body["status"] == "unavailable"
    assert body["models"] == []


@respx.mock
def test_selected_exposure_hides_and_blocks_unselected_models(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _configure_upstream(authenticated_client)
    _mock_models(["keep-me", "hide-me"])

    updated = authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": ["keep-me"]},
    )
    assert updated.status_code == 200
    assert updated.json() == {"expose_all": False, "selected_models": ["keep-me"]}

    listed = _models(authenticated_client, key)
    upstream_ids = [m["id"] for m in listed["data"] if m.get("dgx_source") == "upstream"]
    assert upstream_ids == ["keep-me"]

    hidden = authenticated_client.get(
        "/v1/models/hide-me", headers={"Authorization": f"Bearer {key}"}
    )
    assert hidden.status_code == 404

    blocked = authenticated_client.post(
        "/v1/chat/completions",
        json={"model": "hide-me", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert blocked.status_code == 404
    assert "hide-me" in blocked.json()["error"]["message"]


@respx.mock
def test_selected_model_is_still_forwarded(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _configure_upstream(authenticated_client)
    _mock_models(["keep-me"])
    authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": ["keep-me"]},
    )
    respx.post("https://up.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )
    )

    response = authenticated_client.post(
        "/v1/chat/completions",
        json={"model": "keep-me", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 200


@respx.mock
def test_selected_exposure_also_blocks_undiscovered_models(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    _configure_upstream(authenticated_client)
    _mock_models(["keep-me"])
    authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": ["keep-me"]},
    )

    response = authenticated_client.post(
        "/v1/chat/completions",
        json={"model": "never-selected", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 404


def test_exposure_rejects_invalid_model_names(authenticated_client):
    response = authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": ["bad name!"]},
    )

    assert response.status_code == 422


def test_exposure_rejects_too_many_models(authenticated_client):
    response = authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": [f"m{i}" for i in range(501)]},
    )

    assert response.status_code == 422


def test_exposure_selection_persists_and_is_deduplicated(authenticated_client):
    authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": ["b", "a", "b"]},
    )

    assert authenticated_client.get("/api/gateway/upstream").json()["selected_models"] == ["a", "b"]


def test_exposure_change_is_audited_and_invalidates_the_cache(authenticated_client):
    cache = authenticated_client.app.state.upstream_model_cache
    cache.set("stale", [{"id": "x"}])

    authenticated_client.put(
        "/api/gateway/upstream/exposure",
        json={"expose_all": False, "selected_models": ["a"]},
    )

    assert cache.get("stale") is None
    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import AuditEvent

        actions = {row.action for row in db.query(AuditEvent).all()}
    assert "gateway.upstream.exposure.update" in actions


def test_exposure_requires_admin(client):
    assert (
        client.put(
            "/api/gateway/upstream/exposure",
            json={"expose_all": True, "selected_models": []},
        ).status_code
        == 401
    )
    assert client.get("/api/gateway/upstream/models").status_code == 401


def test_exposure_requires_csrf(client):
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    assert login.status_code == 200

    response = client.put(
        "/api/gateway/upstream/exposure", json={"expose_all": True, "selected_models": []}
    )

    assert response.status_code == 403



@respx.mock
def test_upstream_single_model_lookup_resolves(authenticated_client):
    key = _create_gateway_key(authenticated_client)
    authenticated_client.put(
        "/api/gateway/upstream", json={"base_url": "https://up.test/v1", "api_key": "k"}
    )
    respx.get("https://up.test/v1/models").mock(
        return_value=Response(200, json={"object": "list", "data": [{"id": "remote-only"}]})
    )

    response = authenticated_client.get(
        "/v1/models/remote-only", headers={"Authorization": f"Bearer {key}"}
    )

    assert response.status_code == 200
    assert response.json()["id"] == "remote-only"
    assert response.json()["dgx_source"] == "upstream"


