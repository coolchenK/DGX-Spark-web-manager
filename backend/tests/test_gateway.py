import json

import respx
from app.api import gateway
from app.gateway import proxy as gateway_proxy
from app.models import AuditEvent, Deployment, RequestMetric
from httpx import Response
from sqlalchemy import select


def _create_gateway_key(client):
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
    return client.post("/api/keys", json={"name": "Gateway test"}).json()["key"]


def _seed_deployment(client, *, config=None, capabilities=None):
    with client.app.state.database.session_factory() as db:
        deployment = Deployment(
            name="qwen",
            runtime="sglang",
            endpoint_url="http://127.0.0.1:8001",
            api_model_name="qwen-upstream",
            status="running",
            health="healthy",
            managed=False,
            config=config or {},
            capabilities=capabilities or ["chat", "completion"],
        )
        db.add(deployment)
        db.commit()


def _generation_config(defaults, supported):
    return {
        "spec": {"generation_defaults": defaults},
        "runtime_capabilities": {"generation_defaults": supported},
    }


def test_models_requires_api_key_and_lists_healthy_deployments(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)

    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen-upstream"


def test_models_exposes_runtime_context_and_generation_metadata_and_hides_stopped(client):
    key = _create_gateway_key(client)
    with client.app.state.database.session_factory() as db:
        db.add_all(
            [
                Deployment(
                    name="qwen-metadata",
                    runtime="vllm",
                    endpoint_url="http://127.0.0.1:8012",
                    api_model_name="qwen38-upstream",
                    status="running",
                    health="healthy",
                    config={
                        "spec": {
                            "context_length": 262144,
                            "generation_defaults": {
                                "temperature": 0.0,
                                "top_p": 1.0,
                                "top_k": 20,
                                "max_tokens": 8192,
                            },
                        }
                    },
                ),
                Deployment(
                    name="stopped-metadata",
                    runtime="vllm",
                    endpoint_url="http://127.0.0.1:8013",
                    api_model_name="stopped-upstream",
                    status="stopped",
                    health="unknown",
                    config={"spec": {"context_length": 4096}},
                ),
            ]
        )
        db.commit()

    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["data"]}
    assert models["qwen38-upstream"] == {
        "id": "qwen38-upstream",
        "object": "model",
        "created": models["qwen38-upstream"]["created"],
        "owned_by": "dgx-spark-manager",
        "root": "qwen38-upstream",
        "capabilities": [],
        "instances": 1,
        "runtime": "vllm",
        "endpoint_url": "http://127.0.0.1:8012",
        "context_length": 262144,
        "max_model_len": 262144,
        "max_context_tokens": 262144,
        "max_output_tokens": 8192,
        "generation_defaults": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": 8192,
        },
    }
    assert "stopped-upstream" not in models


def test_models_does_not_overwrite_shared_route_metadata_with_second_instance(client):
    key = _create_gateway_key(client)
    with client.app.state.database.session_factory() as db:
        for name, endpoint, context in (
            ("shared-a", "http://127.0.0.1:8011", 8192),
            ("shared-b", "http://127.0.0.1:8012", 262144),
        ):
            db.add(
                Deployment(
                    name=name,
                    runtime="vllm",
                    endpoint_url=endpoint,
                    api_model_name=f"{name}-upstream",
                    status="running",
                    health="healthy",
                    capabilities=["chat"],
                    config={
                        "route_alias": "shared-route",
                        "spec": {"context_length": context},
                    },
                )
            )
        db.commit()

    models = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {key}"}
    ).json()["data"]
    shared = next(item for item in models if item["id"] == "shared-route")

    assert shared["instances"] == 2
    assert shared["endpoint_url"] == "http://127.0.0.1:8011"
    assert shared["context_length"] == 8192


@respx.mock
def test_chat_completions_routes_alias_and_records_usage(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert route.calls[0].request.content.find(b'"model":"qwen-upstream"') >= 0


def test_unknown_model_uses_openai_error_shape(client):
    key = _create_gateway_key(client)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "missing", "messages": []},
    )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_shared_route_alias_round_robins_across_healthy_instances(client):
    with client.app.state.database.session_factory() as db:
        first = Deployment(
            name="qwen-a",
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8101",
            api_model_name="qwen-instance-a",
            status="running",
            health="healthy",
            capabilities=["chat"],
            config={"route_alias": "qwen-shared"},
        )
        second = Deployment(
            name="qwen-b",
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8102",
            api_model_name="qwen-instance-b",
            status="running",
            health="healthy",
            capabilities=["chat"],
            config={"route_alias": "qwen-shared"},
        )
        db.add_all([first, second])
        db.commit()
        select_route = getattr(gateway, "select_routed_deployment", lambda *_args: None)

        selected = [select_route(db, "qwen-shared") for _ in range(4)]

    selected_ids = [item.id for item in selected]
    assert set(selected_ids[:2]) == {first.id, second.id}
    assert selected_ids[0] == selected_ids[2]
    assert selected_ids[1] == selected_ids[3]


def test_gateway_activity_tracks_concurrent_requests():
    activity_type = getattr(gateway, "GatewayActivity", None)

    assert activity_type is not None
    activity = activity_type()
    activity.start()
    activity.start()
    assert activity.current == 2
    activity.finish()
    assert activity.current == 1


def test_gateway_stats_include_recent_request_and_token_throughput(authenticated_client):
    with authenticated_client.app.state.database.session_factory() as db:
        db.add(
            RequestMetric(
                model="qwen",
                endpoint="/v1/chat/completions",
                status_code=200,
                latency_ms=100,
                prompt_tokens=120,
                completion_tokens=60,
            )
        )
        db.commit()

    response = authenticated_client.get("/api/gateway/stats")

    assert response.status_code == 200
    assert response.json()["requests_last_minute"] == 1
    assert response.json()["tokens_per_second"] == 3.0
    assert response.json()["active_requests"] == 0


def test_merge_generation_defaults_preserves_explicit_false_zero_and_empty_stop():
    merge = getattr(gateway_proxy, "merge_generation_defaults", None)
    assert merge is not None
    body = {"temperature": 0, "top_p": False, "stop": [], "messages": []}
    defaults = {
        "temperature": 0.6,
        "top_p": 0.95,
        "min_p": 0.05,
        "stop": ["END"],
    }

    merged, applied = merge(
        "/v1/chat/completions", body, defaults, supported=set(defaults)
    )

    assert merged["temperature"] == 0
    assert merged["top_p"] is False
    assert merged["stop"] == []
    assert merged["min_p"] == 0.05
    assert applied == ["min_p"]
    assert body == {"temperature": 0, "top_p": False, "stop": [], "messages": []}


def test_max_completion_tokens_prevents_default_max_tokens():
    merge = getattr(gateway_proxy, "merge_generation_defaults", None)
    assert merge is not None

    merged, applied = merge(
        "/v1/chat/completions",
        {"max_completion_tokens": 100},
        {"max_tokens": 500},
        supported={"max_tokens"},
    )

    assert "max_tokens" not in merged
    assert applied == []


def test_merge_generation_defaults_filters_unknown_and_unsupported_extensions():
    merge = getattr(gateway_proxy, "merge_generation_defaults", None)
    assert merge is not None

    merged, applied = merge(
        "/v1/completions",
        {"prompt": "hello"},
        {"temperature": 0.6, "top_k": 40, "min_p": 0.05, "future_sampling": 1},
        supported={"temperature", "top_k", "future_sampling"},
    )

    assert merged["temperature"] == 0.6
    assert merged["top_k"] == 40
    assert "min_p" not in merged
    assert "future_sampling" not in merged
    assert applied == ["temperature", "top_k"]


def test_merge_generation_defaults_leaves_non_generation_endpoints_unchanged():
    merge = getattr(gateway_proxy, "merge_generation_defaults", None)
    assert merge is not None
    body = {"input": "hello"}

    merged, applied = merge(
        "/v1/embeddings",
        body,
        {"temperature": 0.6},
        supported={"temperature"},
    )

    assert merged == body
    assert merged is not body
    assert applied == []


@respx.mock
def test_chat_proxy_applies_saved_defaults_and_records_one_bounded_audit(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config=_generation_config(
            {"temperature": 0.6, "top_p": 0.9, "top_k": 40, "unknown": "secret"},
            ["temperature", "top_p", "unknown"],
        ),
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "temperature": 0,
            "messages": [{"role": "user", "content": "audit-secret-prompt"}],
        },
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["temperature"] == 0
    assert forwarded["top_p"] == 0.9
    assert "top_k" not in forwarded
    assert "unknown" not in forwarded
    with client.app.state.database.session_factory() as db:
        events = list(
            db.scalars(
                select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply")
            )
        )
    assert len(events) == 1
    assert events[0].actor == "gateway"
    assert events[0].resource_type == "deployment"
    assert events[0].resource_id is not None
    assert events[0].details == {
        "endpoint": "/v1/chat/completions",
        "model": "qwen-upstream",
        "applied_fields": ["top_p"],
    }
    assert "audit-secret-prompt" not in json.dumps(events[0].details)
    assert "0.9" not in json.dumps(events[0].details)


@respx.mock
def test_completion_proxy_applies_saved_generation_defaults(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config=_generation_config({"max_tokens": 128}, ["max_tokens"]),
    )
    route = respx.post("http://127.0.0.1:8001/v1/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "prompt": "hello"},
    )

    assert response.status_code == 200
    assert json.loads(route.calls[0].request.content)["max_tokens"] == 128


@respx.mock
def test_embeddings_proxy_does_not_apply_or_audit_generation_defaults(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config=_generation_config({"temperature": 0.6}, ["temperature"]),
        capabilities=["embedding"],
    )
    route = respx.post("http://127.0.0.1:8001/v1/embeddings").mock(
        return_value=Response(200, json={"data": [], "usage": {}})
    )

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "input": "hello"},
    )

    assert response.status_code == 200
    assert "temperature" not in json.loads(route.calls[0].request.content)
    with client.app.state.database.session_factory() as db:
        event = db.scalar(
            select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply")
        )
    assert event is None


@respx.mock
def test_streaming_chat_proxy_uses_the_same_generation_default_merge(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config=_generation_config({"top_p": 0.8}, ["top_p"]),
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=b'data: {"choices": []}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert json.loads(route.calls[0].request.content)["top_p"] == 0.8


@respx.mock
def test_gateway_ignores_defaults_without_a_valid_capability_snapshot(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {"temperature": 0.6}},
            "runtime_capabilities": {"generation_defaults": "temperature"},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": []},
    )

    assert response.status_code == 200
    assert "temperature" not in json.loads(route.calls[0].request.content)


@respx.mock
def test_gateway_strictly_skips_coercible_defaults_without_dropping_valid_fields(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config=_generation_config(
            {
                "temperature": True,
                "top_p": "0.7",
                "max_tokens": True,
                "min_p": 0.05,
            },
            ["temperature", "top_p", "max_tokens", "min_p"],
        ),
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": []},
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert "temperature" not in forwarded
    assert "top_p" not in forwarded
    assert "max_tokens" not in forwarded
    assert forwarded["min_p"] == 0.05
    with client.app.state.database.session_factory() as db:
        event = db.scalar(
            select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply")
        )
    assert event is not None
    assert event.details["applied_fields"] == ["min_p"]


@respx.mock
def test_gateway_does_not_audit_when_all_saved_defaults_fail_strict_validation(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config=_generation_config(
            {"temperature": True, "top_p": "0.7", "max_tokens": True},
            ["temperature", "top_p", "max_tokens"],
        ),
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": []},
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert "temperature" not in forwarded
    assert "top_p" not in forwarded
    assert "max_tokens" not in forwarded
    with client.app.state.database.session_factory() as db:
        event = db.scalar(
            select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply")
        )
    assert event is None
