import respx
from app.api import gateway
from app.models import Deployment, RequestMetric
from httpx import Response


def _create_gateway_key(client):
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
    return client.post("/api/keys", json={"name": "Gateway test"}).json()["key"]


def _seed_deployment(client):
    with client.app.state.database.session_factory() as db:
        deployment = Deployment(
            name="qwen",
            runtime="sglang",
            endpoint_url="http://127.0.0.1:8001",
            api_model_name="qwen-upstream",
            status="running",
            health="healthy",
            managed=False,
            capabilities=["chat", "completion"],
        )
        db.add(deployment)
        db.commit()


def test_models_requires_api_key_and_lists_healthy_deployments(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)

    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen-upstream"


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
