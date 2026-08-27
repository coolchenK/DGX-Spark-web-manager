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
    model = models["qwen38-upstream"]
    assert model["id"] == "qwen38-upstream"
    assert model["object"] == "model"
    assert model["owned_by"] == "dgx-spark-manager"
    assert model["root"] == "qwen38-upstream"
    assert model["capability_names"] == []
    assert model["instances"] == 1
    assert model["runtime"] == "vllm"
    assert model["endpoint_url"] == "http://127.0.0.1:8012"
    assert model["context_length"] == 262144
    assert model["max_model_len"] == 262144
    assert model["max_context_tokens"] == 262144
    assert model["context_window"] == 262144
    assert model["max_input_tokens"] == 253952
    assert model["max_output_tokens"] == 8192
    assert model["max_tokens"] == 8192
    assert model["generation_defaults"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "max_tokens": 8192,
    }
    assert model["performance"]["status"] == "unavailable"
    assert "stopped-upstream" not in models


def test_models_exposes_discovery_aliases_and_benchmark_performance_metadata(client):
    key = _create_gateway_key(client)
    with client.app.state.database.session_factory() as db:
        db.add(
            Deployment(
                name="metadata-compatible",
                runtime="sglang",
                endpoint_url="http://127.0.0.1:8019",
                api_model_name="metadata-compatible-upstream",
                status="running",
                health="healthy",
                capabilities=["chat", "completion"],
                benchmark_status="succeeded",
                benchmark_tps=78.018,
                benchmark_completion_tokens=256,
                benchmark_duration_seconds=3.281,
                config={
                    "route_alias": "metadata-compatible",
                    "spec": {
                        "context_length": 204800,
                        "max_concurrency": 2,
                        "generation_defaults": {"max_tokens": 16384},
                    },
                },
            )
        )
        db.commit()

    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    model = next(item for item in response.json()["data"] if item["id"] == "metadata-compatible")
    assert model["context_length"] == 204800
    assert model["max_model_len"] == 204800
    assert model["max_context_tokens"] == 204800
    assert model["context_window"] == 204800
    assert model["max_input_tokens"] == 188416
    assert model["max_output_tokens"] == 16384
    assert model["max_tokens"] == 16384
    assert model["output_token_limit"] == 16384
    assert model["max_concurrency"] == 2
    assert model["benchmark_tps"] == 78.018
    assert model["tokens_per_second"] == 78.018
    assert model["performance"]["tokens_per_second"] == 78.018
    assert model["performance"]["completion_tokens"] == 256
    assert model["performance"]["duration_seconds"] == 3.281
    assert model["performance"]["status"] == "succeeded"
    assert model["metadata"]["context_window"] == 204800
    assert model["metadata"]["max_output_tokens"] == 16384
    assert model["metadata"]["tokens_per_second"] == 78.018
    assert model["metadata"]["runtime"] == "sglang"
    assert model["limits"] == {
        "context_window": 204800,
        "max_input_tokens": 188416,
        "max_output_tokens": 16384,
        "max_concurrency": 2,
    }


def test_models_exposes_alma_opencode_go_fields_on_standard_route(client):
    key = _create_gateway_key(client)
    with client.app.state.database.session_factory() as db:
        db.add(
            Deployment(
                name="Go Compatible Formal Display Name",
                runtime="vllm",
                endpoint_url="http://127.0.0.1:8020",
                api_model_name="go-compatible-upstream",
                status="running",
                health="healthy",
                capabilities=["chat", "completion"],
                benchmark_status="succeeded",
                benchmark_tps=27.892,
                config={
                    "route_alias": "go-compatible",
                    "spec": {
                        "context_length": 204800,
                        "max_concurrency": 2,
                        "generation_defaults": {"max_tokens": 16384},
                    },
                },
            )
        )
        db.commit()

    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["id"] == "go-compatible"
    assert model["name"] == "Go Compatible Formal Display Name"
    assert model["display_name"] == "Go Compatible Formal Display Name"
    assert model["limit"] == {
        "context": 204800,
        "input": 188416,
        "output": 16384,
    }
    assert model["context_length"] == 204800
    assert model["max_output_tokens"] == 16384
    assert model["apiFormat"] == "openai-chat"
    assert model["api"] == "openai-compatible chat/completions"
    assert model["tool_call"] is True
    assert model["structured_output"] is True
    assert model["reasoning"] is True
    assert model["performance"]["tokens_per_second"] == 27.892


def test_models_reports_null_performance_when_no_benchmark_exists(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {
                "context_length": 8192,
                "generation_defaults": {"max_tokens": 1024},
            }
        },
    )

    response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    model = response.json()["data"][0]
    assert model["benchmark_tps"] is None
    assert model["tokens_per_second"] is None
    assert model["performance"]["status"] == "unavailable"
    assert model["performance"]["tokens_per_second"] is None


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

    models = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).json()["data"]
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

    merged, applied = merge("/v1/chat/completions", body, defaults, supported=set(defaults))

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
def test_gateway_extracts_alma_xml_tools_from_system_prompt(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )
    tool_one = {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "run",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
    tool_two = {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    system = (
        "instructions\n<tools> "
        + json.dumps(tool_one)
        + "\n"
        + json.dumps(tool_two)
        + " </tools>\nmore"
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "search"},
            ],
            "stream": True,
        },
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert [t["function"]["name"] for t in forwarded["tools"]] == ["Bash", "WebSearch"]
    assert forwarded["tool_choice"] == "auto"
    assert "<tools>" not in forwarded["messages"][0]["content"]


@respx.mock
def test_gateway_normalizes_alma_reasoning_effort_for_qwen_templates(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"reasoning_effort": "high"},
        },
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["reasoning_effort"] == "medium"
    assert forwarded["chat_template_kwargs"]["reasoning_effort"] == "medium"


@respx.mock
def test_gateway_normalizes_alma_reasoning_effort_for_nested_opencode_go_options(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [{"role": "user", "content": "hi"}],
            "provider_options": {
                "opencode-go": {"reasoningEffort": "high"},
                "openai": {"reasoningEffort": "high"},
            },
        },
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["reasoning_effort"] == "medium"
    assert "provider_options" not in forwarded


@respx.mock
def test_gateway_preserves_qwen_fixed_template_effort_levels(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {
                "chat_template": "qwen-fixed-v22.4",
                "generation_defaults": {},
            },
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [{"role": "user", "content": "hi"}],
            "provider_options": {"opencode-go": {"reasoningEffort": "MAX"}},
            "chat_template_kwargs": {"reasoning_effort": "minimal"},
        },
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["reasoning_effort"] == "xhigh"
    assert forwarded["chat_template_kwargs"]["reasoning_effort"] == "low"


@respx.mock
def test_gateway_allows_data_tool_after_skill_metadata_response(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )
    skill = {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "load skill",
            "parameters": {"type": "object", "properties": {"skill": {"type": "string"}}},
        },
    }
    web = {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
    system = "<tools> " + json.dumps(skill) + "\n" + json.dumps(web) + " </tools>"
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"Skill","arguments":{}}</tool_call>',
                },
                {
                    "role": "user",
                    "content": (
                        '<tool_response>{"name":"Skill",'
                        '"content":"web search instructions"}</tool_response>'
                    ),
                },
            ],
        },
    )
    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert [t["function"]["name"] for t in forwarded["tools"]] == ["WebSearch"]
    assert forwarded["tool_choice"] == "auto"
    assert "现在必须向用户给出最终回答" not in forwarded["messages"][-1]["content"]


@respx.mock
def test_gateway_enriches_empty_huggingface_search_result_from_live_api(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )
    hf = respx.get("https://huggingface.co/api/models").mock(
        return_value=Response(
            200,
            json=[
                {
                    "id": "Qwen/Qwen3.8-Flash-Next-FP8",
                    "lastModified": "2026-08-26T11:55:24.000Z",
                    "tags": ["fp8", "base_model:quantized:Qwen/Qwen3.8-Flash-Next"],
                },
                {
                    "id": "unsloth/Qwen3.8-Flash-Next-GGUF",
                    "lastModified": "2026-08-26T15:54:43.000Z",
                    "tags": ["gguf", "base_model:quantized:Qwen/Qwen3.8-Flash-Next"],
                },
            ],
        )
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [
                {"role": "user", "content": "搜索 Qwen3.8-Flash-Next 量化版本"},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"WebSearch","arguments":{}}</tool_call>',
                },
                {
                    "role": "user",
                    "content": (
                        '<tool_response>{"name":"WebSearch","content":'
                        '{"query":"Qwen3.8-Flash-Next quantized HuggingFace",'
                        '"results":[]}}</tool_response>'
                    ),
                },
            ],
        },
    )
    assert response.status_code == 200
    assert hf.called
    forwarded = json.loads(route.calls[0].request.content)
    serialized = json.dumps(forwarded["messages"], ensure_ascii=False)
    assert "Qwen/Qwen3.8-Flash-Next-FP8" in serialized
    assert "unsloth/Qwen3.8-Flash-Next-GGUF" in serialized
    assert "实时 Hugging Face API" in serialized


@respx.mock
def test_gateway_guides_multi_step_continuation_after_tool_response(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"WebSearch","arguments":{}}</tool_call>',
                },
                {
                    "role": "user",
                    "content": (
                        '<tool_response>{"name":"WebSearch",'
                        '"content":{"results":[]}}</tool_response>'
                    ),
                },
            ],
        },
    )
    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["messages"][-1]["role"] == "user"
    assert "call the next available tool now" in forwarded["messages"][-1]["content"]
    assert "现在必须向用户给出最终回答" not in forwarded["messages"][-1]["content"]
    assert forwarded["chat_template_kwargs"]["enable_thinking"] is False


@respx.mock
def test_gateway_reexposes_failed_bash_tool_for_corrected_retry(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )
    bash = {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "run",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
    web = {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
    system = "<tools> " + json.dumps(bash) + "\n" + json.dumps(web) + " </tools>"
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"Bash","arguments":{}}</tool_call>',
                },
                {
                    "role": "user",
                    "content": (
                        '<tool_response>{"name":"Bash","content":{"exitCode":127}}</tool_response>'
                    ),
                },
            ],
        },
    )
    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert [t["function"]["name"] for t in forwarded["tools"]] == ["Bash", "WebSearch"]


@respx.mock
def test_gateway_does_not_duplicate_reasoning_after_failed_tool(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-reasoning",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen-upstream",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "curl succeeded.",
                            "reasoning": "python failed. use curl directly.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            },
        )
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "stream": True,
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"Bash","arguments":{}}</tool_call>',
                },
                {
                    "role": "user",
                    "content": (
                        '<tool_response>{"name":"Bash","content":{"exitCode":127}}</tool_response>'
                    ),
                },
            ],
        },
    )
    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["stream"] is False
    assert forwarded["chat_template_kwargs"]["enable_thinking"] is False
    assert b'"reasoning_text":"python failed. use curl directly."' in response.content
    assert b'"content":"python failed. use curl directly."' not in response.content
    assert b'"content":"curl succeeded."' in response.content
    assert b'"finish_reason":"stop"' in response.content
    assert b"[DONE]" in response.content


@respx.mock
def test_gateway_retries_empty_alma_tool_continuation_before_returning(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    responses = iter(
        [
            Response(
                200,
                json={
                    "id": "chatcmpl-empty",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "qwen-upstream",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "total_tokens": 101,
                    },
                },
            ),
            Response(
                200,
                json={
                    "id": "chatcmpl-retry",
                    "object": "chat.completion",
                    "created": 2,
                    "model": "qwen-upstream",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "complete answer",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 10,
                        "total_tokens": 130,
                    },
                },
            ),
        ]
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        side_effect=lambda _request: next(responses)
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "user",
                    "content": '<tool_response>{"name":"WebFetch","content":"ok"}</tool_response>',
                },
            ],
        },
    )

    assert response.status_code == 200
    assert route.call_count == 2
    first = json.loads(route.calls[0].request.content)
    retry = json.loads(route.calls[1].request.content)
    assert first["stream"] is False
    assert "stream_options" not in first
    assert retry["stream"] is False
    assert "preceding assistant generation was empty" in retry["messages"][-1]["content"]
    assert b'"content":"complete answer"' in response.content
    assert b"[DONE]" in response.content

    with client.app.state.database.session_factory() as db:
        metric = db.scalars(
            select(RequestMetric).where(RequestMetric.model == "qwen-upstream")
        ).one()
        assert metric.prompt_tokens == 220
        assert metric.completion_tokens == 11


@respx.mock
def test_gateway_never_returns_an_empty_alma_tool_continuation(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-empty",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen-upstream",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "stream": True,
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "result",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert route.call_count == 3
    first = json.loads(route.calls[0].request.content)
    final_retry = json.loads(route.calls[-1].request.content)
    assert first["stream"] is False
    assert first["chat_template_kwargs"]["enable_thinking"] is False
    assert "call the next available tool now" in first["messages"][-1]["content"]
    assert final_retry["tool_choice"] == "none"
    assert "Do not call another tool" in final_retry["messages"][-1]["content"]
    assert "模型在工具执行后连续返回了空结果" in response.text
    assert b"[DONE]" in response.content


@respx.mock
def test_gateway_maps_vllm_reasoning_field_for_alma_stream(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"reasoning":"think token"},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"final answer"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert b'"reasoning_text":"think token"' in response.content
    assert b'"reasoning":"think token"' in response.content
    assert b'"content":"final answer"' in response.content
    assert b"[DONE]" in response.content


@respx.mock
def test_gateway_adds_reasoning_content_alias_for_alma_stream(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"reasoning_content":"step one"},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"final"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert b'"reasoning_text":"step one"' in response.content
    assert b'"reasoning_content":"step one"' in response.content
    assert b'"content":"final"' in response.content
    assert b"[DONE]" in response.content


@respx.mock
def test_gateway_normalizes_alma_xhigh_to_qwen_supported_xhigh(client):
    key = _create_gateway_key(client)
    _seed_deployment(
        client,
        config={
            "spec": {"generation_defaults": {}},
            "runtime_capabilities": {"generation_defaults": []},
        },
    )
    route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [], "usage": {}})
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [],
            "reasoning_effort": "xhigh",
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        },
    )

    assert response.status_code == 200
    forwarded = json.loads(route.calls[0].request.content)
    assert forwarded["reasoning_effort"] == "xhigh"
    assert forwarded["chat_template_kwargs"]["reasoning_effort"] == "xhigh"


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
            db.scalars(select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply"))
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
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply"))
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
def test_gateway_promotes_qwen_xml_tool_tags_from_reasoning_to_tool_calls(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"reasoning":"planning <tool_call>"},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"reasoning":"{\\"name\\":\\"Bash\\",'
                b'\\"arguments\\":{\\"command\\":\\"alma skill run web-search\\",'
                b'\\"description\\":\\"search hf\\",\\"timeout\\":120,'
                b'\\"run_in_background\\":false}}"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"reasoning":"</tool_call>"},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":9,"total_tokens":12}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert b'"tool_calls"' in response.content
    assert b'"name":"Bash"' in response.content
    assert b'"finish_reason":"tool_calls"' in response.content
    assert b'"reasoning_text":"planning <tool_call>"' in response.content
    assert b"[DONE]" in response.content


@respx.mock
def test_gateway_marks_tool_step_for_alma_without_changing_finish_reason(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"function":{"name":"search","arguments":"{\\"q\\":\\"x\\"}"}}]},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert b'"finish_reason":"tool_calls"' in response.content
    assert b'"alma_tool_step":true' in response.content
    assert b'"tool_calls"' in response.content
    assert b"[DONE]" in response.content


@respx.mock
def test_gateway_corrects_stop_finish_reason_for_streamed_tool_call(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"function":{"name":"ReadValue","arguments":"{}"}}]},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert b'"finish_reason":"tool_calls"' in response.content
    assert b'"alma_tool_step":true' in response.content


@respx.mock
def test_gateway_corrects_stop_finish_reason_for_nonstreamed_tool_call(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "ReadValue", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": []},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert response.json()["choices"][0]["alma_tool_step"] is True


@respx.mock
def test_streaming_gateway_records_usage_from_terminal_chunk(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "qwen-upstream",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert b"[DONE]" in response.content
    with client.app.state.database.session_factory() as db:
        metric = db.scalar(select(RequestMetric).order_by(RequestMetric.created_at.desc()))
    assert metric.prompt_tokens == 3
    assert metric.completion_tokens == 1
    assert metric.status_code == 200


@respx.mock
def test_streaming_gateway_records_completion_when_client_disconnects(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    ) as response:
        assert response.status_code == 200
        next(response.iter_bytes())

    with client.app.state.database.session_factory() as db:
        metric = db.scalar(select(RequestMetric).order_by(RequestMetric.created_at.desc()))
    assert metric.status_code == 499


@respx.mock
def test_streaming_gateway_preserves_upstream_error_status_in_metrics(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            400,
            content=b'{"error":{"message":"bad reasoning effort"}}',
            headers={"content-type": "application/json"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    assert response.status_code == 400
    with client.app.state.database.session_factory() as db:
        metric = db.scalar(select(RequestMetric).order_by(RequestMetric.created_at.desc()))
    assert metric.status_code == 400


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
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply"))
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
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "gateway.defaults.apply"))
    assert event is None


def _task_arguments(description: str) -> dict[str, str]:
    return {
        "description": description,
        "prompt": "Find the requested software on Hugging Face.",
        "agent_id": "developer",
    }


def _stream_payload(events: list[dict], *, done: bool = True) -> bytes:
    payload = "".join(f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events)
    if done:
        payload += "data: [DONE]\n\n"
    return payload.encode()


def _streamed_calls(response) -> dict[int, dict]:
    calls: dict[int, dict] = {}
    for line in response.text.splitlines():
        if not line.startswith("data: {"):
            continue
        event = json.loads(line.removeprefix("data: "))
        for choice in event.get("choices") or []:
            for tool_call in (choice.get("delta") or {}).get("tool_calls") or []:
                current = calls.setdefault(
                    tool_call.get("index", 0),
                    {"name": "", "arguments": []},
                )
                function = tool_call.get("function") or {}
                name = function.get("name")
                if isinstance(name, str) and not current["name"].endswith(name):
                    current["name"] += name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    current["arguments"].append(arguments)
    return calls


@respx.mock
def test_gateway_clamps_streaming_task_description_to_alma_schema_limit(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    overlong = "x" * 90
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_task_1",
                            "type": "function",
                            "function": {
                                "name": "Task",
                                "arguments": json.dumps(_task_arguments(overlong)),
                            },
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=_stream_payload([event]),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    calls = _streamed_calls(response)
    arguments = json.loads("".join(calls[0]["arguments"]))
    assert arguments == _task_arguments(overlong[:80])
    assert response.text.rstrip().endswith("data: [DONE]")


@respx.mock
def test_gateway_clamps_nonstreaming_task_description_to_alma_schema_limit(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    overlong = "y" * 90
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Task",
                                        "arguments": json.dumps(_task_arguments(overlong)),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            },
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": []},
    )

    function = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"]) == _task_arguments(overlong[:80])


@respx.mock
def test_gateway_clamps_fragmented_streaming_task_arguments(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    overlong = "z" * 90
    arguments = json.dumps(_task_arguments(overlong), separators=(",", ":"))
    fragments = [arguments[:25], arguments[25:70], arguments[70:]]
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_task_1",
                                "type": "function",
                                "function": {"name": "Ta", "arguments": fragments[0]},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "sk", "arguments": fragments[1]},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": fragments[2]}}]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=_stream_payload(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    calls = _streamed_calls(response)
    assert calls[0]["name"] == "Task"
    normalized = json.loads("".join(calls[0]["arguments"]))
    assert normalized == _task_arguments(overlong[:80])
    assert response.text.count("\n\ndata: ") == len(events)


@respx.mock
def test_gateway_preserves_fragmented_non_task_tool_calls(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_search_1",
                                "type": "function",
                                "function": {
                                    "name": "WebSearch",
                                    "arguments": '{"query":"Qwen',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '3.8"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=_stream_payload(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    calls = _streamed_calls(response)
    assert calls[0]["name"] == "WebSearch"
    assert json.loads("".join(calls[0]["arguments"])) == {"query": "Qwen3.8"}
    assert response.text.rstrip().endswith("data: [DONE]")


@respx.mock
def test_gateway_flushes_task_arguments_when_stream_has_no_finish_event(client):
    key = _create_gateway_key(client)
    _seed_deployment(client)
    overlong = "q" * 90
    arguments = json.dumps(_task_arguments(overlong), separators=(",", ":"))
    midpoint = len(arguments) // 2
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_task_tail",
                                "type": "function",
                                "function": {
                                    "name": "Task",
                                    "arguments": arguments[:midpoint],
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": arguments[midpoint:]},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
    ]
    respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=_stream_payload(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "qwen-upstream", "messages": [], "stream": True},
    )

    calls = _streamed_calls(response)
    normalized = json.loads("".join(calls[0]["arguments"]))
    assert normalized == _task_arguments(overlong[:80])
    assert response.text.rstrip().endswith("data: [DONE]")
