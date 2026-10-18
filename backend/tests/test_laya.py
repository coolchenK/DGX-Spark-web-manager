import importlib.util
from pathlib import Path
from types import SimpleNamespace

import respx
from app.models import Deployment
from app.runtime.base import DeploymentSpec
from app.runtime.laya import LayaAdapter
from app.services.discovery import container_candidate
from fastapi.testclient import TestClient


def test_laya_panel_adapter_and_authenticated_decision_route(authenticated_client, tmp_path):
    model_path = tmp_path / "laya"
    for sub in ("", "multilingual", "typed-decisions"):
        for name in (
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ):
            path = model_path / sub / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    adapter = LayaAdapter(allowed_images={"dgx-local/laya:0.3.4-cu130"}, model_roots=(tmp_path,))
    spec = DeploymentSpec(
        name="Laya Router",
        runtime="laya",
        model_path=str(model_path),
        api_model_name="laya",
        image="dgx-local/laya:0.3.4-cu130",
        context_length=1024,
        max_concurrency=1,
    )
    preview = adapter.preview(spec)
    assert preview["compatibility"]["compatible"]
    assert preview["capabilities"] == ["decision"]
    candidate = container_candidate(
        {
            "Config": {"Image": spec.image, "Cmd": adapter.command(spec)},
            "State": {"Status": "running"},
        }
    )
    assert candidate["runtime"] == "laya"
    with authenticated_client.app.state.database.session_factory() as db:
        deployment = Deployment(
            name=spec.name,
            runtime="laya",
            api_model_name="laya",
            status="running",
            health="healthy",
            capabilities=["decision"],
            endpoint_url="http://127.0.0.1:8123",
            config={"spec": spec.model_dump()},
        )
        db.add(deployment)
        db.commit()
        discovery = authenticated_client.app.state.discovery_service
        assert discovery._deployment_capabilities(db, deployment) == ["decision"]
    key = authenticated_client.post("/api/keys", json={"name": "Laya test"}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}
    body = {
        "model": "laya",
        "state": "refund",
        "questions": {"refund": {"type": "noul", "instructions": "Refund requested?"}},
    }
    assert authenticated_client.post("/v1/decisions", json=body).status_code == 401
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("http://127.0.0.1:8123/v1/decisions").respond(
            200, json={"answers": {"refund": {"noul": 0.9}}, "usage": {"output_tokens": 0}}
        )
        response = authenticated_client.post("/v1/decisions", json=body, headers=headers)
        assert response.status_code == 200 and response.json()["answers"]["refund"]["noul"] == 0.9
        import json

        assert json.loads(route.calls.last.request.content) == body
    response = authenticated_client.get("/v1/models/laya", headers=headers)
    assert response.status_code == 200
    assert response.json()["capabilities"]["decision"] is True
    assert response.json()["capabilities"]["streaming"] is False
    assert response.json()["api_format"] == "laya-decisions"
    assert response.json()["max_output_tokens"] == 0
    wrong_endpoint = authenticated_client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "laya", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert wrong_endpoint.status_code == 400
    assert "/v1/decisions" in wrong_endpoint.text
    assert (
        authenticated_client.post(
            "/v1/decisions", json={**body, "model": "missing"}, headers=headers
        ).status_code
        == 404
    )


def test_laya_service_validates_typed_questions_and_preserves_sdk_result():
    source = Path(__file__).resolve().parents[2] / "deploy/images/laya/server.py"
    spec = importlib.util.spec_from_file_location("laya_server_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = module.create_app("/unused", "laya")
    app.state.router = SimpleNamespace(
        loaded=["multilingual"],
        load=lambda name: SimpleNamespace(
            device="cuda:0", cfg={"max_len": 1024, "head_max_len": 256}
        ),
        predict=lambda state, questions, model: {
            "answers": {"refund": {"type": "noul", "noul": 0.9}},
            "routing": {"model": "multilingual"},
            "usage": {"output_tokens": 0},
        },
    )
    client = TestClient(app)  # No lifespan/GPU load; exercise request boundary and SDK forwarding.
    body = {
        "model": "laya",
        "state": "请退款",
        "questions": {"refund": {"type": "noul", "instructions": "Refund requested?"}},
    }
    result = client.post("/v1/decisions", json=body)
    assert result.status_code == 200 and result.json()["routing"]["model"] == "multilingual"
    for questions in (
        {},
        {"bad": {"type": "choice", "instructions": "Choose", "criteria": ["one"]}},
        {"bad": {"type": "score", "instructions": "Score", "criteria": {"a": "a", "b": "b"}}},
    ):
        assert (
            client.post("/v1/decisions", json={**body, "questions": questions}).status_code == 422
        )
    assert client.post("/v1/decisions", json={**body, "state": "x" * 65537}).status_code == 422
    assert client.post("/v1/decisions", json={**body, "model": "missing"}).status_code == 404
