import json
from datetime import UTC, datetime
from pathlib import Path

import docker
import pytest
from app.config import Settings
from app.db import Database
from app.models import Deployment, TaskRecord
from app.runtime.base import (
    DeploymentSpec,
    GenerationDefaults,
    RecommendationProvenance,
    ResolvedDeploymentSpec,
    ResourceSnapshot,
    SpeculativeConfig,
    deterministic_container_name,
    validate_model_path,
)
from app.runtime.sglang import SGLangAdapter
from app.runtime.vllm import VllmAdapter
from app.services import deployments as deployment_service


def valid_spec_payload(tmp_path):
    return {
        "name": "Qwen Local",
        "model_id": "org/qwen-model",
        "model_path": str(tmp_path / "models" / "qwen"),
        "api_model_name": "qwen-local",
        "runtime": "vllm",
        "image": "vllm:test",
        "port": 8100,
        "quantization": "modelopt_fp4",
        "generation_defaults": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": -0.1,
            "max_tokens": 2048,
            "stop": ["</s>", "<|end|>"],
        },
        "speculative": {
            "draft_model_id": "org/qwen-draft",
            "method": "eagle3",
            "num_speculative_tokens": 8,
            "num_steps": 2,
            "eagle_top_k": 4,
            "num_draft_tokens": 16,
            "manual_review_acknowledged": True,
        },
        "recommendation": {
            "generated_at": datetime(2026, 8, 16, tzinfo=UTC),
            "evidence_hash": "a" * 64,
            "provider_id": "provider-1",
            "resource_snapshot": {
                "total_bytes": 1_000,
                "available_bytes": 800,
                "reserved_bytes": 200,
            },
            "modified_fields": ["generation_defaults.temperature", "quantization"],
            "sources": {
                "generation_defaults.temperature": "model_card",
                "quantization": "local_config",
                "context_length": "runtime_default",
                "memory_fraction": "device_rule",
                "generation_defaults.max_tokens": "ai",
            },
        },
    }


def valid_recommendation_payload():
    return {
        "generated_at": "2026-08-16T00:00:00Z",
        "evidence_hash": "a" * 64,
        "provider_id": "provider-1",
        "resource_snapshot": {
            "total_bytes": 1_000,
            "available_bytes": 800,
            "reserved_bytes": 200,
        },
        "modified_fields": ["generation_defaults.temperature"],
        "sources": {"generation_defaults.temperature": "model_card"},
    }


def test_deployment_spec_serializes_recommendation_settings(tmp_path):
    spec = DeploymentSpec.model_validate(valid_spec_payload(tmp_path))

    dumped = spec.model_dump(mode="json")

    assert dumped["quantization"] == "modelopt_fp4"
    assert dumped["generation_defaults"]["stop"] == ["</s>", "<|end|>"]
    assert dumped["speculative"]["method"] == "eagle3"
    assert dumped["speculative"]["num_draft_tokens"] == 16
    assert dumped["recommendation"]["generated_at"] == "2026-08-16T00:00:00Z"
    assert dumped["recommendation"]["sources"]["memory_fraction"] == "device_rule"


@pytest.mark.parametrize(
    ("field", "lower", "upper", "below", "above"),
    [
        ("temperature", 0, 2, -0.01, 2.01),
        ("top_p", 0.01, 1, 0, 1.01),
        ("top_k", 0, 1_000_000, -1, 1_000_001),
        ("min_p", 0, 1, -0.01, 1.01),
        ("repetition_penalty", 0.01, 2, 0, 2.01),
        ("presence_penalty", -2, 2, -2.01, 2.01),
        ("frequency_penalty", -2, 2, -2.01, 2.01),
        ("max_tokens", 1, 1_048_576, 0, 1_048_577),
    ],
)
def test_generation_defaults_enforces_numeric_boundaries(
    field, lower, upper, below, above
):
    assert getattr(GenerationDefaults.model_validate({field: lower}), field) == lower
    assert getattr(GenerationDefaults.model_validate({field: upper}), field) == upper

    for invalid in (below, above):
        with pytest.raises(ValueError):
            GenerationDefaults.model_validate({field: invalid})


@pytest.mark.parametrize(
    "stop",
    [[], "x", "x" * 500, ["stop"] * 16],
)
def test_generation_defaults_accepts_valid_stop_boundaries(stop):
    assert GenerationDefaults(stop=stop).stop == stop


@pytest.mark.parametrize(
    "stop",
    ["", "x" * 501, [""], ["stop"] * 17],
)
def test_generation_defaults_rejects_invalid_stop_boundaries(stop):
    with pytest.raises(ValueError, match="stop must contain"):
        GenerationDefaults(stop=stop)


@pytest.mark.parametrize(
    ("field", "lower", "upper", "below", "above"),
    [
        ("num_speculative_tokens", 1, 64, 0, 65),
        ("num_steps", 1, 32, 0, 33),
        ("eagle_top_k", 1, 32, 0, 33),
        ("num_draft_tokens", 1, 256, 0, 257),
    ],
)
def test_speculative_config_enforces_numeric_boundaries(
    field, lower, upper, below, above
):
    def payload(value):
        settings = {
            "draft_model_id": "draft-id",
            "method": "eagle",
            field: value,
        }
        if field in {"num_steps", "eagle_top_k", "num_draft_tokens"}:
            settings.update(
                {
                    "num_steps": 1,
                    "eagle_top_k": 1,
                    "num_draft_tokens": 1,
                    field: value,
                }
            )
        return settings

    assert getattr(SpeculativeConfig.model_validate(payload(lower)), field) == lower
    assert getattr(SpeculativeConfig.model_validate(payload(upper)), field) == upper

    for invalid in (below, above):
        with pytest.raises(ValueError):
            SpeculativeConfig.model_validate(payload(invalid))


@pytest.mark.parametrize("draft_model_id", ["a", "a" * 64])
def test_speculative_config_accepts_draft_model_id_boundaries(draft_model_id):
    spec = SpeculativeConfig(draft_model_id=draft_model_id, method="draft_model")

    assert spec.draft_model_id == draft_model_id


@pytest.mark.parametrize("draft_model_id", ["", "a" * 65])
def test_speculative_config_rejects_invalid_draft_model_id_lengths(draft_model_id):
    with pytest.raises(ValueError):
        SpeculativeConfig(draft_model_id=draft_model_id, method="draft_model")


@pytest.mark.parametrize("method", ["draft_model", "eagle", "eagle3", "mtp"])
def test_speculative_config_accepts_supported_methods(method):
    assert SpeculativeConfig(draft_model_id="draft-id", method=method).method == method


def test_speculative_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        SpeculativeConfig(draft_model_id="draft-id", method="unknown")


@pytest.mark.parametrize(
    "tuning",
    [
        {"num_steps": 2},
        {"eagle_top_k": 4},
        {"num_draft_tokens": 16},
        {"num_steps": 2, "eagle_top_k": 4},
        {"num_steps": 2, "num_draft_tokens": 16},
        {"eagle_top_k": 4, "num_draft_tokens": 16},
    ],
)
def test_speculative_eagle_tuning_fields_must_be_set_together(tmp_path, tuning):
    payload = valid_spec_payload(tmp_path)
    payload["quantization"] = "fp8"
    payload["speculative"] = {
        "draft_model_id": "org/qwen-draft",
        "method": "eagle",
        **tuning,
    }

    with pytest.raises(
        ValueError,
        match="set num_steps, eagle_top_k and num_draft_tokens together",
    ):
        DeploymentSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolved_draft_model_path", "/models/qwen-draft"),
        ("speculative_runtime_method", "eagle"),
    ],
)
def test_public_deployment_spec_rejects_internal_resolution_fields(
    tmp_path, field, value
):
    payload = valid_spec_payload(tmp_path)
    payload["quantization"] = "fp8"
    payload[field] = value

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        DeploymentSpec.model_validate(payload)


def test_resolved_deployment_spec_public_dump_is_json_safe_and_excludes_internal_fields(
    tmp_path,
):
    resolved = ResolvedDeploymentSpec.model_validate(
        {
            **valid_spec_payload(tmp_path),
            "resolved_draft_model_path": str(tmp_path / "models" / "qwen-draft"),
            "draft_container_model_path": "/models/qwen-draft",
            "speculative_runtime_method": "eagle",
        }
    )

    public = resolved.public_dump()

    assert json.loads(json.dumps(public)) == public
    assert public["recommendation"]["generated_at"] == "2026-08-16T00:00:00Z"
    assert set(public) == set(DeploymentSpec.model_fields)
    assert "resolved_draft_model_path" not in public
    assert "draft_container_model_path" not in public
    assert "speculative_runtime_method" not in public


@pytest.mark.parametrize(
    "field",
    ["total_bytes", "available_bytes", "reserved_bytes"],
)
def test_resource_snapshot_enforces_nonnegative_boundaries(field):
    payload = {"total_bytes": 0, "available_bytes": 0, "reserved_bytes": 0}
    assert getattr(ResourceSnapshot.model_validate(payload), field) == 0

    payload[field] = -1
    with pytest.raises(ValueError):
        ResourceSnapshot.model_validate(payload)


@pytest.mark.parametrize(
    "evidence_hash",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_recommendation_provenance_rejects_invalid_evidence_hash(evidence_hash):
    payload = valid_recommendation_payload()
    payload["evidence_hash"] = evidence_hash

    with pytest.raises(ValueError):
        RecommendationProvenance.model_validate(payload)


def test_recommendation_provenance_accepts_lowercase_sha256_hash():
    provenance = RecommendationProvenance.model_validate(valid_recommendation_payload())

    assert provenance.evidence_hash == "a" * 64


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (GenerationDefaults, {"temperature": 0.7}),
        (
            SpeculativeConfig,
            {"draft_model_id": "draft-id", "method": "draft_model"},
        ),
        (
            ResourceSnapshot,
            {"total_bytes": 1, "available_bytes": 1, "reserved_bytes": 0},
        ),
        (RecommendationProvenance, valid_recommendation_payload()),
    ],
)
def test_nested_deployment_models_reject_extra_fields(model_type, payload):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        model_type.model_validate({**payload, "unexpected": True})


def test_container_name_is_deterministic_and_safe():
    assert deterministic_container_name("Qwen 3.8 / 27B") == "dgx-qwen-3-8-27b"
    assert deterministic_container_name("Qwen 3.8 / 27B") == "dgx-qwen-3-8-27b"


def test_model_path_must_be_in_allowed_root(tmp_path):
    root = tmp_path / "models"
    model = root / "org" / "model"
    model.mkdir(parents=True)

    assert validate_model_path(model, (root,)) == model.resolve()
    with pytest.raises(ValueError):
        validate_model_path(Path(tmp_path, "outside"), (root,))


def test_runtime_adapters_generate_only_validated_arguments(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    spec = DeploymentSpec(
        name="Qwen Local",
        model_path=str(model_path),
        api_model_name="qwen-local",
        runtime="vllm",
        image="vllm/vllm-openai:v0.27.1",
        port=8100,
        context_length=32768,
        memory_fraction=0.5,
        max_concurrency=8,
    )

    vllm = VllmAdapter(
        allowed_images={"vllm/vllm-openai:v0.27.1"}, model_roots=(tmp_path / "models",)
    )
    command = vllm.command(spec)
    assert command[:4] == ["--model", "/models/qwen", "--served-model-name", "qwen-local"]
    assert "--gpu-memory-utilization" in command

    sglang_spec = spec.model_copy(update={"runtime": "sglang", "image": "sglang:test"})
    sglang = SGLangAdapter(
        allowed_images={"sglang:test"}, model_roots=(tmp_path / "models",)
    )
    assert sglang.command(sglang_spec)[:3] == ["python3", "-m", "sglang.launch_server"]


def test_deployment_preview_preserves_shared_gateway_route(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(
        allowed_images={"vllm/vllm-openai:v0.27.1"}, model_roots=(tmp_path / "models",)
    )
    spec = DeploymentSpec(
        name="Qwen replica A",
        model_path=str(model_path),
        api_model_name="qwen-replica-a",
        route_alias="qwen-shared",
        runtime="vllm",
        image="vllm/vllm-openai:v0.27.1",
        port=8100,
        recommendation=valid_recommendation_payload(),
    )
    (model_path / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")

    preview = adapter.preview(spec)
    assert preview["route_alias"] == "qwen-shared"
    assert preview["compatibility"]["compatible"] is True
    assert preview["compatibility"]["architectures"] == ["Qwen2ForCausalLM"]
    assert preview["estimated_disk_bytes"] > 0
    assert preview["estimated_memory_bytes"] >= preview["estimated_disk_bytes"]
    assert preview["spec"] == spec.model_dump(mode="json")
    json.dumps(preview)
    assert preview["operations"][-1] == "Probe /v1/models and register the gateway route"
    assert "client.chat.completions.create" in preview["api_example"]

    with pytest.raises(ValueError):
        DeploymentSpec(
            name="Invalid route",
            model_path=str(model_path),
            api_model_name="qwen-replica-b",
            route_alias="invalid route with spaces",
            runtime="vllm",
            image="vllm/vllm-openai:v0.27.1",
            port=8101,
        )


def test_runtime_adapter_exposes_lifecycle_contract(tmp_path):
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path,))

    for method in (
        "start",
        "stop",
        "restart",
        "health_check",
        "logs",
        "metrics",
        "uninstall",
    ):
        assert callable(getattr(adapter, method, None)), method


def test_vllm_command_includes_batch_and_quantization_settings(tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path / "models",))
    spec = DeploymentSpec(
        name="Qwen",
        model_path=str(model_path),
        api_model_name="qwen",
        runtime="vllm",
        image="vllm:test",
        port=8100,
        max_batched_tokens=4096,
        quantization="fp8",
    )

    command = adapter.command(spec)

    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
    assert command[command.index("--quantization") + 1] == "fp8"


def test_create_endpoint_persists_json_safe_recommendation(authenticated_client, tmp_path):
    payload = valid_spec_payload(tmp_path)
    Path(payload["model_path"]).mkdir(parents=True)
    payload["model_id"] = None
    payload["image"] = "vllm/vllm-openai:v0.27.1"
    payload["recommendation"] = valid_recommendation_payload()

    response = authenticated_client.post("/api/deployments", json=payload)

    assert response.status_code == 202
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        assert task.input_json["recommendation"]["generated_at"] == (
            "2026-08-16T00:00:00Z"
        )
        json.dumps(task.input_json)


def test_update_endpoint_queues_managed_deployment_change(authenticated_client, tmp_path):
    model_path = tmp_path / "models" / "qwen"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"architectures":["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")
    with authenticated_client.app.state.database.session_factory() as db:
        deployment = Deployment(
            name="managed",
            runtime="vllm",
            container_id="container-id",
            container_name="dgx-managed",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="running",
            health="healthy",
            managed=True,
            image="vllm/vllm-openai:v0.27.1",
            port=8100,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id

    response = authenticated_client.patch(
        f"/api/deployments/{deployment_id}",
        json={
            "name": "managed",
            "model_path": str(model_path),
            "api_model_name": "managed",
            "runtime": "vllm",
            "image": "vllm/vllm-openai:v0.27.1",
            "port": 8100,
            "recommendation": valid_recommendation_payload(),
        },
    )

    assert response.status_code == 202
    assert response.json()["type"] == "deployment.update"
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        assert task.input_json["spec"]["recommendation"]["generated_at"] == (
            "2026-08-16T00:00:00Z"
        )
        json.dumps(task.input_json)


def test_container_model_path_maps_to_the_configured_host_root(tmp_path):
    container_root = tmp_path / "manager" / "hf-cache" / "hub"
    model_path = container_root / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "abc"
    model_path.mkdir(parents=True)
    host_root = Path("/home/operator/.cache/huggingface/hub")

    resolver = getattr(deployment_service, "resolve_host_model_mount", lambda *_: None)

    assert resolver(model_path, (container_root,), (host_root,)) == host_root


def test_settings_preserve_container_to_host_model_root_order(tmp_path):
    container_models = tmp_path / "container-models"
    container_hf = tmp_path / "container-hf"
    host_models = Path("/srv/models")
    host_hf = Path("/home/operator/.cache/huggingface/hub")
    settings = Settings(
        secret_key="test-secret-key-with-at-least-32-characters",
        admin_password="Test-password-1234",
        model_roots=f"{container_models},{container_hf}",
        model_root_mappings=(
            f"{container_models}={host_models};{container_hf}={host_hf}"
        ),
    )

    assert getattr(settings, "host_model_root_paths", ()) == (host_models, host_hf)
    assert getattr(settings, "deployment_startup_timeout_seconds", None) == 300


def test_deployment_service_mounts_the_host_model_root(tmp_path, monkeypatch):
    container_root = tmp_path / "manager-hf"
    model_path = container_root / "models--org--model" / "snapshots" / "abc"
    model_path.mkdir(parents=True)
    host_root = Path("/home/operator/.cache/huggingface/hub")
    database = Database(f"sqlite:///{tmp_path / 'manager.db'}")
    database.create_schema()
    adapter = VllmAdapter(
        allowed_images={"vllm/vllm-openai:v0.27.1"}, model_roots=(container_root,)
    )
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(container_root,),
        host_model_roots=(host_root,),
    )
    captured: dict = {}

    class FakeContainer:
        id = "container-id"
        name = "dgx-test-model"
        status = "running"
        attrs = {"Config": {"Labels": {"com.dgx-spark-manager.managed": "true"}}}

        def reload(self):
            return None

        def stop(self, **_kwargs):
            return None

        def remove(self):
            return None

    class FakeContainers:
        def get(self, _name):
            raise docker.errors.NotFound("missing")

        def run(self, _image, **kwargs):
            captured.update(kwargs)
            return FakeContainer()

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    client_type = type("Client", (), {"containers": FakeContainers()})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    monkeypatch.setattr(
        deployment_service.httpx,
        "get",
        lambda *_args, **_kwargs: type("Response", (), {"is_success": True})(),
    )

    service.create_handler(
        Context(),
        DeploymentSpec(
            name="Test Model",
            model_path=str(model_path),
            api_model_name="test-model",
            runtime="vllm",
            image="vllm/vllm-openai:v0.27.1",
            port=8100,
        ).model_dump(),
    )

    assert captured["volumes"] == {str(host_root): {"bind": "/models", "mode": "ro"}}
    assert captured["log_config"] == {
        "Type": "json-file",
        "Config": {"max-size": "10m", "max-file": "5"},
    }


def test_deployment_timeout_captures_logs_and_rolls_back_new_container(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_path = model_root / "model"
    model_path.mkdir(parents=True)
    database = Database(f"sqlite:///{tmp_path / 'timeout.db'}")
    database.create_schema()
    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(model_root,))
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(model_root,),
        startup_timeout_seconds=2,
    )

    class FakeContainer:
        id = "new-container"
        name = "dgx-timeout"
        status = "running"
        stopped = False
        removed = False

        def logs(self, **_kwargs):
            return b"engine failed while allocating memory"

        def stop(self, **_kwargs):
            self.stopped = True

        def remove(self):
            self.removed = True

    container = FakeContainer()

    class FakeContainers:
        def get(self, _name):
            raise docker.errors.NotFound("missing")

        def run(self, _image, **_kwargs):
            return container

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    client_type = type("Client", (), {"containers": FakeContainers()})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    response_type = type("Response", (), {"is_success": False})
    monkeypatch.setattr(
        deployment_service.httpx,
        "get",
        lambda *_args, **_kwargs: response_type(),
    )
    monkeypatch.setattr(deployment_service.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="engine failed while allocating memory"):
        service.create_handler(
            Context(),
            DeploymentSpec(
                name="Timeout",
                model_path=str(model_path),
                api_model_name="timeout",
                runtime="vllm",
                image="vllm:test",
                port=8100,
            ).model_dump(),
        )

    assert container.stopped is True
    assert container.removed is True


def test_start_action_waits_for_real_runtime_health(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'action.db'}")
    database.create_schema()
    with database.session_factory() as db:
        deployment = Deployment(
            name="managed",
            runtime="vllm",
            container_id="container-id",
            container_name="dgx-managed",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="exited",
            health="unknown",
            managed=True,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id
    service = deployment_service.DeploymentService(
        adapters={
            "vllm": VllmAdapter(allowed_images={"vllm:test"}, model_roots=(tmp_path,))
        },
        session_factory=database.session_factory,
        model_roots=(tmp_path,),
        startup_timeout_seconds=4,
    )

    class FakeContainer:
        name = "dgx-managed"
        starts = 0

        def start(self):
            self.starts += 1

    container = FakeContainer()
    containers = type("Containers", (), {"get": lambda _self, _id: container})()
    client_type = type("Client", (), {"containers": containers})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    health_results = iter([False, True])
    response_type = type(
        "Response",
        (),
        {"is_success": property(lambda _self: next(health_results))},
    )
    monkeypatch.setattr(deployment_service.httpx, "get", lambda *_args, **_kwargs: response_type())
    monkeypatch.setattr(deployment_service.time, "sleep", lambda _seconds: None)

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    result = service.action_handler(
        Context(),
        {"deployment_id": deployment_id, "action": "start"},
    )

    assert container.starts == 1
    assert result["health"] == "healthy"
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.status == "running"
        assert updated.health == "healthy"


def test_update_handler_replaces_container_and_keeps_deployment_id(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_path = model_root / "qwen"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"architectures":["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")
    database = Database(f"sqlite:///{tmp_path / 'update.db'}")
    database.create_schema()
    with database.session_factory() as db:
        deployment = Deployment(
            name="managed",
            runtime="vllm",
            container_id="old-container",
            container_name="dgx-managed",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="running",
            health="healthy",
            managed=True,
            image="vllm:test",
            port=8100,
        )
        db.add(deployment)
        db.commit()
        deployment_id = deployment.id

    adapter = VllmAdapter(allowed_images={"vllm:test"}, model_roots=(model_root,))
    service = deployment_service.DeploymentService(
        adapters={"vllm": adapter},
        session_factory=database.session_factory,
        model_roots=(model_root,),
    )

    class FakeContainer:
        def __init__(self, container_id, name):
            self.id = container_id
            self.name = name
            self.status = "running"
            self.removed = False

        def reload(self):
            return None

        def stop(self, **_kwargs):
            self.status = "exited"

        def start(self):
            self.status = "running"

        def rename(self, name):
            self.name = name

        def remove(self, **_kwargs):
            self.removed = True

        def logs(self, **_kwargs):
            return b"ready"

    old = FakeContainer("old-container", "dgx-managed")
    new = FakeContainer("new-container", "dgx-managed")

    class FakeContainers:
        def get(self, identifier):
            assert identifier == "old-container"
            return old

        def run(self, _image, **_kwargs):
            return new

    client_type = type("Client", (), {"containers": FakeContainers()})
    monkeypatch.setattr(service, "docker_client", lambda: client_type())
    monkeypatch.setattr(service, "wait_for_health", lambda *_args, **_kwargs: True)

    class Context:
        def update(self, **_kwargs):
            return None

        def check_control(self):
            return None

    result = service.update_handler(
        Context(),
        {
            "deployment_id": deployment_id,
            "spec": DeploymentSpec(
                name="managed",
                model_path=str(model_path),
                api_model_name="managed",
                route_alias="managed-route",
                runtime="vllm",
                image="vllm:test",
                port=8100,
                context_length=8192,
            ).model_dump(),
        },
    )

    assert result["deployment_id"] == deployment_id
    assert old.removed is True
    with database.session_factory() as db:
        updated = db.get(Deployment, deployment_id)
        assert updated.container_id == "new-container"
        assert updated.config["spec"]["context_length"] == 8192
        assert updated.config["route_alias"] == "managed-route"

