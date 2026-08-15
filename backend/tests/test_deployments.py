from pathlib import Path

import docker
import pytest
from app.config import Settings
from app.db import Database
from app.runtime.base import DeploymentSpec, deterministic_container_name, validate_model_path
from app.runtime.sglang import SGLangAdapter
from app.runtime.vllm import VllmAdapter
from app.services import deployments as deployment_service


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
    )
    (model_path / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (model_path / "model.safetensors").write_bytes(b"weights")

    preview = adapter.preview(spec)
    assert preview["route_alias"] == "qwen-shared"
    assert preview["compatibility"]["compatible"] is True
    assert preview["compatibility"]["architectures"] == ["Qwen2ForCausalLM"]
    assert preview["estimated_disk_bytes"] > 0
    assert preview["estimated_memory_bytes"] >= preview["estimated_disk_bytes"]
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

