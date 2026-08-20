from pathlib import Path
from types import SimpleNamespace

import pytest
from app.config import Settings
from app.runtime.base import DeploymentSpec
from app.runtime.llama_cpp import LlamaCppAdapter
from app.services.runtime_capabilities import RuntimeCapabilityService

IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu24.04"


def _adapter(tmp_path: Path) -> tuple[LlamaCppAdapter, Path, Path]:
    model_root = tmp_path / "models"
    model_path = model_root / "qwen"
    runtime_dir = tmp_path / "llamacpp"
    (runtime_dir / "lib").mkdir(parents=True)
    binary = runtime_dir / "llama-server"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    model_path.mkdir(parents=True)
    adapter = LlamaCppAdapter(
        allowed_images={IMAGE},
        model_roots=(model_root,),
        host_runtime_dir="/opt/llamacpp",
        manager_runtime_dir=runtime_dir,
    )
    return adapter, model_path, runtime_dir


def _spec(model_path: Path, **updates) -> DeploymentSpec:
    values = {
        "name": "Qwen GGUF",
        "model_id": "model-1",
        "model_path": str(model_path),
        "api_model_name": "qwen-gguf",
        "runtime": "llama_cpp",
        "image": IMAGE,
        "port": 8014,
        "context_length": 262144,
        "memory_fraction": 0.8,
        "max_concurrency": 1,
        "quantization": "gguf",
        "generation_defaults": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0,
        },
        "llama_cpp": {
            "model_file": "model-Q8_0.gguf",
            "mmproj_file": "mmproj-F16.gguf",
            "gpu_layers": "all",
            "jinja": True,
            "continuous_batching": True,
            "mtp_enabled": True,
            "mtp_tokens": 3,
        },
    }
    values.update(updates)
    return DeploymentSpec.model_validate(values)


def test_llama_cpp_command_preserves_gguf_mmproj_and_mtp_settings(tmp_path):
    adapter, model_path, _ = _adapter(tmp_path)
    (model_path / "model-Q8_0.gguf").write_bytes(b"model")
    (model_path / "mmproj-F16.gguf").write_bytes(b"projector")

    command = adapter.command(_spec(model_path))

    assert command[:3] == [
        "/opt/llamacpp/llama-server",
        "--model",
        "/models/qwen/model-Q8_0.gguf",
    ]
    assert command[command.index("--alias") + 1] == "qwen-gguf"
    assert command[command.index("--mmproj") + 1] == "/models/qwen/mmproj-F16.gguf"
    assert command[command.index("--ctx-size") + 1] == "262144"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--spec-draft-model") + 1] == command[2]
    assert command[command.index("--spec-draft-n-max") + 1] == "3"
    assert "--jinja" in command
    assert "--cont-batching" in command


def test_llama_cpp_passes_chat_template_kwargs_only_with_jinja(tmp_path):
    adapter, model_path, _ = _adapter(tmp_path)
    (model_path / "model-Q8_0.gguf").write_bytes(b"model")
    (model_path / "mmproj-F16.gguf").write_bytes(b"projector")
    kwargs = {"enable_thinking": False, "reasoning_effort": "low"}

    command = adapter.command(_spec(model_path, chat_template_kwargs=kwargs))
    index = command.index("--chat-template-kwargs")
    assert command[index + 1] == '{"enable_thinking":false,"reasoning_effort":"low"}'

    no_jinja = _spec(
        model_path,
        chat_template_kwargs=kwargs,
        llama_cpp={"model_file": "model-Q8_0.gguf", "jinja": False},
    )
    assert "--chat-template-kwargs" not in adapter.command(no_jinja)


def test_llama_cpp_auto_selects_single_model_and_f16_projector(tmp_path):
    adapter, model_path, _ = _adapter(tmp_path)
    (model_path / "model.gguf").write_bytes(b"model")
    (model_path / "mmproj-F16.gguf").write_bytes(b"projector")
    spec = _spec(model_path, llama_cpp={})

    command = adapter.command(spec)

    assert command[command.index("--model") + 1] == "/models/qwen/model.gguf"
    assert command[command.index("--mmproj") + 1] == "/models/qwen/mmproj-F16.gguf"
    assert "--spec-type" not in command


@pytest.mark.parametrize(
    "llama_config,error",
    [
        ({"model_file": "../outside.gguf"}, "GGUF basenames"),
        ({"model_file": "missing.gguf"}, "GGUF file is unavailable"),
        ({"model_file": "mmproj-F16.gguf"}, "cannot be an mmproj"),
    ],
)
def test_llama_cpp_rejects_unsafe_or_invalid_model_selection(
    tmp_path, llama_config, error
):
    adapter, model_path, _ = _adapter(tmp_path)
    (model_path / "model.gguf").write_bytes(b"model")
    (model_path / "mmproj-F16.gguf").write_bytes(b"projector")

    with pytest.raises(ValueError, match=error):
        adapter.command(_spec(model_path, llama_cpp=llama_config))


def test_llama_cpp_exposes_only_server_configured_runtime_mount(tmp_path):
    adapter, model_path, _ = _adapter(tmp_path)
    (model_path / "model-Q8_0.gguf").write_bytes(b"model")
    (model_path / "mmproj-F16.gguf").write_bytes(b"projector")
    spec = _spec(model_path)

    assert adapter.extra_volumes(spec) == {
        "/opt/llamacpp": {"bind": "/opt/llamacpp", "mode": "ro"}
    }
    assert adapter.environment(spec) == {"LD_LIBRARY_PATH": "/opt/llamacpp/lib"}


def test_llama_cpp_capabilities_use_manifest_without_probe(tmp_path):
    calls = []
    image = SimpleNamespace(id="sha256:llama")
    client = SimpleNamespace(images=SimpleNamespace(get=lambda _name: image))
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'manager.db'}",
        secret_key="s" * 32,
        admin_password="Test-password-1234",
    )
    service = RuntimeCapabilityService(
        settings=settings,
        docker_client=client,
        probe_runner=lambda runtime, name: calls.append((runtime, name)) or "",
    )

    capabilities = service.get("llama_cpp", IMAGE)

    assert calls == []
    assert capabilities.source == "manifest"
    assert capabilities.quantization_methods == ["auto", "gguf"]
    assert capabilities.speculative_methods == []
