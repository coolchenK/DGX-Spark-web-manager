from pathlib import Path

import pytest
from app.runtime.base import DeploymentSpec, deterministic_container_name, validate_model_path
from app.runtime.sglang import SGLangAdapter
from app.runtime.vllm import VllmAdapter


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

