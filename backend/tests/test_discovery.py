import json
import os

from app.services import discovery
from app.services.discovery import (
    DiscoveryService,
    container_candidate,
    infer_runtime,
    parse_hf_cache_repository,
    resolve_hf_snapshot,
)


def test_parse_hugging_face_cache_repository():
    assert (
        parse_hf_cache_repository(
            "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
        )
        == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    )
    assert parse_hf_cache_repository(".locks") is None


def test_resolve_hf_snapshot_prefers_revision_reference(tmp_path):
    repository = tmp_path / "models--org--model"
    (repository / "refs").mkdir(parents=True)
    (repository / "snapshots" / "abc123").mkdir(parents=True)
    (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")

    assert resolve_hf_snapshot(repository) == repository / "snapshots" / "abc123"


def test_resolve_hf_snapshot_uses_newest_snapshot_without_main_ref(tmp_path):
    repository = tmp_path / "models--org--model"
    older = repository / "snapshots" / "older"
    newer = repository / "snapshots" / "newer"
    older.mkdir(parents=True)
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert resolve_hf_snapshot(repository) == newer


def test_scan_models_skips_inaccessible_roots(settings):
    class InaccessibleRoot:
        def exists(self):
            raise PermissionError("not readable")

    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    service = DiscoveryService((InaccessibleRoot(),))

    with database.session_factory() as db:
        assert service.scan_models(db) == []


def test_infer_runtime_from_image_and_command():
    assert (
        infer_runtime("sglang-inkling:specforge", ["python", "-m", "sglang.launch_server"])
        == "sglang"
    )
    assert infer_runtime("vllm/vllm-openai:v0.27.1", ["--model", "nvidia/model"]) == "vllm"
    assert infer_runtime("redis:7", ["redis-server"]) is None


def test_container_candidate_extracts_openai_endpoint_and_model():
    attrs = {
        "Id": "abc123",
        "Name": "/qwen38-dspark",
        "Config": {
            "Image": "sglang-inkling:specforge",
            "Cmd": [
                "python3",
                "-m",
                "sglang.launch_server",
                "--served-model-name",
                "qwen3.8-27b",
                "--port",
                "8000",
            ],
            "Labels": {},
        },
        "State": {"Status": "running", "Health": {"Status": "healthy"}},
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8001"}]}
        },
    }

    candidate = container_candidate(attrs)

    assert candidate is not None
    assert candidate["name"] == "qwen38-dspark"
    assert candidate["runtime"] == "sglang"
    assert candidate["endpoint_url"] == "http://127.0.0.1:8001"
    assert candidate["api_model_name"] == "qwen3.8-27b"
    assert candidate["managed"] is False


def test_model_metadata_is_derived_from_snapshot_and_config(tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "quantization_config": {"quant_method": "awq"},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    infer = getattr(discovery, "infer_model_metadata", lambda *_args, **_kwargs: {})

    assert infer(snapshot, "Qwen/Qwen2.5-0.5B-Instruct") == {
        "commit_hash": "abc123",
        "format": "safetensors",
        "quantization": "awq",
        "parameter_count": "0.5B",
        "capabilities": ["chat", "completion"],
    }


def test_scan_uses_commit_as_revision_when_no_named_ref_exists(settings):
    repository = settings.model_root_paths[0] / "models--org--model"
    snapshot = repository / "snapshots" / "commit123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"architectures": ["CausalLM"]}')
    (snapshot / "model.safetensors").write_bytes(b"weights")
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    service = DiscoveryService(settings.model_root_paths)
    with database.session_factory() as db:
        model = service.scan_models(db)[0]

    assert model.commit_hash == "commit123"
    assert model.revision == "commit123"
