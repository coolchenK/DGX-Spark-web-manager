from app.services.discovery import (
    container_candidate,
    infer_runtime,
    parse_hf_cache_repository,
)


def test_parse_hugging_face_cache_repository():
    assert (
        parse_hf_cache_repository(
            "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
        )
        == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    )
    assert parse_hf_cache_repository(".locks") is None


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
