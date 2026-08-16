from __future__ import annotations

from typing import Any

import pytest
from app.config import Settings
from app.services.runtime_capabilities import (
    GENERATION_DEFAULTS,
    MAX_PROBE_LOG_BYTES,
    RuntimeCapabilityService,
    parse_runtime_help,
    run_runtime_probe,
)
from pydantic import ValidationError


def make_settings(**overrides: Any) -> Settings:
    values = {
        "database_url": "sqlite:///:memory:",
        "secret_key": "test-secret-key-with-at-least-32-characters",
        "admin_password": "Test-password-1234",
    }
    values.update(overrides)
    return Settings(**values)


def test_parse_vllm_help_reports_known_capabilities():
    help_text = """
      --speculative-config SPECULATIVE_CONFIG
      --speculative-config.method {draft_model,eagle,eagle3,mtp}
      --max-model-len MAX_MODEL_LEN
      --max-num-seqs MAX_NUM_SEQS
      --quantization {auto,awq,gptq,modelopt_fp4}
    """

    capabilities = parse_runtime_help(
        "vllm",
        help_text,
        image="vllm:test",
        image_digest="sha256:vllm",
    )

    assert capabilities.source == "probe"
    assert capabilities.speculative_methods == ["draft_model", "eagle", "eagle3", "mtp"]
    assert capabilities.speculative_transport == "json"
    assert capabilities.generation_defaults == list(GENERATION_DEFAULTS)
    assert "temperature" in capabilities.generation_defaults
    assert "modelopt_fp4" in capabilities.quantization_methods


def test_capability_service_caches_by_runtime_and_image_digest():
    class Image:
        id = "sha256:shared"

    class Images:
        calls: list[str] = []

        def get(self, image: str):
            self.calls.append(image)
            return Image()

    images = Images()
    client = type("Client", (), {"images": images})()
    probe_calls: list[tuple[str, str]] = []

    def probe_runner(runtime: str, image: str) -> str:
        probe_calls.append((runtime, image))
        return "--speculative-config --speculative-config.method {draft_model,eagle3,mtp}"

    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client,
        probe_runner=probe_runner,
    )

    first = service.get("vllm", "vllm:first")
    second = service.get("vllm", "vllm:second")

    assert images.calls == ["vllm:first", "vllm:second"]
    assert probe_calls == [("vllm", "vllm:first")]
    assert first.image == "vllm:first"
    assert second.image == "vllm:second"
    assert second.image_digest == "sha256:shared"


def test_capability_service_cache_is_not_mutated_by_callers():
    image = type("Image", (), {"id": "sha256:immutable"})()
    images = type("Images", (), {"get": lambda self, _name: image})()
    client = type("Client", (), {"images": images})()
    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client,
        probe_runner=lambda _runtime, _image: "--speculative-config",
    )

    first = service.get("vllm", "vllm:first")
    first.speculative_methods.append("caller-added")

    second = service.get("vllm", "vllm:second")

    assert "caller-added" not in second.speculative_methods


def test_probe_failure_falls_back_to_conservative_manifest():
    class Images:
        @staticmethod
        def get(_image: str):
            return type("Image", (), {"id": "sha256:failed"})()

    client = type("Client", (), {"images": Images()})()

    def failed_probe(_runtime: str, _image: str) -> str:
        raise RuntimeError("probe failed")

    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client,
        probe_runner=failed_probe,
    )

    capabilities = service.get("sglang", "sglang:test")

    assert capabilities.source == "manifest"
    assert capabilities.warnings
    assert capabilities.method_mapping["draft_model"] == "STANDALONE"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_probe_timeout_seconds", 4),
        ("runtime_probe_timeout_seconds", 181),
        ("recommendation_cache_ttl_seconds", 59),
        ("recommendation_cache_ttl_seconds", 86_401),
        ("recommendation_card_max_chars", 9_999),
        ("recommendation_card_max_chars", 500_001),
        ("memory_reserve_fraction", 0.049),
        ("memory_reserve_fraction", 0.301),
        ("memory_reserve_min_bytes", 1024**3 - 1),
    ],
)
def test_recommendation_settings_reject_out_of_range_values(field: str, value: Any):
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_recommendation_settings_have_expected_defaults():
    settings = make_settings()

    assert settings.runtime_probe_timeout_seconds == 45
    assert settings.recommendation_cache_ttl_seconds == 900
    assert settings.recommendation_card_max_chars == 100_000
    assert settings.memory_reserve_fraction == 0.10
    assert settings.memory_reserve_min_bytes == 8 * 1024**3


def test_production_probe_uses_fixed_isolated_container_and_removes_it():
    class Container:
        def __init__(self):
            self.wait_timeout: int | None = None
            self.log_kwargs: dict[str, Any] | None = None
            self.removed_force: bool | None = None

        def wait(self, *, timeout: int):
            self.wait_timeout = timeout
            return {"StatusCode": 0}

        def logs(self, **kwargs: Any):
            self.log_kwargs = kwargs
            return iter([b"a" * MAX_PROBE_LOG_BYTES, b"not-read"])

        def remove(self, *, force: bool):
            self.removed_force = force

    container = Container()

    class Containers:
        def __init__(self):
            self.args: tuple[Any, ...] | None = None
            self.kwargs: dict[str, Any] | None = None

        def run(self, *args: Any, **kwargs: Any):
            self.args = args
            self.kwargs = kwargs
            return container

    containers = Containers()
    client = type("Client", (), {"containers": containers})()

    output = run_runtime_probe(client, make_settings(), "vllm", "vllm:test")

    assert len(output.encode()) == MAX_PROBE_LOG_BYTES
    assert containers.args == ("vllm:test",)
    assert containers.kwargs == {
        "entrypoint": "vllm",
        "command": ["serve", "--help=speculative_config"],
        "detach": True,
        "network_disabled": True,
        "network_mode": "none",
        "stdin_open": False,
        "tty": False,
        "volumes": {},
    }
    assert container.wait_timeout == 45
    assert container.log_kwargs == {
        "stdout": True,
        "stderr": True,
        "stream": True,
        "follow": False,
    }
    assert container.removed_force is True


def test_production_probe_removes_container_when_wait_fails():
    class Container:
        removed_force: bool | None = None

        @staticmethod
        def wait(*, timeout: int):
            raise TimeoutError(timeout)

        def remove(self, *, force: bool):
            self.removed_force = force

    container = Container()
    containers = type("Containers", (), {"run": lambda self, *args, **kwargs: container})()
    client = type("Client", (), {"containers": containers})()

    with pytest.raises(TimeoutError):
        run_runtime_probe(client, make_settings(), "sglang", "sglang:test")

    assert container.removed_force is True
