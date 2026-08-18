from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError, Event, Lock
from typing import Any

import pytest
from app.config import Settings
from app.services.runtime_capabilities import (
    GENERATION_DEFAULTS,
    MAX_PROBE_LOG_BYTES,
    RuntimeCapabilityService,
    _read_bounded_logs,
    parse_runtime_help,
    run_runtime_probe,
)
from docker.types import DeviceRequest
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


def test_vllm_parser_does_not_match_prefixed_flags():
    capabilities = parse_runtime_help(
        "vllm",
        "--speculative-config-extra VALUE\n"
        "--speculative-config.method-extra {draft_model,eagle3,mtp}",
        image="vllm:test",
        image_digest="sha256:vllm",
    )

    assert capabilities.speculative_transport == "none"
    assert capabilities.speculative_methods == []


def test_vllm_parser_keeps_transport_without_inventing_methods():
    capabilities = parse_runtime_help(
        "vllm",
        "--speculative-config SPECULATIVE_CONFIG",
        image="vllm:test",
        image_digest="sha256:vllm",
    )

    assert capabilities.speculative_transport == "json"
    assert capabilities.speculative_methods == ["draft_model", "eagle", "eagle3", "mtp"]
    assert capabilities.warnings


def test_sglang_parser_ignores_unrelated_speculative_flags():
    capabilities = parse_runtime_help(
        "sglang",
        "--speculative-num-steps N",
        image="sglang:test",
        image_digest="sha256:sglang",
    )

    assert capabilities.speculative_transport == "none"
    assert capabilities.speculative_methods == []


def test_sglang_parser_maps_exact_algorithm_choices():
    capabilities = parse_runtime_help(
        "sglang",
        "--speculative-algorithm {EAGLE,EAGLE3,NEXTN,STANDALONE}",
        image="sglang:test",
        image_digest="sha256:sglang",
    )

    assert capabilities.speculative_transport == "flags"
    assert capabilities.speculative_methods == ["draft_model", "eagle", "eagle3", "mtp"]


def test_sglang_parser_keeps_transport_without_inventing_methods():
    capabilities = parse_runtime_help(
        "sglang",
        "--speculative-algorithm ALGORITHM",
        image="sglang:test",
        image_digest="sha256:sglang",
    )

    assert capabilities.speculative_transport == "flags"
    assert capabilities.speculative_methods == []
    assert capabilities.warnings


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


def test_capability_service_accepts_a_docker_client_factory():
    image = type("Image", (), {"id": "sha256:factory"})()
    images = type("Images", (), {"get": lambda self, _name: image})()
    client = type("Client", (), {"images": images})()
    factory_calls = 0

    def client_factory():
        nonlocal factory_calls
        factory_calls += 1
        return client

    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client_factory,
        probe_runner=lambda _runtime, _image: "--speculative-config",
    )

    result = service.get("vllm", "vllm:test")

    assert result.image_digest == "sha256:factory"
    assert factory_calls == 1


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


def test_same_digest_concurrent_misses_run_only_one_probe():
    image_barrier = Barrier(2)

    class Images:
        @staticmethod
        def get(_image: str):
            image_barrier.wait(timeout=2)
            return type("Image", (), {"id": "sha256:shared-concurrent"})()

    client = type("Client", (), {"images": Images()})()
    probe_started = Event()
    second_probe_started = Event()
    release_probe = Event()
    calls_lock = Lock()
    probe_calls = 0

    def probe_runner(_runtime: str, _image: str) -> str:
        nonlocal probe_calls
        with calls_lock:
            probe_calls += 1
            if probe_calls == 2:
                second_probe_started.set()
        probe_started.set()
        assert release_probe.wait(timeout=2)
        return "--speculative-config"

    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client,
        probe_runner=probe_runner,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.get, "vllm", "vllm:first")
        second = executor.submit(service.get, "vllm", "vllm:second")
        assert probe_started.wait(timeout=2)
        try:
            assert not second_probe_started.wait(timeout=0.5)
        finally:
            release_probe.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert probe_calls == 1
    assert {result.image for result in results} == {"vllm:first", "vllm:second"}
    assert service._key_locks == {}


def test_different_digest_misses_probe_in_parallel():
    image_ids = {"vllm:first": "sha256:first", "vllm:second": "sha256:second"}

    class Images:
        @staticmethod
        def get(image: str):
            return type("Image", (), {"id": image_ids[image]})()

    client = type("Client", (), {"images": Images()})()
    probe_barrier = Barrier(2)

    def probe_runner(_runtime: str, _image: str) -> str:
        try:
            probe_barrier.wait(timeout=1)
        except BrokenBarrierError as exc:
            raise RuntimeError("probes were serialized") from exc
        return "--speculative-config"

    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client,
        probe_runner=probe_runner,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda image: service.get("vllm", image),
                ("vllm:first", "vllm:second"),
            )
        )

    assert [result.source for result in results] == ["probe", "probe"]
    assert service._key_locks == {}


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


def test_probe_fallback_preserves_bounded_cleanup_diagnostics():
    class Images:
        @staticmethod
        def get(_image: str):
            return type("Image", (), {"id": "sha256:cleanup-failed"})()

    client = type("Client", (), {"images": Images()})()

    def failed_probe(_runtime: str, _image: str) -> str:
        error = RuntimeError("wait failed " + "x" * 2_000)
        error.add_note("Runtime probe container cleanup failed: " + "y" * 2_000)
        error.__notes__.append({"not": "a string"})
        raise error

    service = RuntimeCapabilityService(
        settings=make_settings(),
        docker_client=client,
        probe_runner=failed_probe,
    )

    capabilities = service.get("vllm", "vllm:test")

    assert capabilities.source == "manifest"
    assert len(capabilities.warnings) == 2
    assert capabilities.warnings[0].startswith("Runtime capability probe failed: wait failed")
    assert capabilities.warnings[1].startswith("Runtime probe container cleanup failed:")
    assert all(len(warning) <= 500 for warning in capabilities.warnings)
    assert "{'not': 'a string'}" not in " ".join(capabilities.warnings)


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
        "device_requests": [DeviceRequest(count=-1, capabilities=[["gpu"]])],
        "environment": {
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        },
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


def test_bounded_log_reader_closes_stream_after_reaching_limit():
    class LogStream:
        closed = False

        def __iter__(self):
            yield b"x" * MAX_PROBE_LOG_BYTES
            raise AssertionError("reader consumed beyond its byte limit")

        def close(self):
            self.closed = True

    stream = LogStream()

    output = _read_bounded_logs(stream)

    assert len(output) == MAX_PROBE_LOG_BYTES
    assert stream.closed is True


def test_bounded_log_reader_closes_stream_when_iteration_fails():
    class LogStream:
        closed = False

        def __iter__(self):
            yield b"partial"
            raise RuntimeError("stream failed")

        def close(self):
            self.closed = True

    stream = LogStream()

    with pytest.raises(RuntimeError, match="stream failed"):
        _read_bounded_logs(stream)

    assert stream.closed is True


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("wait", "wait failed"),
        ("logs", "logs failed"),
        ("status", "exited with status 7"),
    ],
)
def test_cleanup_failure_does_not_replace_probe_failure(failure: str, message: str):
    class Container:
        @staticmethod
        def wait(*, timeout: int):
            if failure == "wait":
                raise RuntimeError("wait failed")
            return {"StatusCode": 7 if failure == "status" else 0}

        @staticmethod
        def logs(**_kwargs: Any):
            if failure == "logs":
                raise RuntimeError("logs failed")
            return b"help"

        @staticmethod
        def remove(*, force: bool):
            raise RuntimeError("remove failed")

    container = Container()
    containers = type("Containers", (), {"run": lambda self, *args, **kwargs: container})()
    client = type("Client", (), {"containers": containers})()

    with pytest.raises(RuntimeError, match=message) as captured:
        run_runtime_probe(client, make_settings(), "vllm", "vllm:test")

    assert any("remove failed" in note for note in getattr(captured.value, "__notes__", []))


def test_cleanup_failure_is_reported_after_successful_probe():
    cleanup_error = RuntimeError("remove failed")

    class Container:
        @staticmethod
        def wait(*, timeout: int):
            return {"StatusCode": 0}

        @staticmethod
        def logs(**_kwargs: Any):
            return b"help"

        @staticmethod
        def remove(*, force: bool):
            raise cleanup_error

    container = Container()
    containers = type("Containers", (), {"run": lambda self, *args, **kwargs: container})()
    client = type("Client", (), {"containers": containers})()

    with pytest.raises(RuntimeError, match="remove failed") as captured:
        run_runtime_probe(client, make_settings(), "vllm", "vllm:test")

    assert captured.value is cleanup_error
