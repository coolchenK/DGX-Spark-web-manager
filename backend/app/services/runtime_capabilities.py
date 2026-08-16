from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, Literal

import docker
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings

RuntimeName = Literal["vllm", "sglang"]

GENERATION_DEFAULTS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "stop",
)

QUANTIZATION_METHODS = (
    "auto",
    "awq",
    "gptq",
    "fp8",
    "bitsandbytes",
    "marlin",
    "gguf",
    "modelopt",
    "modelopt_fp4",
    "nvfp4_online",
    "compressed-tensors",
)

MAX_PROBE_LOG_BYTES = 128 * 1024


class RuntimeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeName
    image: str
    image_digest: str
    source: Literal["probe", "manifest"]
    generation_defaults: list[str]
    quantization_methods: list[str]
    quantization_mapping: dict[str, str]
    speculative_methods: list[str]
    method_mapping: dict[str, str]
    speculative_transport: Literal["json", "flags", "none"]
    warnings: list[str] = Field(default_factory=list)


CONSERVATIVE_MANIFESTS: dict[RuntimeName, dict[str, Any]] = {
    "vllm": {
        "generation_defaults": list(GENERATION_DEFAULTS),
        "quantization_methods": list(QUANTIZATION_METHODS),
        "quantization_mapping": {"nvfp4": "modelopt_fp4"},
        "speculative_methods": ["draft_model", "eagle3", "mtp"],
        "method_mapping": {
            "draft_model": "draft_model",
            "eagle3": "eagle3",
            "mtp": "mtp",
        },
        "speculative_transport": "json",
    },
    "sglang": {
        "generation_defaults": list(GENERATION_DEFAULTS),
        "quantization_methods": list(QUANTIZATION_METHODS),
        "quantization_mapping": {"nvfp4": "modelopt_fp4"},
        "speculative_methods": ["draft_model", "eagle", "eagle3", "mtp"],
        "method_mapping": {
            "draft_model": "STANDALONE",
            "eagle": "EAGLE",
            "eagle3": "EAGLE3",
            "mtp": "NEXTN",
        },
        "speculative_transport": "flags",
    },
}

PROBE_COMMANDS: dict[RuntimeName, tuple[str, list[str]]] = {
    "vllm": ("vllm", ["serve", "--help=speculative_config"]),
    "sglang": ("python3", ["-m", "sglang.launch_server", "--help"]),
}


def _choice_values(help_text: str, flag: str) -> list[str]:
    match = re.search(rf"{re.escape(flag)}[^\n{{}}]*\{{([^}}]+)\}}", help_text)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def parse_runtime_help(
    runtime: RuntimeName,
    help_text: str,
    *,
    image: str,
    image_digest: str,
) -> RuntimeCapabilities:
    manifest = CONSERVATIVE_MANIFESTS[runtime]
    method_mapping = dict(manifest["method_mapping"])

    if runtime == "vllm":
        has_speculative_config = "--speculative-config" in help_text
        choices = _choice_values(help_text, "--speculative-config.method")
        speculative_methods = [
            method
            for method in ("draft_model", "eagle", "eagle3", "mtp")
            if method in choices
        ]
        if not choices and has_speculative_config:
            speculative_methods = list(manifest["speculative_methods"])
        for method in speculative_methods:
            method_mapping.setdefault(method, method)
        transport = "json" if has_speculative_config else "none"
    else:
        has_speculative_flags = "--speculative-" in help_text
        choices = _choice_values(help_text, "--speculative-algorithm")
        normalized_choices = {choice.upper() for choice in choices}
        speculative_methods = [
            method
            for method, runtime_method in method_mapping.items()
            if runtime_method in normalized_choices
        ]
        if not choices and has_speculative_flags:
            speculative_methods = list(manifest["speculative_methods"])
        transport = "flags" if has_speculative_flags else "none"

    return RuntimeCapabilities(
        runtime=runtime,
        image=image,
        image_digest=image_digest,
        source="probe",
        generation_defaults=list(GENERATION_DEFAULTS),
        quantization_methods=list(QUANTIZATION_METHODS),
        quantization_mapping=dict(manifest["quantization_mapping"]),
        speculative_methods=speculative_methods,
        method_mapping=method_mapping,
        speculative_transport=transport,
    )


def _read_bounded_logs(logs: bytes | str | Iterable[bytes | str]) -> bytes:
    if isinstance(logs, str):
        return logs.encode("utf-8")[:MAX_PROBE_LOG_BYTES]
    if isinstance(logs, bytes):
        return logs[:MAX_PROBE_LOG_BYTES]

    output = bytearray()
    for chunk in logs:
        encoded = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        remaining = MAX_PROBE_LOG_BYTES - len(output)
        if remaining <= 0:
            break
        output.extend(encoded[:remaining])
        if len(output) >= MAX_PROBE_LOG_BYTES:
            break
    return bytes(output)


def run_runtime_probe(
    client: Any,
    settings: Settings,
    runtime: RuntimeName,
    image: str,
) -> str:
    entrypoint, command = PROBE_COMMANDS[runtime]
    container = None
    try:
        container = client.containers.run(
            image,
            entrypoint=entrypoint,
            command=list(command),
            detach=True,
            network_disabled=True,
            network_mode="none",
            stdin_open=False,
            tty=False,
            volumes={},
        )
        result = container.wait(timeout=settings.runtime_probe_timeout_seconds)
        raw_logs = _read_bounded_logs(
            container.logs(stdout=True, stderr=True, stream=True, follow=False)
        )
        status_code = result.get("StatusCode") if isinstance(result, dict) else result
        if status_code != 0:
            raise RuntimeError(f"Runtime capability probe exited with status {status_code}")
        return raw_logs.decode("utf-8", errors="replace")
    finally:
        if container is not None:
            container.remove(force=True)


ProbeRunner = Callable[[RuntimeName, str], str]


class RuntimeCapabilityService:
    def __init__(
        self,
        *,
        settings: Settings,
        docker_client: Any | None = None,
        probe_runner: ProbeRunner | None = None,
    ):
        self.settings = settings
        self.docker_client = docker_client if docker_client is not None else docker.from_env()
        self.probe_runner = probe_runner or self._run_probe
        self._cache: dict[tuple[RuntimeName, str], RuntimeCapabilities] = {}

    def _run_probe(self, runtime: RuntimeName, image: str) -> str:
        return run_runtime_probe(self.docker_client, self.settings, runtime, image)

    def get(self, runtime: RuntimeName, image: str) -> RuntimeCapabilities:
        image_digest = str(self.docker_client.images.get(image).id)
        cache_key = (runtime, image_digest)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"image": image}, deep=True)

        try:
            help_text = self.probe_runner(runtime, image)
            capabilities = parse_runtime_help(
                runtime,
                help_text,
                image=image,
                image_digest=image_digest,
            )
        except Exception as exc:
            capabilities = RuntimeCapabilities(
                runtime=runtime,
                image=image,
                image_digest=image_digest,
                source="manifest",
                **CONSERVATIVE_MANIFESTS[runtime],
                warnings=[f"Runtime capability probe failed: {exc}"],
            )

        self._cache[cache_key] = capabilities.model_copy(deep=True)
        return capabilities
