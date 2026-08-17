from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any, Literal

import docker
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings

RuntimeName = Literal["vllm", "sglang", "llama_cpp"]

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
MAX_PROBE_WARNING_CHARS = 500
MAX_PROBE_WARNINGS = 8


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
    "llama_cpp": {
        "generation_defaults": list(GENERATION_DEFAULTS),
        "quantization_methods": ["auto", "gguf"],
        "quantization_mapping": {},
        "speculative_methods": [],
        "method_mapping": {},
        "speculative_transport": "none",
    },
}

PROBE_COMMANDS: dict[RuntimeName, tuple[str, list[str]]] = {
    "vllm": ("vllm", ["serve", "--help=speculative_config"]),
    "sglang": ("python3", ["-m", "sglang.launch_server", "--help"]),
}


def _has_flag(help_text: str, flag: str) -> bool:
    pattern = rf"(?<!\S){re.escape(flag)}(?=$|[\s=,])"
    return re.search(pattern, help_text, flags=re.MULTILINE) is not None


def _choice_values(help_text: str, flag: str) -> list[str]:
    pattern = rf"(?<!\S){re.escape(flag)}(?=[\s=])[^\n{{}}]*\{{([^}}]+)\}}"
    match = re.search(pattern, help_text)
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
    if runtime == "llama_cpp":
        raise ValueError("llama.cpp capabilities use the server-side manifest")
    manifest = CONSERVATIVE_MANIFESTS[runtime]
    supported_mapping = dict(manifest["method_mapping"])
    warnings: list[str] = []

    if runtime == "vllm":
        has_speculative_config = _has_flag(help_text, "--speculative-config")
        choices = (
            _choice_values(help_text, "--speculative-config.method")
            if has_speculative_config
            else []
        )
        speculative_methods = [
            method
            for method in ("draft_model", "eagle", "eagle3", "mtp")
            if method in choices
        ]
        for method in speculative_methods:
            supported_mapping.setdefault(method, method)
        transport = "json" if has_speculative_config else "none"
    else:
        has_speculative_flags = _has_flag(help_text, "--speculative-algorithm")
        choices = (
            _choice_values(help_text, "--speculative-algorithm")
            if has_speculative_flags
            else []
        )
        normalized_choices = {choice.upper() for choice in choices}
        speculative_methods = [
            method
            for method, runtime_method in supported_mapping.items()
            if runtime_method in normalized_choices
        ]
        transport = "flags" if has_speculative_flags else "none"

    if transport != "none" and not speculative_methods:
        warnings.append("Runtime help did not expose any recognized speculative methods")
    method_mapping = {
        method: supported_mapping[method]
        for method in speculative_methods
        if method in supported_mapping
    }
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
        warnings=warnings,
    )


def _read_bounded_logs(logs: bytes | str | Iterable[bytes | str]) -> bytes:
    if isinstance(logs, str):
        return logs.encode("utf-8")[:MAX_PROBE_LOG_BYTES]
    if isinstance(logs, bytes):
        return logs[:MAX_PROBE_LOG_BYTES]

    output = bytearray()
    primary_error: BaseException | None = None
    try:
        for chunk in logs:
            encoded = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            remaining = MAX_PROBE_LOG_BYTES - len(output)
            if remaining <= 0:
                break
            output.extend(encoded[:remaining])
            if len(output) >= MAX_PROBE_LOG_BYTES:
                break
        return bytes(output)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close = getattr(logs, "close", None)
        if callable(close):
            try:
                close()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Runtime probe log stream cleanup failed: {cleanup_error}")


def _bounded_warning(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= MAX_PROBE_WARNING_CHARS:
        return normalized
    return normalized[: MAX_PROBE_WARNING_CHARS - 3] + "..."


def _probe_failure_warnings(exc: Exception) -> list[str]:
    try:
        message = str(exc).strip() or type(exc).__name__
    except Exception:
        message = type(exc).__name__
    warnings = [_bounded_warning(f"Runtime capability probe failed: {message}")]

    notes = getattr(exc, "__notes__", None)
    if isinstance(notes, (list, tuple)):
        for note in notes:
            if len(warnings) >= MAX_PROBE_WARNINGS:
                break
            if isinstance(note, str) and note.strip():
                warnings.append(_bounded_warning(note))
    return warnings


def run_runtime_probe(
    client: Any,
    settings: Settings,
    runtime: RuntimeName,
    image: str,
) -> str:
    entrypoint, command = PROBE_COMMANDS[runtime]
    container = None
    primary_error: BaseException | None = None
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
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Runtime probe container cleanup failed: {cleanup_error}")


ProbeRunner = Callable[[RuntimeName, str], str]


class _KeyLockEntry:
    def __init__(self):
        self.lock = Lock()
        self.users = 0


class RuntimeCapabilityService:
    def __init__(
        self,
        *,
        settings: Settings,
        docker_client: Any | None = None,
        probe_runner: ProbeRunner | None = None,
    ):
        self.settings = settings
        if docker_client is None:
            self.docker_client = docker.from_env()
        elif callable(docker_client):
            self.docker_client = docker_client()
        else:
            self.docker_client = docker_client
        self.probe_runner = probe_runner or self._run_probe
        self._cache: dict[tuple[RuntimeName, str], RuntimeCapabilities] = {}
        self._key_locks: dict[tuple[RuntimeName, str], _KeyLockEntry] = {}
        self._key_locks_guard = Lock()

    def _run_probe(self, runtime: RuntimeName, image: str) -> str:
        return run_runtime_probe(self.docker_client, self.settings, runtime, image)

    @contextmanager
    def _key_lock(self, cache_key: tuple[RuntimeName, str]) -> Iterator[None]:
        with self._key_locks_guard:
            entry = self._key_locks.get(cache_key)
            if entry is None:
                entry = _KeyLockEntry()
                self._key_locks[cache_key] = entry
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._key_locks_guard:
                entry.users -= 1
                if entry.users == 0 and self._key_locks.get(cache_key) is entry:
                    del self._key_locks[cache_key]

    def get(self, runtime: RuntimeName, image: str) -> RuntimeCapabilities:
        image_digest = str(self.docker_client.images.get(image).id)
        cache_key = (runtime, image_digest)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"image": image}, deep=True)

        with self._key_lock(cache_key):
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.model_copy(update={"image": image}, deep=True)

            try:
                if runtime == "llama_cpp":
                    capabilities = RuntimeCapabilities(
                        runtime=runtime,
                        image=image,
                        image_digest=image_digest,
                        source="manifest",
                        **CONSERVATIVE_MANIFESTS[runtime],
                    )
                else:
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
                    warnings=_probe_failure_warnings(exc),
                )

            self._cache[cache_key] = capabilities.model_copy(deep=True)
            return capabilities
