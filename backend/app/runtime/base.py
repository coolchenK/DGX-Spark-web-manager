from __future__ import annotations

import json
import platform
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

RecommendationSource = Literal[
    "model_card", "local_config", "runtime_default", "device_rule", "ai"
]

QuantizationMethod = Literal[
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
]


class GenerationDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0, le=1_000_000)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    stop: str | list[str] | None = None

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None):
        values = [value] if isinstance(value, str) else value or []
        if len(values) > 16 or any(not item or len(item) > 500 for item in values):
            raise ValueError(
                "stop must contain 1-16 non-empty strings of at most 500 characters"
            )
        return value


class SpeculativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_model_id: str = Field(min_length=1, max_length=64)
    method: Literal["draft_model", "eagle", "eagle3", "mtp"]
    num_speculative_tokens: int | None = Field(default=None, ge=1, le=64)
    num_steps: int | None = Field(default=None, ge=1, le=32)
    eagle_top_k: int | None = Field(default=None, ge=1, le=32)
    num_draft_tokens: int | None = Field(default=None, ge=1, le=256)
    manual_review_acknowledged: bool = False

    @model_validator(mode="after")
    def validate_tuning_group(self):
        values = (self.num_steps, self.eagle_top_k, self.num_draft_tokens)
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("set num_steps, eagle_top_k and num_draft_tokens together")
        return self


class ResourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    reserved_bytes: int = Field(ge=0)


class RecommendationProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_id: str | None = Field(default=None, max_length=64)
    resource_snapshot: ResourceSnapshot
    modified_fields: list[str] = Field(default_factory=list, max_length=64)
    sources: dict[str, RecommendationSource] = Field(default_factory=dict)


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    model_id: str | None = None
    model_path: str
    api_model_name: str = Field(min_length=1, max_length=255)
    route_alias: str | None = Field(default=None, min_length=1, max_length=255)
    runtime: str
    image: str
    port: int = Field(ge=1024, le=65535)
    context_length: int = Field(default=32768, ge=1024, le=1_048_576)
    memory_fraction: float = Field(default=0.8, ge=0.05, le=0.98)
    max_concurrency: int = Field(default=8, ge=1, le=1024)
    max_batched_tokens: int | None = Field(default=None, ge=1024, le=1_048_576)
    quantization: QuantizationMethod | None = None
    trust_remote_code: bool = False
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    speculative: SpeculativeConfig | None = None
    recommendation: RecommendationProvenance | None = None

    @field_validator("api_model_name", "route_alias")
    @classmethod
    def validate_api_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
            raise ValueError("API model name contains unsupported characters")
        return value


class ResolvedDeploymentSpec(DeploymentSpec):
    resolved_draft_model_path: str | None = None
    draft_container_model_path: str | None = None
    speculative_runtime_method: str | None = None

    def public_dump(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in DeploymentSpec.model_fields}


def deterministic_container_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"dgx-{slug[:55] or 'model'}"


def validate_model_path(path: Path, model_roots: tuple[Path, ...]) -> Path:
    resolved = path.resolve()
    for root in model_roots:
        resolved_root = root.resolve()
        if resolved == resolved_root or resolved.is_relative_to(resolved_root):
            if not resolved.exists():
                raise ValueError("Model path does not exist")
            return resolved
    raise ValueError("Model path is outside configured model roots")


class RuntimeAdapter(ABC):
    runtime: str

    def __init__(self, *, allowed_images: set[str], model_roots: tuple[Path, ...]):
        self.allowed_images = allowed_images
        self.model_roots = model_roots

    def validate(self, spec: DeploymentSpec) -> Path:
        if spec.runtime != self.runtime:
            raise ValueError(f"Adapter {self.runtime} cannot deploy runtime {spec.runtime}")
        if spec.image not in self.allowed_images:
            raise ValueError(f"Image is not allowed for {self.runtime}")
        return validate_model_path(Path(spec.model_path), self.model_roots)

    def container_model_path(self, spec: DeploymentSpec) -> str:
        model_path = self.validate(spec)
        for root in self.model_roots:
            resolved_root = root.resolve()
            if model_path == resolved_root or model_path.is_relative_to(resolved_root):
                relative = model_path.relative_to(resolved_root)
                return str(Path("/models", relative)).replace("\\", "/")
        raise ValueError("Model path is outside configured model roots")

    def detect_environment(self, spec: DeploymentSpec) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "architecture": platform.machine(),
            "image": spec.image,
            "image_allowed": spec.image in self.allowed_images,
        }

    def check_model_compatibility(self, model_path: Path) -> dict[str, Any]:
        config_path = model_path / "config.json"
        architectures: list[str] = []
        config_error: str | None = None
        if config_path.is_file():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("architectures"), list):
                    architectures = [str(item) for item in payload["architectures"][:20]]
            except (OSError, json.JSONDecodeError) as exc:
                config_error = str(exc)
        weight_files = [
            path
            for pattern in ("*.safetensors", "*.bin")
            for path in model_path.rglob(pattern)
            if path.is_file()
        ]
        reasons: list[str] = []
        if not config_path.is_file():
            reasons.append("config.json was not found")
        if config_error:
            reasons.append(f"config.json is invalid: {config_error}")
        if not weight_files:
            reasons.append("No Safetensors or PyTorch weight files were found")
        return {
            "compatible": not reasons,
            "architectures": architectures,
            "weight_files": len(weight_files),
            "reasons": reasons,
        }

    @staticmethod
    def model_size(model_path: Path) -> int:
        total = 0
        seen: set[Path] = set()
        for path in model_path.rglob("*"):
            try:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def openai_capabilities(self) -> list[str]:
        return ["chat", "completion"]

    def start(self, container: Any) -> None:
        container.start()

    def stop(self, container: Any, *, timeout: int = 30) -> None:
        container.stop(timeout=timeout)

    def restart(self, container: Any, *, timeout: int = 30) -> None:
        container.restart(timeout=timeout)

    def health_check(self, endpoint: str, *, timeout: float = 3) -> bool:
        try:
            return httpx.get(f"{endpoint}/v1/models", timeout=timeout).is_success
        except httpx.HTTPError:
            return False

    def logs(self, container: Any, *, tail: int = 500) -> str:
        value = container.logs(tail=min(max(tail, 1), 5000), timestamps=True)
        return value.decode("utf-8", errors="replace")[-500_000:]

    def metrics(self, container: Any) -> dict[str, float | int | None]:
        stats = container.stats(stream=False)
        cpu_stats = stats.get("cpu_stats") or {}
        previous = stats.get("precpu_stats") or {}
        cpu_delta = (cpu_stats.get("cpu_usage") or {}).get("total_usage", 0) - (
            (previous.get("cpu_usage") or {}).get("total_usage", 0)
        )
        system_delta = cpu_stats.get("system_cpu_usage", 0) - previous.get(
            "system_cpu_usage", 0
        )
        online_cpus = cpu_stats.get("online_cpus") or len(
            (cpu_stats.get("cpu_usage") or {}).get("percpu_usage") or []
        )
        cpu_percent = (
            cpu_delta / system_delta * max(online_cpus, 1) * 100
            if cpu_delta > 0 and system_delta > 0
            else 0.0
        )
        memory_stats = stats.get("memory_stats") or {}
        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_used_bytes": memory_stats.get("usage"),
            "memory_limit_bytes": memory_stats.get("limit"),
        }

    def uninstall(self, container: Any) -> None:
        if getattr(container, "status", None) == "running":
            self.stop(container)
        container.remove()

    def preview(self, spec: DeploymentSpec) -> dict[str, Any]:
        model_path = self.validate(spec)
        compatibility = self.check_model_compatibility(model_path)
        model_size = self.model_size(model_path)
        route_name = spec.route_alias or spec.api_model_name
        return {
            "spec": spec.model_dump(),
            "container_name": deterministic_container_name(spec.name),
            "runtime": self.runtime,
            "image": spec.image,
            "model_path": str(model_path),
            "container_model_path": self.container_model_path(spec),
            "port": spec.port,
            "route_alias": spec.route_alias,
            "command": self.command(spec),
            "estimated_memory_fraction": spec.memory_fraction,
            "estimated_disk_bytes": model_size,
            "estimated_memory_bytes": int(model_size * 1.2),
            "environment": self.detect_environment(spec),
            "compatibility": compatibility,
            "capabilities": self.openai_capabilities(),
            "operations": [
                f"Create manager-owned container {deterministic_container_name(spec.name)}",
                "Mount the selected model root read-only",
                f"Publish runtime port 8000 on host port {spec.port}",
                "Wait for runtime startup with cancellation enabled",
                "Probe /v1/models and register the gateway route",
            ],
            "api_example": (
                "client.chat.completions.create("
                f"model=\"{route_name}\", "
                "messages=[{\"role\": \"user\", \"content\": \"Hello\"}])"
            ),
            "rollback": "Remove the newly created manager-owned container and retain model files.",
        }

    @abstractmethod
    def command(self, spec: DeploymentSpec) -> list[str]:
        raise NotImplementedError

