from __future__ import annotations

import json
import platform
import re
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path, PurePosixPath
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

ChatTemplateProfile = Literal["model", "qwen-fixed-v22.4"]
QWEN_FIXED_CHAT_TEMPLATE_PROFILE = "qwen-fixed-v22.4"
QWEN_FIXED_CHAT_TEMPLATE_VERSION = "qwen3.8-froggeric-v22.4"
QWEN_FIXED_CHAT_TEMPLATE_FILENAME = ".dgx-qwen-fixed-v22.4.jinja"
QWEN_FIXED_CHAT_TEMPLATE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "chat_templates"
    / "qwen_fixed_v22_4.jinja"
)


class LlamaCppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_file: str | None = Field(default=None, max_length=255)
    mmproj_file: str | None = Field(default=None, max_length=255)
    gpu_layers: int | Literal["all"] = "all"
    jinja: bool = True
    continuous_batching: bool = True
    mtp_enabled: bool = False
    mtp_tokens: int = Field(default=3, ge=1, le=64)

    @field_validator("model_file", "mmproj_file")
    @classmethod
    def validate_gguf_filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
            or not value.lower().endswith(".gguf")
        ):
            raise ValueError("llama.cpp files must be GGUF basenames")
        return value

    @field_validator("gpu_layers")
    @classmethod
    def validate_gpu_layers(cls, value: int | str) -> int | str:
        if value == "all":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000:
            raise ValueError("gpu_layers must be 'all' or an integer from 0 to 10000")
        return value


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

    draft_model_id: str | None = Field(default=None, min_length=1, max_length=64)
    method: Literal["draft_model", "dflash", "dspark", "eagle", "eagle3", "mtp"]
    num_speculative_tokens: int | None = Field(default=None, ge=1, le=64)
    num_steps: int | None = Field(default=None, ge=1, le=32)
    eagle_top_k: int | None = Field(default=None, ge=1, le=32)
    num_draft_tokens: int | None = Field(default=None, ge=1, le=256)
    manual_review_acknowledged: bool = False

    @model_validator(mode="after")
    def validate_tuning_group(self):
        if self.method != "mtp" and self.draft_model_id is None:
            raise ValueError("draft_model_id is required for external Draft Models")
        if self.method == "dflash":
            if self.num_steps is not None or self.eagle_top_k is not None:
                raise ValueError("DFlash tuning only accepts num_draft_tokens")
            return self
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
    port: int | None = Field(default=None, ge=1024, le=65535)
    context_length: int = Field(default=32768, ge=1024, le=1_048_576)
    max_total_tokens: int | None = Field(default=None, ge=1024, le=1_048_576)
    memory_fraction: float = Field(default=0.8, ge=0.05, le=0.98)
    max_concurrency: int = Field(default=8, ge=1, le=1024)
    max_batched_tokens: int | None = Field(default=None, ge=1024, le=1_048_576)
    quantization: QuantizationMethod | None = None
    trust_remote_code: bool = False
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    chat_template: ChatTemplateProfile | None = None
    chat_template_kwargs: dict[str, str | bool | int | float] | None = None
    speculative: SpeculativeConfig | None = None
    llama_cpp: LlamaCppConfig | None = None
    recommendation: RecommendationProvenance | None = None
    resource_warning_acknowledged: bool = False

    @field_validator("chat_template_kwargs")
    @classmethod
    def validate_chat_template_kwargs(
        cls, value: dict[str, str | bool | int | float] | None
    ) -> dict[str, str | bool | int | float] | None:
        """Defaults handed to the model's chat template at launch.

        Keys are whatever that template reads -- enable_thinking,
        reasoning_effort, preserve_thinking -- so they cannot be validated
        against a fixed list here; a wrong key is simply ignored by the
        template. The shape is constrained instead, since the value is
        serialised into a single argv element.
        """
        if not value:
            return None
        if len(value) > 16:
            raise ValueError("chat_template_kwargs accepts at most 16 entries")
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
                raise ValueError(f"Unsupported chat template kwarg name: {key!r}")
            if isinstance(item, str) and (not item or len(item) > 200):
                raise ValueError(
                    f"chat template kwarg {key!r} must be 1-200 characters"
                )
        return value

    @field_validator("api_model_name", "route_alias")
    @classmethod
    def validate_api_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
            raise ValueError("API model name contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_runtime_specific_settings(self):
        if (
            self.max_total_tokens is not None
            and self.max_total_tokens > self.context_length
        ):
            raise ValueError("max_total_tokens cannot exceed context_length")
        if self.runtime == "llama_cpp":
            if self.quantization not in {None, "auto", "gguf"}:
                raise ValueError("llama.cpp deployments require GGUF quantization")
            if self.trust_remote_code:
                raise ValueError("llama.cpp does not support trust_remote_code")
            if self.speculative is not None:
                raise ValueError("llama.cpp MTP must use llama_cpp settings")
        elif self.llama_cpp is not None:
            raise ValueError("llama_cpp settings require the llama_cpp runtime")
        return self


class ResolvedDeploymentSpec(DeploymentSpec):
    base_model_root: str | None = None
    base_host_model_root: str | None = None
    base_container_model_path: str | None = None
    resolved_draft_model_path: str | None = None
    draft_model_root: str | None = None
    draft_host_model_root: str | None = None
    draft_container_model_path: str | None = None
    speculative_runtime_method: str | None = None

    def public_dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", include=set(DeploymentSpec.model_fields))


def require_draft_container_path(spec: DeploymentSpec) -> str:
    if spec.speculative is None:
        return ""
    path = getattr(spec, "draft_container_model_path", None)
    if (
        not isinstance(path, str)
        or not path
        or not PurePosixPath(path).is_absolute()
        or "\x00" in path
    ):
        raise ValueError("resolved draft container path is required")
    return path


def require_speculative_runtime_method(spec: DeploymentSpec) -> str:
    if spec.speculative is None:
        return ""
    method = getattr(spec, "speculative_runtime_method", None)
    if not isinstance(method, str) or not method:
        raise ValueError("resolved speculative runtime method is required")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", method):
        raise ValueError("resolved speculative runtime method is invalid")
    return method


CHAT_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")


def is_qwen38_identifier(value: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    return "qwen38" in normalized


def recommended_chat_template_profile(
    spec: DeploymentSpec,
    *identifiers: object,
) -> ChatTemplateProfile | None:
    if spec.chat_template is not None:
        return spec.chat_template
    candidates = (
        spec.name,
        spec.model_id,
        spec.model_path,
        spec.api_model_name,
        spec.route_alias,
        *identifiers,
    )
    if any(is_qwen38_identifier(value) for value in candidates):
        return QWEN_FIXED_CHAT_TEMPLATE_PROFILE
    return None


def ensure_managed_chat_template(
    spec: DeploymentSpec,
    model_path: Path,
) -> Path | None:
    """Materialize a versioned template beside the model for read-only mounts."""
    profile = recommended_chat_template_profile(spec)
    if profile != QWEN_FIXED_CHAT_TEMPLATE_PROFILE:
        return None
    try:
        content = QWEN_FIXED_CHAT_TEMPLATE_SOURCE.read_bytes()
    except OSError as exc:
        raise ValueError("Bundled Qwen fixed chat template is unavailable") from exc
    if QWEN_FIXED_CHAT_TEMPLATE_VERSION.encode() not in content:
        raise ValueError("Bundled Qwen fixed chat template has an unexpected version")

    target = model_path / QWEN_FIXED_CHAT_TEMPLATE_FILENAME
    try:
        if not target.is_symlink() and target.is_file() and target.read_bytes() == content:
            return target
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=model_path,
                prefix=f".{QWEN_FIXED_CHAT_TEMPLATE_FILENAME}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_name = temporary.name
            Path(temporary_name).replace(target)
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
    except OSError as exc:
        raise ValueError("Qwen fixed chat template could not be installed") from exc
    return target


def effective_chat_template(
    spec: DeploymentSpec,
    model_path: Path,
) -> tuple[str, Path | None]:
    managed = ensure_managed_chat_template(spec, model_path)
    if managed is not None:
        try:
            return managed.read_text(encoding="utf-8"), managed
        except OSError as exc:
            raise ValueError("Qwen fixed chat template could not be read") from exc
    return chat_template_text(model_path), None


def container_chat_template_path(container_model_path: str, template_path: Path) -> str:
    return str(PurePosixPath(container_model_path) / template_path.name)


def command_with_managed_chat_template(
    runtime: str,
    command: list[str],
    container_model_path: str,
) -> list[str]:
    """Return an idempotently upgraded runtime command for the managed template."""
    template_flag = {
        "vllm": "--chat-template",
        "sglang": "--chat-template",
        "llama_cpp": "--chat-template-file",
    }.get(runtime)
    if template_flag is None:
        raise ValueError(f"Unsupported runtime for managed chat template: {runtime}")
    container_root = PurePosixPath(container_model_path)
    if not container_root.is_absolute() or "\x00" in container_model_path:
        raise ValueError("Managed chat template requires an absolute container model path")

    flags_to_replace = {"--chat-template", "--chat-template-file"}
    if runtime == "llama_cpp":
        flags_to_replace.add("--reasoning-format")
    upgraded: list[str] = []
    index = 0
    while index < len(command):
        item = str(command[index])
        if item in flags_to_replace:
            index += 2
            continue
        upgraded.append(item)
        index += 1

    template_path = str(container_root / QWEN_FIXED_CHAT_TEMPLATE_FILENAME)
    upgraded.extend([template_flag, template_path])
    if runtime == "llama_cpp":
        upgraded.extend(["--reasoning-format", "deepseek"])
    return upgraded


def chat_template_text(model_path: Path) -> str:
    """Return the model's chat template, or "" when it ships without one.

    Both runtimes need the tool-call and reasoning parsers named at launch, and
    the right parser follows from the payload the template asks the model to
    emit, so the template is the one place worth reading for it.
    """
    for filename in CHAT_TEMPLATE_FILES:
        candidate = model_path / filename
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
    # Older checkpoints keep the template inside tokenizer_config.json.
    tokenizer_config = model_path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        try:
            payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        template = payload.get("chat_template") if isinstance(payload, dict) else None
        if isinstance(template, str):
            return template
        # Some checkpoints ship a list of named templates.
        if isinstance(template, list):
            return "".join(
                item.get("template", "")
                for item in template
                if isinstance(item, dict)
            )
    return ""


def match_parser(template: str, markers: tuple) -> str | None:
    """First parser whose marker set is fully present in the template.

    Marker tables are ordered most-specific first, since the XML formats are
    supersets of the bare-JSON one.
    """
    for name, required in markers:
        if all(marker in template for marker in required):
            return name
    return None


def default_chat_template_kwargs_flags(spec: DeploymentSpec) -> list[str]:
    """Launch flags carrying the deployment's chat-template defaults.

    Both runtimes spell this the same way and give per-request
    chat_template_kwargs precedence over it, so a caller can still override
    thinking behaviour on a single request.
    """
    if not spec.chat_template_kwargs:
        return []
    return [
        "--default-chat-template-kwargs",
        json.dumps(spec.chat_template_kwargs, sort_keys=True, separators=(",", ":")),
    ]


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

    def resident_model_size(self, model_path: Path) -> int:
        """Return model bytes expected to occupy unified memory at runtime."""
        return self.model_size(model_path)

    def openai_capabilities(self) -> list[str]:
        return ["chat", "completion"]

    def extra_volumes(self, spec: DeploymentSpec) -> dict[str, dict[str, str]]:
        del spec
        return {}

    def environment(self, spec: DeploymentSpec) -> dict[str, str]:
        del spec
        return {}

    def entrypoint(self, spec: DeploymentSpec) -> list[str] | None:
        del spec
        return None

    def security_options(self, spec: DeploymentSpec) -> list[str]:
        del spec
        return []

    def ulimits(self, spec: DeploymentSpec) -> list[dict[str, int | str]]:
        del spec
        return []

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
        resident_model_size = self.resident_model_size(model_path)
        route_name = spec.route_alias or spec.api_model_name
        return {
            "spec": (
                spec.public_dump()
                if isinstance(spec, ResolvedDeploymentSpec)
                else spec.model_dump(mode="json")
            ),
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
            "estimated_memory_bytes": int(resident_model_size * 1.2),
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
