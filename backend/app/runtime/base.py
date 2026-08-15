from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DeploymentSpec(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    model_id: str | None = None
    model_path: str
    api_model_name: str = Field(min_length=1, max_length=255)
    runtime: str
    image: str
    port: int = Field(ge=1024, le=65535)
    context_length: int = Field(default=32768, ge=1024, le=1_048_576)
    memory_fraction: float = Field(default=0.8, ge=0.05, le=0.98)
    max_concurrency: int = Field(default=8, ge=1, le=1024)
    trust_remote_code: bool = False

    @field_validator("api_model_name")
    @classmethod
    def validate_api_model_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
            raise ValueError("API model name contains unsupported characters")
        return value


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

    def preview(self, spec: DeploymentSpec) -> dict[str, Any]:
        model_path = self.validate(spec)
        return {
            "container_name": deterministic_container_name(spec.name),
            "runtime": self.runtime,
            "image": spec.image,
            "model_path": str(model_path),
            "container_model_path": self.container_model_path(spec),
            "port": spec.port,
            "command": self.command(spec),
            "estimated_memory_fraction": spec.memory_fraction,
            "rollback": "Remove the newly created manager-owned container and retain model files.",
        }

    @abstractmethod
    def command(self, spec: DeploymentSpec) -> list[str]:
        raise NotImplementedError

