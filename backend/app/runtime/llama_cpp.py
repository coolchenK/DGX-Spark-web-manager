from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from app.runtime.base import (
    DeploymentSpec,
    LlamaCppConfig,
    RuntimeAdapter,
    default_chat_template_kwargs_flags,
)

DEFAULT_CONTAINER_RUNTIME_DIR = PurePosixPath("/opt/llamacpp")


class LlamaCppAdapter(RuntimeAdapter):
    runtime = "llama_cpp"

    def __init__(
        self,
        *,
        allowed_images: set[str],
        model_roots: tuple[Path, ...],
        host_runtime_dir: Path | str,
        manager_runtime_dir: Path,
        container_runtime_dir: PurePosixPath = DEFAULT_CONTAINER_RUNTIME_DIR,
    ):
        super().__init__(allowed_images=allowed_images, model_roots=model_roots)
        self.host_runtime_dir = str(host_runtime_dir).replace("\\", "/")
        self.manager_runtime_dir = manager_runtime_dir
        self.container_runtime_dir = container_runtime_dir

    @staticmethod
    def _contained_file(model_path: Path, filename: str) -> Path:
        candidate = model_path / filename
        resolved_root = model_path.resolve()
        resolved = candidate.resolve()
        if (
            not candidate.is_file()
            or not resolved.is_relative_to(resolved_root)
            or candidate.name != filename
        ):
            raise ValueError(f"GGUF file is unavailable: {filename}")
        return candidate

    @staticmethod
    def _config(spec: DeploymentSpec) -> LlamaCppConfig:
        return spec.llama_cpp or LlamaCppConfig()

    def _select_files(self, spec: DeploymentSpec, model_path: Path) -> tuple[Path, Path | None]:
        config = self._config(spec)
        candidates = sorted(
            (
                path
                for path in model_path.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".gguf"
                and not path.name.lower().startswith("mmproj")
            ),
            key=lambda path: path.name.lower(),
        )
        if config.model_file is not None:
            model_file = self._contained_file(model_path, config.model_file)
            if model_file.name.lower().startswith("mmproj"):
                raise ValueError("The llama.cpp model file cannot be an mmproj file")
        elif len(candidates) == 1:
            model_file = self._contained_file(model_path, candidates[0].name)
        elif not candidates:
            raise ValueError("No primary GGUF model file was found")
        else:
            raise ValueError("Multiple GGUF model files require an explicit model_file")

        mmproj_file = None
        if config.mmproj_file is not None:
            mmproj_file = self._contained_file(model_path, config.mmproj_file)
            if not mmproj_file.name.lower().startswith("mmproj"):
                raise ValueError("The llama.cpp projector must be an mmproj GGUF file")
        else:
            preferred = model_path / "mmproj-F16.gguf"
            if preferred.is_file():
                mmproj_file = self._contained_file(model_path, preferred.name)
        return model_file, mmproj_file

    def _validate_runtime(self) -> None:
        binary = self.manager_runtime_dir / "llama-server"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("The configured llama-server binary is unavailable")
        library_dir = self.manager_runtime_dir / "lib"
        if not library_dir.is_dir():
            raise ValueError("The configured llama.cpp library directory is unavailable")
        if not PurePosixPath(self.host_runtime_dir).is_absolute():
            raise ValueError("The llama.cpp host runtime directory must be absolute")

    def validate(self, spec: DeploymentSpec) -> Path:
        model_path = super().validate(spec)
        self._validate_runtime()
        self._select_files(spec, model_path)
        return model_path

    def check_model_compatibility(self, model_path: Path) -> dict[str, Any]:
        gguf_files = [
            path
            for path in model_path.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".gguf"
            and not path.name.lower().startswith("mmproj")
        ]
        reasons = [] if gguf_files else ["No primary GGUF model file was found"]
        return {
            "compatible": not reasons,
            "architectures": [],
            "weight_files": len(gguf_files),
            "reasons": reasons,
        }

    def extra_volumes(self, spec: DeploymentSpec) -> dict[str, dict[str, str]]:
        self.validate(spec)
        return {
            self.host_runtime_dir: {
                "bind": str(self.container_runtime_dir),
                "mode": "ro",
            }
        }

    def environment(self, spec: DeploymentSpec) -> dict[str, str]:
        self.validate(spec)
        return {"LD_LIBRARY_PATH": f"{self.container_runtime_dir}/lib"}

    def command(self, spec: DeploymentSpec) -> list[str]:
        model_path = self.validate(spec)
        model_file, mmproj_file = self._select_files(spec, model_path)
        container_model_root = PurePosixPath(self.container_model_path(spec))
        container_model = str(container_model_root / model_file.name)
        config = self._config(spec)
        command = [
            str(self.container_runtime_dir / "llama-server"),
            "--model",
            container_model,
            "--alias",
            spec.api_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--ctx-size",
            str(spec.context_length),
            "--gpu-layers",
            str(config.gpu_layers),
            "--parallel",
            str(spec.max_concurrency),
        ]
        if mmproj_file is not None:
            command.extend(["--mmproj", str(container_model_root / mmproj_file.name)])
        command.append("--jinja" if config.jinja else "--no-jinja")
        command.append(
            "--cont-batching" if config.continuous_batching else "--no-cont-batching"
        )
        if config.mtp_enabled:
            command.extend(
                [
                    "--spec-type",
                    "draft-mtp",
                    "--spec-draft-model",
                    container_model,
                    "--spec-draft-n-max",
                    str(config.mtp_tokens),
                ]
            )

        # llama-server takes the same JSON object but under a shorter flag
        # name, and only when the jinja engine is on -- with --no-jinja there is
        # no template to hand the kwargs to.
        template_kwargs = default_chat_template_kwargs_flags(spec)
        if template_kwargs and config.jinja:
            command.extend(["--chat-template-kwargs", template_kwargs[1]])

        generation_flags = {
            "temperature": "--temp",
            "top_p": "--top-p",
            "top_k": "--top-k",
            "min_p": "--min-p",
            "repetition_penalty": "--repeat-penalty",
            "presence_penalty": "--presence-penalty",
            "frequency_penalty": "--frequency-penalty",
        }
        for field, flag in generation_flags.items():
            value = getattr(spec.generation_defaults, field)
            if value is not None:
                command.extend([flag, str(value)])
        return command
