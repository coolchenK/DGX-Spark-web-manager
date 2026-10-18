import json
from pathlib import Path

from app.image_schema import BASE_REVISION, LORA_FILE
from app.runtime.base import DeploymentSpec, RuntimeAdapter


class ViggleAdapter(RuntimeAdapter):
    runtime = "viggle"

    def __init__(self, *, runtime_token: str, **kwargs):
        super().__init__(**kwargs)
        self.runtime_token = runtime_token

    def validate(self, spec: DeploymentSpec) -> Path:
        path = super().validate(spec)
        if (
            spec.speculative
            or spec.generation_defaults.model_dump(exclude_none=True)
            or spec.quantization not in {None, "auto"}
        ):
            raise ValueError("Viggle uses image-generation settings from the image studio")
        if spec.max_concurrency != 1 or spec.chat_template or spec.chat_template_kwargs:
            raise ValueError("Viggle requires concurrency 1 and the original image pipeline")
        if (
            path.parent.name != "snapshots"
            or path.parent.parent.name != "models--Viggle--Qwen-Image-2.1-viggle-turbo"
        ):
            raise ValueError("Select the downloaded Viggle adapter snapshot")
        return path

    @staticmethod
    def base_path(model_path: Path) -> Path:
        return model_path.parents[2] / "models--Qwen--Qwen-Image-2.1" / "snapshots" / BASE_REVISION

    def check_model_compatibility(self, model_path: Path) -> dict:
        base = self.base_path(model_path)
        required = [
            model_path / LORA_FILE,
            model_path / "scheduler/scheduler_config.json",
            base / "model_index.json",
        ]
        missing = [str(p) for p in required if not p.is_file()]
        for sub in ("transformer", "text_encoder", "vae"):
            if not list((base / sub).glob("*.safetensors")):
                missing.append(f"base/{sub} weights")
            for index in (base / sub).glob("*.safetensors.index.json"):
                try:
                    shards = set(json.loads(index.read_text())["weight_map"].values())
                    missing.extend(
                        f"base/{sub}/{name}"
                        for name in shards
                        if not (index.parent / name).is_file()
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    missing.append(f"base/{sub} valid weight index")
        return {
            "compatible": not missing,
            "architectures": ["QwenImage21Pipeline"],
            "reasons": [f"Missing {name}" for name in missing],
        }

    def resident_model_size(self, model_path: Path) -> int:
        return self.model_size(model_path) + self.model_size(self.base_path(model_path))

    def openai_capabilities(self) -> list[str]:
        return ["image_generation", "image_editing", "image"]

    def entrypoint(self, spec: DeploymentSpec) -> list[str]:
        return ["python3", "/opt/viggle/server.py"]

    def environment(self, spec: DeploymentSpec) -> dict[str, str]:
        cache = str(Path(self.container_model_path(spec)).parents[2])
        return {"HF_HUB_CACHE": cache, "VIGGLE_API_KEY": self.runtime_token}

    def command(self, spec: DeploymentSpec) -> list[str]:
        return [
            "--model-path",
            self.container_model_path(spec),
            "--served-model-name",
            spec.api_model_name,
            "--port",
            "8000",
        ]

    def preview(self, spec: DeploymentSpec) -> dict:
        result = super().preview(spec)
        result["api_example"] = "在「图片生成」页面输入提示词，或上传 1–3 张参考图进行编辑。"
        result["warnings"] = [
            "Large image generations may temporarily need 60–70 GiB unified memory"
        ]
        return result
