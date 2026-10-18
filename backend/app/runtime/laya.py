from pathlib import Path

from app.runtime.base import DeploymentSpec, RuntimeAdapter


class LayaAdapter(RuntimeAdapter):
    runtime = "laya"

    def validate(self, spec: DeploymentSpec) -> Path:
        path = super().validate(spec)
        if (
            spec.speculative
            or spec.generation_defaults.model_dump(exclude_none=True)
            or spec.chat_template
            or spec.chat_template_kwargs
            or spec.quantization not in {None, "auto"}
        ):
            raise ValueError(
                "Laya uses typed decisions, without generation or speculative settings"
            )
        if spec.context_length != 1024 or spec.max_concurrency != 1:
            raise ValueError(
                "Laya Router uses checkpoint defaults (512/1024 tokens), concurrency 1"
            )
        return path

    def check_model_compatibility(self, model_path: Path) -> dict:
        required = [
            model_path / sub / name
            for sub in ("", "multilingual", "typed-decisions")
            for name in (
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            )
        ]
        missing = [str(p.relative_to(model_path)) for p in required if not p.is_file()]
        return {
            "compatible": not missing,
            "architectures": ["LayaDecisionModel"],
            "reasons": [f"Missing {name}" for name in missing],
        }

    def resident_model_size(self, model_path: Path) -> int:
        # SDK holds FP32 parameters, with BF16 autocast for the forward pass.
        return 2 * sum(
            (model_path / sub / "model.safetensors").stat().st_size
            for sub in ("", "multilingual", "typed-decisions")
        )

    def openai_capabilities(self) -> list[str]:
        return ["decision"]

    def entrypoint(self, spec: DeploymentSpec) -> list[str]:
        return ["python3", "/opt/laya/server.py"]

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
        result["api_example"] = (
            'POST /v1/decisions: {"model":"'
            + (spec.route_alias or spec.api_model_name)
            + '","state":"Please refund me.","questions":{"refund":'
            + '{"type":"noul","instructions":"Is a refund requested?"}}}'
        )
        return result
