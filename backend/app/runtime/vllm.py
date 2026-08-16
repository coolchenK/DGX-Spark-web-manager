import json

from app.runtime.base import (
    DeploymentSpec,
    RuntimeAdapter,
    require_draft_container_path,
    require_speculative_runtime_method,
)


class VllmAdapter(RuntimeAdapter):
    runtime = "vllm"

    def command(self, spec: DeploymentSpec) -> list[str]:
        model_path = self.container_model_path(spec)
        command = [
            "--model",
            model_path,
            "--served-model-name",
            spec.api_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            str(spec.context_length),
            "--gpu-memory-utilization",
            str(spec.memory_fraction),
            "--max-num-seqs",
            str(spec.max_concurrency),
        ]
        if spec.max_batched_tokens is not None:
            command.extend(["--max-num-batched-tokens", str(spec.max_batched_tokens)])
        if spec.quantization and spec.quantization != "auto":
            command.extend(["--quantization", spec.quantization])
        if spec.trust_remote_code:
            command.append("--trust-remote-code")
        if spec.speculative is not None:
            tuning_fields = (
                spec.speculative.num_steps,
                spec.speculative.eagle_top_k,
                spec.speculative.num_draft_tokens,
            )
            if any(value is not None for value in tuning_fields):
                raise ValueError(
                    "grouped speculative tuning fields are not supported by vLLM"
                )
            runtime_method = require_speculative_runtime_method(spec)
            if runtime_method != spec.speculative.method:
                raise ValueError(
                    "resolved runtime method does not match speculative method"
                )
            payload: dict[str, str | int] = {
                "method": runtime_method,
                "model": require_draft_container_path(spec),
            }
            if spec.speculative.num_speculative_tokens is not None:
                payload["num_speculative_tokens"] = (
                    spec.speculative.num_speculative_tokens
                )
            command.extend(
                [
                    "--speculative-config",
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ]
            )
        return command

