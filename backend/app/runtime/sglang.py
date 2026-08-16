from app.runtime.base import (
    DeploymentSpec,
    RuntimeAdapter,
    require_draft_container_path,
    require_speculative_runtime_method,
)

SGLANG_SPECULATIVE_METHODS = {
    "draft_model": "STANDALONE",
    "eagle": "EAGLE",
    "eagle3": "EAGLE3",
    "mtp": "NEXTN",
}


class SGLangAdapter(RuntimeAdapter):
    runtime = "sglang"

    def command(self, spec: DeploymentSpec) -> list[str]:
        model_path = self.container_model_path(spec)
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--served-model-name",
            spec.api_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--context-length",
            str(spec.context_length),
            "--mem-fraction-static",
            str(spec.memory_fraction),
            "--max-running-requests",
            str(spec.max_concurrency),
        ]
        if spec.quantization and spec.quantization != "auto":
            command.extend(["--quantization", spec.quantization])
        if spec.trust_remote_code:
            command.append("--trust-remote-code")
        if spec.speculative is not None:
            runtime_method = require_speculative_runtime_method(spec)
            expected_method = SGLANG_SPECULATIVE_METHODS[spec.speculative.method]
            if runtime_method != expected_method:
                raise ValueError(
                    "resolved runtime method does not match speculative method"
                )
            if spec.speculative.num_speculative_tokens is not None:
                raise ValueError(
                    "num_speculative_tokens is not supported by SGLang"
                )
            command.extend(
                [
                    "--speculative-algorithm",
                    runtime_method,
                    "--speculative-draft-model-path",
                    require_draft_container_path(spec),
                ]
            )
            if spec.speculative.num_steps is not None:
                command.extend(
                    [
                        "--speculative-num-steps",
                        str(spec.speculative.num_steps),
                        "--speculative-eagle-topk",
                        str(spec.speculative.eagle_top_k),
                        "--speculative-num-draft-tokens",
                        str(spec.speculative.num_draft_tokens),
                    ]
                )
        return command

