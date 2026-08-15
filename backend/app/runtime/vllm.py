from app.runtime.base import DeploymentSpec, RuntimeAdapter


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
        if spec.trust_remote_code:
            command.append("--trust-remote-code")
        return command

