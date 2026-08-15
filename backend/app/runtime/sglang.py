from app.runtime.base import DeploymentSpec, RuntimeAdapter


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
        if spec.trust_remote_code:
            command.append("--trust-remote-code")
        return command

