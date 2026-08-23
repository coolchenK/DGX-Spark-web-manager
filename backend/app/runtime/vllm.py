import json

from app.runtime.base import (
    DeploymentSpec,
    RuntimeAdapter,
    chat_template_text,
    default_chat_template_kwargs_flags,
    match_parser,
    require_draft_container_path,
    require_speculative_runtime_method,
)

# vLLM rejects tool_choice="auto" outright unless both --enable-auto-tool-choice
# and --tool-call-parser are set, so a model whose chat template emits tool
# calls has to declare its parser at launch. Names are vLLM's own: it calls the
# <function=>/<parameter=> XML dialect qwen3_xml (qwen3_coder is an alias for
# the same parser), against SGLang's qwen3_coder. Ordering matters, since the
# XML markers are a superset of the bare-JSON ones.
TOOL_CALL_PARSER_MARKERS = (
    # Onyx ATEM function-call blocks used by Muse Glimmer.
    ("muse_glimmer", ("<atem:function_calls>", "<atem:invoke name=")),
    # <tool_call><function=name><parameter=key>value</parameter></function>
    ("qwen3_xml", ("<tool_call>", "<function=", "<parameter=")),
    # <function name="fn"><param name="k">v</param></function> -- MiniCPM5's
    # own XML dialect. It shares no marker with the Qwen one, and its template
    # never emits <tool_call>, so ordering against hermes does not matter; it
    # sits above anyway to keep the table most-specific first.
    ("minicpm5", ("<function name=", "<param name=")),
    # bare JSON inside <tool_call> ... </tool_call>
    ("hermes", ("<tool_call>",)),
)

REASONING_PARSER_MARKERS = (
    ("muse_glimmer", ("<|start|>assistant to=self<|message|>",)),
    ("qwen3", ("<think>",)),
)

NEMOTRON_H_FLAGS = (
    "--moe-backend",
    "marlin",
    "--kv-cache-dtype",
    "fp8",
    "--enable-prefix-caching",
    "--mamba-backend",
    "flashinfer",
    "--mamba-cache-mode",
    "align",
    "--mamba-ssm-cache-dtype",
    "float16",
)


def _is_nemotron_h(model_path) -> bool:
    """Detect NVIDIA Nemotron-H checkpoints from bounded local metadata."""
    config = model_path / "config.json"
    if not config.is_file():
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    architectures = payload.get("architectures") or []
    return payload.get("model_type") == "nemotron_h" or any(
        str(name).lower().startswith("nemotronh") for name in architectures
    )


def _is_qwen35_nvfp4(model_path) -> bool:
    """Detect the Qwen3.5 compressed-tensors NVFP4 builds that need the
    Blackwell launch settings documented by their model cards."""
    config = model_path / "config.json"
    if not config.is_file():
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    architectures = payload.get("architectures") or []
    is_qwen35 = payload.get("model_type") == "qwen3_5" or any(
        str(name).lower().startswith("qwen3_5") for name in architectures
    )
    quantization = payload.get("quantization_config") or {}
    if not is_qwen35 or not isinstance(quantization, dict):
        return False
    quantization_text = json.dumps(quantization, sort_keys=True).lower()
    return (
        quantization.get("quant_method") == "compressed-tensors"
        and "nvfp4" in quantization_text
    )


class VllmAdapter(RuntimeAdapter):
    runtime = "vllm"

    def environment(self, spec: DeploymentSpec) -> dict[str, str]:
        model_path = self.validate(spec)
        if not _is_qwen35_nvfp4(model_path):
            return {}
        return {
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_TRITON_FP8_GEMM": "1",
        }

    def command(self, spec: DeploymentSpec) -> list[str]:
        validated_model_path = self.validate(spec)
        model_path = self.container_model_path(spec)
        is_nemotron_h = _is_nemotron_h(validated_model_path)
        is_qwen35_nvfp4 = _is_qwen35_nvfp4(validated_model_path)
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
        if is_qwen35_nvfp4:
            command.extend(["--generation-config", "auto"])
        if is_nemotron_h:
            command.extend(NEMOTRON_H_FLAGS)
        template = chat_template_text(validated_model_path)
        tool_call_parser = (
            "qwen3_coder"
            if is_nemotron_h
            else match_parser(template, TOOL_CALL_PARSER_MARKERS)
        )
        if tool_call_parser:
            command.extend(
                ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
            )
        reasoning_parser = (
            "nemotron_v3"
            if is_nemotron_h
            else match_parser(template, REASONING_PARSER_MARKERS)
        )
        if reasoning_parser:
            command.extend(["--reasoning-parser", reasoning_parser])
        if spec.quantization and spec.quantization != "auto":
            command.extend(["--quantization", spec.quantization])
        command.extend(default_chat_template_kwargs_flags(spec))
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
            payload: dict[str, str | int] = {"method": runtime_method}
            if spec.speculative.draft_model_id is not None:
                payload["model"] = require_draft_container_path(spec)
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

