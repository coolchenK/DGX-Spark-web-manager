import json
import struct
from pathlib import Path, PurePosixPath

from app.runtime.base import (
    DeploymentSpec,
    ResolvedDeploymentSpec,
    RuntimeAdapter,
    container_chat_template_path,
    default_chat_template_kwargs_flags,
    effective_chat_template,
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

AEON_MIXED_SPARK_IMAGE = "ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-07-reasoning-eos"
AEON_MIXED_SPARK_OMNI_IMAGE = "ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-11-v0.29.0-omni"
AEON_MIXED_SPARK_DFLASH_FIXED_IMAGE = "dgx-local/aeon-vllm-ultimate:2026-09-11-dflash2-fix"
AEON_MIXED_SPARK_0918_IMAGE = "ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-18-v0.29.0-omni"
AEON_MIXED_SPARK_0918_FIXED_IMAGE = "dgx-local/aeon-vllm-ultimate:2026-09-18-dflash2-fix"
AEON_MIXED_SPARK_0918_IMAGES = {AEON_MIXED_SPARK_0918_IMAGE, AEON_MIXED_SPARK_0918_FIXED_IMAGE}
AEON_MIXED_SPARK_OMNI_IMAGES = {
    AEON_MIXED_SPARK_OMNI_IMAGE, AEON_MIXED_SPARK_DFLASH_FIXED_IMAGE,
    *AEON_MIXED_SPARK_0918_IMAGES,
}
AEON_MIXED_SPARK_IMAGES = {AEON_MIXED_SPARK_IMAGE, *AEON_MIXED_SPARK_OMNI_IMAGES}
AEON_MIXED_SPARK_DFLASH_BATCH_TOKENS = [
    [1, 1, 10],
    [2, 2, 10],
    [3, 4, 8],
    [5, 8, 7],
    [9, 10, 6],
    [11, 12, 5],
    [13, 14, 4],
    [15, 16, 3],
]

ORCA_FLASH_REPOSITORY = "models--orcarouter--Qwen3.8-Flash-Next-Uncensored-NVFP4"
ORCA_FLASH_REVISION = "cddc6ec5a218d26a81c57353dead52839c6ab2b9"
ORCA_FLASH_PLE_BYTES = 102_400_491_520
ORCA_FLASH_KV_BYTES = 8 * 1024**3
ORCA_FLASH_SPARK_IMAGE = "dgx-local/orcarouter-flash-next:v030-spark-20261010"

LONG_CONTEXT_FP8_KV_THRESHOLD = 131_072


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
    return quantization.get("quant_method") == "compressed-tensors" and "nvfp4" in quantization_text


def _orca_mmap_checkpoint(path: Path) -> bool:
    return (path.name == ORCA_FLASH_REVISION and path.parent.name == "snapshots"
            and path.parent.parent.name == ORCA_FLASH_REPOSITORY)


def _orca_ple_bytes(path: Path) -> int:
    shard = path / "model-00002-of-00017.safetensors"
    with shard.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 256 * 1024:
            raise ValueError("Invalid Orca BF16 PLE header size")
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    if set(header) != {f"{prefix}{i}.weight" for i in range(128)}:
        raise ValueError("Orca BF16 PLE tensors are missing or unexpected")
    ranges = []
    for value in header.values():
        begin, end = value["data_offsets"]
        if (value["dtype"] != "BF16" or value["shape"] != [2500012, 160]
                or begin < 0 or end - begin != 2500012 * 160 * 2):
            raise ValueError("Orca BF16 PLE tensor layout is invalid")
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise ValueError("Orca BF16 PLE tensor ranges are not contiguous")
        cursor = end
    if cursor != ORCA_FLASH_PLE_BYTES or shard.stat().st_size != 8 + size + cursor:
        raise ValueError("Orca BF16 PLE shard is incomplete")
    return cursor


class VllmAdapter(RuntimeAdapter):
    runtime = "vllm"

    def validate(self, spec: DeploymentSpec) -> Path:
        path = super().validate(spec)
        if _orca_mmap_checkpoint(path) or spec.image == ORCA_FLASH_SPARK_IMAGE:
            if not _orca_mmap_checkpoint(path) or spec.image != ORCA_FLASH_SPARK_IMAGE:
                raise ValueError("This Orca checkpoint requires its pinned Spark mmap runtime")
            _orca_ple_bytes(path)
        return path

    def resident_model_size(self, model_path: Path) -> int:
        total = self.model_size(model_path)
        return total - _orca_ple_bytes(model_path) if _orca_mmap_checkpoint(model_path) else total

    def kv_cache_budget(self, spec: DeploymentSpec) -> int | None:
        return ORCA_FLASH_KV_BYTES if spec.image == ORCA_FLASH_SPARK_IMAGE else None

    def environment(self, spec: DeploymentSpec) -> dict[str, str]:
        model_path = self.validate(spec)
        environment = {}
        if spec.image == ORCA_FLASH_SPARK_IMAGE:
            environment.update({"VLLM_USE_DEEP_GEMM": "0", "HF_HUB_OFFLINE": "1",
                                "TRANSFORMERS_OFFLINE": "1"})
        if spec.image in AEON_MIXED_SPARK_IMAGES:
            environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        if spec.image in AEON_MIXED_SPARK_OMNI_IMAGES:
            environment.update(
                {"VLLM_ENABLE_CUDA_COMPATIBILITY": "0", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
            )
        if not _is_qwen35_nvfp4(model_path):
            return environment
        return {
            **environment,
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_TRITON_FP8_GEMM": "1",
        }

    def extra_volumes(self, spec: DeploymentSpec) -> dict[str, dict[str, str]]:
        model_path = self.validate(spec)
        if spec.image in AEON_MIXED_SPARK_0918_IMAGES:
            return {}  # ModelOpt and DFlash2 fixes are built into this dated image.
        if spec.image in AEON_MIXED_SPARK_IMAGES:
            if not isinstance(spec, ResolvedDeploymentSpec):
                raise ValueError("Resolved deployment spec is required for AEON ModelOpt patch")
            if not spec.base_model_root or not spec.base_host_model_root:
                raise ValueError("Resolved model roots are required for AEON ModelOpt patch")
            patch = (model_path / "modelopt-54367.py").resolve()
            if not patch.is_file():
                raise ValueError("AEON MIXED requires modelopt-54367.py from its model repository")
            relative = patch.relative_to(Path(spec.base_model_root).resolve())
            source = PurePosixPath(spec.base_host_model_root).joinpath(*relative.parts)
            return {
                str(source): {
                    "bind": (
                        "/usr/local/lib/python3.12/site-packages/vllm/"
                        "model_executor/layers/quantization/modelopt.py"
                    ),
                    "mode": "ro",
                }
            }
        return {}

    def entrypoint(self, spec: DeploymentSpec) -> list[str] | None:
        return [] if spec.image in AEON_MIXED_SPARK_IMAGES else None

    def command(self, spec: DeploymentSpec) -> list[str]:
        validated_model_path = self.validate(spec)
        model_path = self.container_model_path(spec)
        is_nemotron_h = _is_nemotron_h(validated_model_path)
        is_qwen35_nvfp4 = _is_qwen35_nvfp4(validated_model_path)
        is_aeon_mixed = spec.image in AEON_MIXED_SPARK_IMAGES
        is_aeon_omni = spec.image in AEON_MIXED_SPARK_OMNI_IMAGES
        is_aeon_0918 = spec.image in AEON_MIXED_SPARK_0918_IMAGES
        if is_aeon_mixed and spec.quantization not in {None, "auto"}:
            raise ValueError("AEON MIXED must auto-detect quantization from hf_quant_config.json")
        command = [
            *(["vllm", "serve", model_path] if is_aeon_mixed else ["--model", model_path]),
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
        if spec.image == ORCA_FLASH_SPARK_IMAGE:
            command.extend(
                [
                    "--tensor-parallel-size",
                    "1",
                    "--engram-config",
                    '{"cpu_offload":true}',
                    "--kv-cache-memory-bytes",
                    str(ORCA_FLASH_KV_BYTES),
                    "--kv-cache-dtype",
                    "auto",
                    "--enable-prefix-caching",
                    "--enable-prompt-tokens-details",
                    "--enable-chunked-prefill",
                    "--mamba-cache-mode",
                    "align",
                    "--no-enable-flashinfer-autotune",
                    "--compilation-config",
                    '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,8,12,16,20]}',
                ]
            )
        if is_aeon_mixed:
            command.extend(
                [
                    "--kv-cache-dtype",
                    "fp8",
                    "--enable-chunked-prefill",
                    "--enable-prefix-caching" if is_aeon_0918 else "--no-enable-prefix-caching",
                    "--attention-backend",
                    "TRITON_ATTN",
                    "--limit-mm-per-prompt",
                    '{"image":4,"video":2}',
                    "--override-generation-config",
                    json.dumps(spec.generation_defaults.model_dump(exclude_none=True)),
                ]
            )
        if is_aeon_omni:
            command.extend(["--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'])
        if is_aeon_0918:
            command.append("--enable-prompt-tokens-details")
        if is_qwen35_nvfp4:
            command.extend(["--generation-config", "auto"])
            if spec.context_length >= LONG_CONTEXT_FP8_KV_THRESHOLD:
                command.extend(["--kv-cache-dtype", "fp8"])
        if is_nemotron_h:
            command.extend(NEMOTRON_H_FLAGS)
        template, template_path = effective_chat_template(spec, validated_model_path)
        if template_path is not None:
            command.extend(
                ["--chat-template", container_chat_template_path(model_path, template_path)]
            )
        tool_call_parser = (
            "qwen3_coder"
            if is_nemotron_h or is_aeon_mixed or spec.image == ORCA_FLASH_SPARK_IMAGE
            else match_parser(template, TOOL_CALL_PARSER_MARKERS)
        )
        if tool_call_parser:
            command.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser])
        reasoning_parser = (
            "qwen3"
            if is_aeon_mixed or spec.image == ORCA_FLASH_SPARK_IMAGE
            else "nemotron_v3"
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
                raise ValueError("grouped speculative tuning fields are not supported by vLLM")
            runtime_method = require_speculative_runtime_method(spec)
            if runtime_method != spec.speculative.method:
                raise ValueError("resolved runtime method does not match speculative method")
            payload: dict[str, str | int] = {"method": runtime_method}
            if spec.speculative.draft_model_id is not None:
                payload["model"] = require_draft_container_path(spec)
            if spec.speculative.num_speculative_tokens is not None:
                payload["num_speculative_tokens"] = spec.speculative.num_speculative_tokens
            if is_aeon_omni and runtime_method == "dflash":
                required_tokens = 9 if is_aeon_0918 else 10
                draft_tokens = spec.speculative.num_speculative_tokens
                if is_aeon_0918 and draft_tokens not in {7, 9}:
                    raise ValueError("AEON September 18 DFlash supports 7 or 9 speculative tokens")
                if not is_aeon_0918 and draft_tokens != required_tokens:
                    raise ValueError(
                        f"AEON Omni DFlash recipe requires num_speculative_tokens={required_tokens}"
                    )
                if spec.max_concurrency > 16:
                    raise ValueError(
                        "AEON Omni DFlash recipe supports at most 16 concurrent requests"
                    )
                payload.update(
                    {
                        "attention_backend": "TRITON_ATTN",
                        "num_speculative_tokens_per_batch_size": (
                            [[start, end, min(tokens, draft_tokens)] for start, end, tokens in
                             [[1, 2, 9], [3, 4, 8], [5, 8, 7], [9, 12, 6], [13, 16, 5]]]
                            if is_aeon_0918 else AEON_MIXED_SPARK_DFLASH_BATCH_TOKENS
                        ),
                    }
                )
            command.extend(
                [
                    "--speculative-config",
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ]
            )
        return command
