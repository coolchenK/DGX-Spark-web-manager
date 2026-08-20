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

SGLANG_SPECULATIVE_METHODS = {
    "draft_model": "STANDALONE",
    "dflash": "DFLASH",
    "dspark": "DSPARK",
    "eagle": "EAGLE",
    "eagle3": "EAGLE3",
    "mtp": "NEXTN",
}

# The rest of the SGLang cookbook's DGX Spark command for
# hw=dgx-spark, quant=nvfp4, spec=dspark, ssmDtype=float32. These are tuning
# specific to that recipe rather than requirements of every GDN model, so they
# stay gated on DSpark; the state-pool sizing below applies to all of them.
SGLANG_DSPARK_FLAGS = (
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--attention-backend",
    "flashinfer",
    "--chunked-prefill-size",
    "2048",
    "--mamba-ssm-dtype",
    "float32",
)

# SGLang only decodes structured `tool_calls` and `reasoning_content` when the
# parsers are named at launch; without them a harness receives the raw
# <tool_call> text instead. The right parser follows from the payload the
# model's chat template asks for, so detect it there rather than making the
# operator supply a name. Ordering matters: qwen3_coder's markers are a
# superset of hermes', so it has to be tried first.
TOOL_CALL_PARSER_MARKERS = (
    # <tool_call><function=name><parameter=key>value</parameter></function>
    ("qwen3_coder", ("<tool_call>", "<function=", "<parameter=")),
    # <function name="fn"><param name="k">v</param></function> -- MiniCPM5's
    # own XML dialect. It shares no marker with the Qwen one, and its template
    # never emits <tool_call>, so ordering against hermes does not matter; it
    # sits above anyway to keep the table most-specific first.
    ("minicpm5", ("<function name=", "<param name=")),
    # bare JSON inside <tool_call> ... </tool_call>
    ("hermes", ("<tool_call>",)),
)

REASONING_PARSER_MARKERS = (("qwen3", ("<think>",)),)

# Hybrid GDN models carry a state pool alongside the KV pool, and left alone
# SGLang splits the two by --mamba-full-memory-ratio. That ratio is awkward to
# get right: the balanced value is (S + D) * token_equiv / L, where token_equiv
# is one state slot expressed in KV tokens and therefore differs per
# checkpoint -- 4698 for Qwen3.8-27B, ~6300 for Ornith-1.5-35B. Left at the
# default it over-provisions the state pool badly (Ornith drew 445 slots to
# serve 16 requests).
#
# --max-mamba-cache-size is the same knob from the other end: pin the slot
# count and every remaining byte goes to KV, with no per-checkpoint constant to
# model. SGLang charges a fixed number of slots per running request --
# _calculate_mamba_ratio starts at 3 and adds 2 for the extra_buffer ping-pong
# under the default overlap schedule -- so pinning concurrency x 5 lets
# max_running_requests through unclamped. extra_buffer is also the only
# strategy DSpark accepts.
MAMBA_STATE_SLOTS_PER_REQUEST = 5
MAMBA_RADIX_CACHE_STRATEGY = "extra_buffer"


def _is_hybrid_gdn(model_path) -> bool:
    """Whether the checkpoint interleaves linear-attention (GDN) layers with
    full attention, which is what gives it a state pool to size."""
    config = model_path / "config.json"
    if not config.is_file():
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    for section in (payload, payload.get("text_config")):
        if not isinstance(section, dict):
            continue
        layer_types = section.get("layer_types")
        if isinstance(layer_types, list) and "linear_attention" in layer_types:
            return True
        if section.get("linear_num_value_heads"):
            return True
    return False


def _is_nvfp4_moe(model_path) -> bool:
    """Whether the checkpoint is an NVFP4-quantized MoE.

    SGLang's `auto` MoE runner resolves to flashinfer_trtllm, which carries no
    NVFP4 MoE kernel and aborts at startup with "Use --moe-runner-backend
    flashinfer_cutlass instead". Detecting the pair here ships that flag with
    the launch command instead of letting the server crash-loop.
    """
    quant_config = model_path / "hf_quant_config.json"
    if not quant_config.is_file():
        return False
    try:
        if "NVFP4" not in quant_config.read_text(encoding="utf-8", errors="replace"):
            return False
    except OSError:
        return False
    config = model_path / "config.json"
    if not config.is_file():
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if any("moe" in str(name).lower() for name in payload.get("architectures") or []):
        return True
    return any(
        isinstance(section, dict) and section.get("num_experts")
        for section in (payload, payload.get("text_config"))
    )


def _hf_repo_id_from_cache_path(path: str) -> str | None:
    """Translate a mounted HF cache path to its repository id.

    '/models/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/<sha>'
    becomes 'RadixArk/Qwen3.8-27B-DSpark' so the SGLang DSPARK loader can
    resolve it offline through the local HF cache.
    """
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part.startswith("models--") and index + 1 < len(parts):
            return part[len("models--"):].replace("--", "/", 1)
    return None


def _hf_cache_root_from_container_path(path: str) -> str | None:
    """Return the mounted Hub cache root containing a models--* directory."""
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            return "/".join(parts[:index]) or "/"
    return None


class SGLangAdapter(RuntimeAdapter):
    runtime = "sglang"

    def environment(self, spec: DeploymentSpec) -> dict[str, str]:
        # Use every host CPU core for tensor/linear-algebra work and model
        # loading. SGLang defaults to a fraction of the cores on ARM hosts.
        env = {
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "OPENBLAS_NUM_THREADS": "20",
            "NUMEXPR_NUM_THREADS": "20",
            "VECLIB_MAXIMUM_THREADS": "20",
        }
        # DSPARK draft checkpoints are resolved through the Hugging Face Hub.
        # Point HF at the mounted cache root so an offline resolution of the
        # repository id finds models--* directories under /models.
        cache_root = "/models"
        if spec.speculative is not None and spec.speculative.method == "dspark":
            draft_path = require_draft_container_path(spec)
            cache_root = _hf_cache_root_from_container_path(draft_path) or cache_root
        env["HF_HOME"] = cache_root
        env["HF_HUB_CACHE"] = cache_root
        return env

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
            "--weight-loader-prefetch-num-threads",
            "20",
        ]
        resolved_model_path = self.validate(spec)
        if _is_hybrid_gdn(resolved_model_path):
            command.extend(
                [
                    "--mamba-radix-cache-strategy",
                    MAMBA_RADIX_CACHE_STRATEGY,
                    "--max-mamba-cache-size",
                    str(spec.max_concurrency * MAMBA_STATE_SLOTS_PER_REQUEST),
                ]
            )
        if _is_nvfp4_moe(resolved_model_path):
            command.extend(["--moe-runner-backend", "flashinfer_cutlass"])
        template = chat_template_text(resolved_model_path)
        tool_call_parser = match_parser(template, TOOL_CALL_PARSER_MARKERS)
        if tool_call_parser:
            command.extend(["--tool-call-parser", tool_call_parser])
        reasoning_parser = match_parser(template, REASONING_PARSER_MARKERS)
        if reasoning_parser:
            command.extend(["--reasoning-parser", reasoning_parser])
        if spec.quantization and spec.quantization != "auto":
            command.extend(["--quantization", spec.quantization])
        command.extend(default_chat_template_kwargs_flags(spec))
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
            if spec.speculative.method == "dspark":
                # DSPARK resolves the draft checkpoint through the HF Hub API.
                # A local cache path is not a valid repo id, so translate the
                # mounted models--* cache path back to its repository id.
                draft_container_path = require_draft_container_path(spec)
                command.extend(
                    [
                        "--speculative-algorithm",
                        runtime_method,
                        "--speculative-draft-model-path",
                        _hf_repo_id_from_cache_path(draft_container_path)
                        or draft_container_path,
                    ]
                )
            else:
                command.extend(
                    [
                        "--speculative-algorithm",
                        runtime_method,
                        "--speculative-draft-model-path",
                        require_draft_container_path(spec),
                    ]
                )
            if spec.speculative.method == "dspark":
                # DSPARK verify window is block_size + 1 (gamma auto-inferred
                # from the draft checkpoint when omitted). The DSpark draft is
                # a BF16 checkpoint, so it must load unquantized instead of
                # inheriting the target model's quantization path.
                command.extend(
                    [
                        "--speculative-draft-attention-backend",
                        "flashinfer",
                        "--speculative-draft-model-quantization",
                        "unquant",
                    ]
                )
            if spec.speculative.method == "dflash":
                if spec.speculative.num_draft_tokens is not None:
                    command.extend(
                        [
                            "--speculative-num-draft-tokens",
                            str(spec.speculative.num_draft_tokens),
                        ]
                    )
            elif spec.speculative.num_steps is not None:
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
        if spec.speculative is not None and spec.speculative.method == "dspark":
            # Remaining tuning from the SGLang cookbook's DGX Spark recipe.
            command.extend(list(SGLANG_DSPARK_FLAGS))
        return command
