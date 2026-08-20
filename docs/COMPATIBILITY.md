# DGX Spark Compatibility

## Verified Reference System

The manager is installed and tested natively on the following DGX Spark class system:

| Component | Verified value |
| --- | --- |
| Architecture | `aarch64` |
| Operating system | Ubuntu 24.04.4 LTS |
| Kernel | `6.17.0-1029-nvidia` |
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.173.02 |
| CUDA reported by driver | 13.0 |
| Unified memory | 130,595,860,480 bytes (121.6 GiB) |
| Root filesystem | 982,819,848,192 bytes total; 742,425,849,856 bytes free at audit time |
| Docker Engine | 29.2.1 |
| Docker Compose | 5.0.2 |
| NVIDIA Container Toolkit | 1.20.0; `nvidia` Docker runtime present |
| Host Python | 3.12.3 |
| Host Node.js | Not installed; Docker Compose installation does not require it |
| SGLang | `0.0.0.dev1+gb7252cc6b`, `linux/arm64` service health-probed and called |
| vLLM | 0.27.1, `linux/arm64` service health-probed and called |

The manager image uses multi-architecture Debian, Node.js, and Python base images and is built
directly on the Spark. No x86 emulation is required.

## Runtime Compatibility Matrix

| Runtime | Device evidence | Manager status |
| --- | --- | --- |
| SGLang | Local `sglang-inkling:specforge` image reports `linux/arm64`; Qwen responds through `/v1/models` and chat completions | Supported and enabled |
| vLLM | `vllm/vllm-openai:v0.27.1` reports `linux/arm64`; Nemotron responds through `/v1/models` | Supported and enabled |
| llama.cpp | Native ARM64/CUDA `llama-server` under `/opt/llamacpp`; GGUF, mmproj and model-native MTP are adapter-managed | Supported and enabled |
| TensorRT-LLM | No installed server/image was found and no minimum run was performed | Not enabled in this release |
| Transformers server | No standalone OpenAI-compatible runtime was found | Not enabled in this release |

Only verified adapters are offered in the deployment wizard. Upstream availability alone
is not treated as DGX Spark compatibility evidence.

The current allowlist defaults include `vllm/vllm-openai:v0.27.1`,
`lmsysorg/sglang:qwen38-27b`, `sglang-inkling:specforge`,
`lmsysorg/sglang:dev-cu13-inkling-dspark`, and the CUDA 12.9 development image used with the
server-managed llama.cpp runtime mount. An image must be locally available, ARM64/CUDA compatible,
and present in the relevant configured allowlist before preview.

## Runtime Policy

Managed deployments accept only the images listed in `DGX_ALLOWED_VLLM_IMAGES` and
`DGX_ALLOWED_SGLANG_IMAGES` or `DGX_ALLOWED_LLAMA_CPP_IMAGES`. This is an intentional security and
compatibility boundary. Add a tested ARM64/CUDA image to the appropriate comma-separated setting
before using it in a deployment. The llama.cpp binary mount is controlled only by
`DGX_LLAMA_CPP_HOST_DIR` and `DGX_LLAMA_CPP_MANAGER_DIR`; clients cannot submit host mount paths.

Unmanaged inference containers are read-only imports. The manager can display, probe, and route to
them, but removal is restricted to containers carrying the manager ownership label.

## Capability Probes and Fallback

The manager keys runtime capability results by the local image digest. It runs a temporary container
with networking disabled and no volumes:

- vLLM: `vllm serve --help=speculative_config`
- SGLang: `python3 -m sglang.launch_server --help`
- llama.cpp: conservative manifest after validating the configured native server during preview

Recognized help output determines speculative transport, supported methods, and method mappings.
The probe has bounded logs and a timeout, and its container is removed. If the probe fails, the
manager returns a conservative per-runtime manifest with `source="manifest"` and warnings. The
administrator must review those warnings; preview and deployment still enforce the resulting
capability snapshot and runtime adapter rules.

Quantization choices are restricted to the selected image's returned `quantization_methods`.
Model-card or local values not in that list are clamped to `auto` with a warning. The canonical
`nvfp4` evidence value maps to `modelopt_fp4` when the runtime exposes that mapping. Hugging Face
search sorts first by compatibility level (`recommended`, `compatible`, then `review`) and then by
score within the same level. NVFP4 receives the strongest positive score signal, but a `compatible`
AWQ result can rank above a `review` NVFP4 result. Search rank is not proof that a particular
repository/image combination will pass capability and resource preflight.

## Speculative and Draft Model Support

Both runtimes accept only methods present in the selected image capability snapshot. The common
methods are `draft_model`, `eagle`, `eagle3`, and `mtp`; SGLang additionally supports `dspark` when
its help output or bounded manifest exposes `DSPARK`. Transport and tuning differ:

| Runtime | Transport | Tuning accepted by the adapter |
| --- | --- | --- |
| vLLM | One JSON value passed to `--speculative-config` | Optional `num_speculative_tokens` (1-64); SGLang grouped fields are rejected |
| SGLang | `--speculative-algorithm` and `--speculative-draft-model-path` flags | `num_steps` (1-32), `eagle_top_k` (1-32), and `num_draft_tokens` (1-256), set together or all omitted; `num_speculative_tokens` is rejected; DSpark adds its validated DGX Spark flags |
| llama.cpp | Dedicated GGUF settings | Optional same-model `draft-mtp` with `mtp_tokens` (1-64); external Draft Model settings are rejected |

Method mappings are resolved from the capability snapshot. The SGLang adapter maps `draft_model` to
`STANDALONE`, `dspark` to `DSPARK`, `eagle` to `EAGLE`, `eagle3` to `EAGLE3`, and `mtp` to `NEXTN`;
a missing or mismatched mapping fails preview. DSpark repository IDs are resolved against the
mounted local Hugging Face cache, so offline runtime containers do not redownload the Draft Model.

SGLang also sizes hybrid GDN/Mamba state slots from deployment concurrency, selects
`flashinfer_cutlass` for NVFP4 MoE checkpoints, and detects tool-call/reasoning parsers from local chat
templates. vLLM performs the corresponding parser detection. Deployment-level
`chat_template_kwargs` can set template defaults such as thinking mode and reasoning effort.

Draft candidates must be separate available local assets with readable evidence and paths. Explicit
target pairing, supported method, tokenizer fingerprints for ordinary draft models, and combined
base-plus-Draft memory determine `compatible`, `review`, or `incompatible`. A `review` candidate
requires explicit acknowledgement. An incompatible candidate, a blocked resource estimate, or an
unverifiable final preflight cannot be deployed.

## Model-Card Recommendations

The manager reads bounded deployment flags from model-card shell examples, generation values from
model-card JSON and local `generation_config.json`, architectural limits from `config.json`, and
Draft metadata/tokenizer evidence. When a card contains several hardware recipes, the DGX
Spark/GB10 section wins over H100, GB200, and generic examples. The deterministic order is model
card, local config, runtime default, then device rules and clamps.

For new deployment recommendations, the panel forces the configured, enabled DeepSeek provider to
analyze the bounded model-card/device context, including when deterministic fields are already
complete. AI output is limited to requested allowlisted fields and remains a medium-confidence
suggestion. A reasoning-only response is accepted as validation with no overrides; invalid output
degrades the recommendation to `partial`. AI never bypasses capability, compatibility, memory,
preview, or human-review requirements.

## Model Layout

- Hugging Face cache: mounted at `/hf-cache/hub`
- Local model root: mounted at `/models`
- Hugging Face assets resolve to an active `snapshots/<commit>` directory before deployment
- Paths outside these roots are rejected by deployment validation

## Known Limits

- Embeddings are exposed only when the selected upstream runtime implements the endpoint.
- GPU memory usage can be unavailable on GB10 because `nvidia-smi` reports unified memory as `N/A`.
  The dashboard still shows system unified memory from the host.
- Runtime capability probes can fall back to a conservative manifest. Read the returned warnings and
  perform a preview before deploying a newly added image.
- HTTPS termination is not bundled. Use a trusted reverse proxy and set `DGX_COOKIE_SECURE=true`.
