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
| llama.cpp | No installed server/image was found and no GB10 minimum run was performed | Not enabled in this release |
| TensorRT-LLM | No installed server/image was found and no minimum run was performed | Not enabled in this release |
| Transformers server | No standalone OpenAI-compatible runtime was found | Not enabled in this release |

Only the two verified adapters are offered in the deployment wizard. Upstream availability alone
is not treated as DGX Spark compatibility evidence.

## Runtime Policy

Managed deployments accept only the images listed in `DGX_ALLOWED_VLLM_IMAGES` and
`DGX_ALLOWED_SGLANG_IMAGES`. This is an intentional security and compatibility boundary. Add a
tested ARM64/CUDA image to the appropriate comma-separated setting before using it in a deployment.

Unmanaged inference containers are read-only imports. The manager can display, probe, and route to
them, but removal is restricted to containers carrying the manager ownership label.

## Model Layout

- Hugging Face cache: mounted at `/hf-cache/hub`
- Local model root: mounted at `/models`
- Hugging Face assets resolve to an active `snapshots/<commit>` directory before deployment
- Paths outside these roots are rejected by deployment validation

## Known Limits

- Embeddings are exposed only when the selected upstream runtime implements the endpoint.
- GPU memory usage can be unavailable on GB10 because `nvidia-smi` reports unified memory as `N/A`.
  The dashboard still shows system unified memory from the host.
- HTTPS termination is not bundled. Use a trusted reverse proxy and set `DGX_COOKIE_SECURE=true`.
