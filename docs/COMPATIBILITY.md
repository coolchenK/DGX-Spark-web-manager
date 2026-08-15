# DGX Spark Compatibility

## Verified Reference System

The manager is installed and tested natively on the following DGX Spark class system:

| Component | Verified value |
| --- | --- |
| Architecture | `aarch64` |
| Operating system | Ubuntu 24.04.4 LTS |
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.173.02 |
| CUDA reported by driver | 13.0 |
| Docker Engine | 29.2.1 |
| Docker Compose | 5.0.2 |
| SGLang discovery | Existing service discovered and health-probed |
| vLLM discovery | Existing v0.27.1 service discovered and health-probed |

The manager image uses multi-architecture Debian, Node.js, and Python base images and is built
directly on the Spark. No x86 emulation is required.

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
