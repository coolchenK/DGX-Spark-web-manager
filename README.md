# DGX Spark Web Manager

ARM64-native management plane for NVIDIA DGX Spark. It discovers existing SGLang and vLLM containers, manages Hugging Face model downloads, deploys validated inference services, and exposes one OpenAI-compatible gateway.

![Python](https://img.shields.io/badge/Python-3.12-3776ab)
![React](https://img.shields.io/badge/React-19-149eca)
![Ant Design](https://img.shields.io/badge/Ant%20Design-5-1677ff)
![Architecture](https://img.shields.io/badge/architecture-ARM64-657d20)

## Features

- Real DGX Spark CPU, unified memory, disk, GPU, temperature, power, and driver status.
- Read-only discovery of existing Docker inference services and Hugging Face caches.
- Persistent Hugging Face search/download tasks with pause, resume, cancellation, and restart recovery.
- Validated SGLang, vLLM, and llama.cpp deployment adapters with image and argument allowlists.
- Deployment preview, edit, clone, health-gated replacement, automatic rollback, and retained task history.
- Automatic post-deployment warmup and TPS benchmarking with persistent results shown in desktop
  and mobile deployment views; benchmark failures are recorded without rolling back healthy services.
- Automatic host-port allocation starts at `8000`, checks both Manager reservations and Docker
  bindings, reuses the lowest released gap, and keeps the Docker port, endpoint, database row,
  persisted spec, and ownership label consistent. Explicit ports are validated for conflicts.
- Context, concurrency, batch-token, memory, quantization, route alias, and remote-code controls.
- Model-card and device-aware deployment recommendations with per-field source, confidence, and
  warnings; optional bounded AI fallback for unresolved fields.
- Optional compatible/reviewed Draft Model selection with runtime-specific speculative settings and
  combined base-plus-Draft resource checks. SGLang DSpark checkpoints are detected, paired with the
  base model, and launched with the DGX Spark cache, attention, and Mamba settings.
- Model-aware SGLang/vLLM launch flags detect tool-call and reasoning parsers from the chat template;
  optional deployment defaults control template thinking behavior without overriding each request.
- OpenAI-compatible `/v1/models`, chat completions, completions, embeddings (when supported), and SSE streaming.
  Model discovery exposes runtime, endpoint, context length, maximum output tokens, and saved
  generation defaults for each healthy running route; stopped deployments are hidden.
- Hashed gateway API keys, encrypted provider/Hugging Face secrets, administrator sessions, CSRF protection, and audit history.
- DeepSeek is the intended reasoning backend for the AI operations assistant. Local Qwen, Gemma,
  and GGUF services remain diagnosis targets, never the assistant brain. Configured OpenAI-compatible
  providers can supply bounded model-card recommendations and diagnosis; AI operations persist
  sessions and read-only tool results, while exact Shell plans require administrator approval before
  Host Agent execution.
- Responsive Ant Design interface with desktop/mobile layouts and light/dark/system themes.

## DGX Spark Installation

Requirements:

- NVIDIA DGX Spark running Ubuntu/DGX OS on `aarch64`
- Docker Engine and Docker Compose
- NVIDIA Container Toolkit
- Existing user access to `/var/run/docker.sock`
- `openssl` and `curl`

Clone the repository, inspect the planned changes, then apply:

```bash
git clone https://github.com/coolchenK/DGX-Spark-web-manager.git
cd DGX-Spark-web-manager
./scripts/install.sh
./scripts/install.sh --apply
```

The first command is preview-only. `--apply` creates `.env`, generates an administrator password, builds the native ARM64 image, starts the service, and waits for `/api/health`.

Open `http://<DGX-SPARK-IP>:3000`. The generated credentials are printed once and remain in the mode-600 `.env` file on the host.

### Manual Installation

```bash
cp .env.example .env
# Replace every placeholder and set host paths/PUID/PGID/DOCKER_GID.
docker compose build
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/api/health
```

### Native systemd Installation

The manager can run directly from an ARM64 Python virtual environment while continuing to use the
host Docker Engine for model runtimes. Python 3.11+, Node.js 22+, Corepack, Docker, and user systemd
are required. The first command only prints the changes, disk estimate, commands, and rollback path.

```bash
./scripts/install-native.sh
./scripts/install-native.sh --apply

# Upgrade (creates a SQLite backup first)
./scripts/update-native.sh

# Preview removal, then remove the service while retaining data/models
./scripts/uninstall-native.sh
./scripts/uninstall-native.sh --apply
```

Native logs are managed by journald and can be read with
`journalctl --user -u dgx-spark-web-manager.service`.

Do not expose port 3000 to the public internet without TLS and an authenticated reverse proxy. Set
`DGX_COOKIE_SECURE=true` when the panel is served over HTTPS. A host reverse proxy can keep Uvicorn
private by setting `DGX_LISTEN_HOST=127.0.0.1` and moving it to another port with
`DGX_LISTEN_PORT`; the defaults remain `0.0.0.0:3000`.

## Existing Services

On startup the manager scans Docker without restarting or recreating existing containers. Discovered SGLang/vLLM services are registered as unmanaged deployments. They can be observed and controlled, but the manager refuses to delete them. Only containers created by this project carry the `com.dgx-spark-manager.managed=true` label and can be removed from the panel.

## Administrator Workflow

1. In **Settings**, optionally store a Hugging Face token for gated repositories. In **Hugging
   Face**, search for a model and inspect its repository metadata. Results sort first by compatibility
   level (`recommended`, `compatible`, then `review`) and then by score within that level. NVFP4 gets
   the strongest positive score signal, but a `compatible` AWQ result can still rank above a `review`
   NVFP4 result. Ranking remains discovery guidance rather than a deployment guarantee.
2. Start a download and follow its persistent task. Pause, resume, cancellation, and restart recovery
   reuse the Hugging Face cache. After inventory refresh marks the asset available, open
   **Deployments** and create a deployment.
3. Select the base model, vLLM or SGLang runtime, and an allowlisted image. The panel reads model-card
   and local configuration evidence, prefers DGX Spark/GB10 hardware sections when a card contains
   multiple recipes, probes/caches image capabilities by digest, applies unified-memory rules, and
   shows the source, confidence, reason, and warning for each suggested field.
4. The wizard selects the enabled DeepSeek provider by default and forces a bounded AI analysis for
   every new recommendation. The provider receives the model card, runtime capability probe, and
   current DGX Spark resource snapshot; it can only fill allowlisted fields. Review every suggestion
   and edit it as needed; AI output is never applied as an unreviewed deployment action.
5. A compatible Draft Model is selected automatically for a new deployment when the model card
   recommends speculative decoding (DSpark/DFlash cards are recognized even when their metadata is
   incomplete). DFlash uses the card's `num_draft_tokens`; other methods use their runtime-specific
   tuning fields. `review` candidates require explicit acknowledgement; incompatible candidates are
   not deployable. Native SGLang DFlash and DSpark use the `DFLASH` and `DSPARK` algorithms and resolve
   repositories from the mounted local Hugging Face cache, including when the Draft Model comes from
   a different configured model root.
6. Review the normalized spec, generated runtime command, mounts, current capability snapshot,
   memory estimate, provenance, and warnings. Resource warnings and review candidates require
   explicit acknowledgement. Submit only from the current preview; any form change invalidates it.
7. Follow the queued task through runtime health checks, warmup, and the automatic TPS benchmark.
   The result remains visible on the desktop table and mobile deployment record. Benchmark failures
   are recorded for diagnosis without rolling back a healthy deployment. The same panel then provides
   start, stop, restart, edit, clone, rollback-aware replacement, logs, task history, audit history,
   and gateway metrics for managed models. Discovered external containers remain protected from
   manager delete.
8. Leave **主机端口** empty for a new deployment when automatic allocation is preferred. The service
   chooses the lowest free port from `8000`, including ports held by stopped Manager deployments and
   existing Docker containers. Clones also receive a fresh automatically allocated port.
9. Under **API Gateway**, create a gateway key and record it at creation time; only its hash is stored.
   Call `GET /v1/models` to obtain the healthy route names and their effective metadata, then use the
   selected model ID with an OpenAI-compatible client. Saved generation defaults fill only omitted,
   runtime-supported request fields, so callers can override them explicitly.

Third-party AI providers are configured with an OpenAI-compatible base URL, API key, default model,
timeout, and optional headers. Configure the paid DeepSeek endpoint as the operations assistant
provider and test it before use. The test reports connection/model-list readiness separately from a
structured default-model probe. Diagnostic read tools use the local Host Agent automatically. When a
model returns no data, disconnects, stops while thinking, reports a missing model, or becomes slow,
the assistant can inspect Manager inventory, Docker state, GPU/memory, ports, logs, tasks, and gateway
metrics without first asking for information already available to the Manager. A proposed Shell
command remains a separate immutable plan: the panel shows its exact command, working directory,
timeout, impact, and rollback, and execution requires explicit administrator approval.

## OpenAI API

Create a key under **API 网关**, then point the official SDK to the manager:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://dgx-spark.local:3000/v1",
    api_key="dgx_your_key",
)

# Use an ID returned by GET /v1/models. Only running and healthy deployments are listed.
model_id = client.models.list().data[0].id

for chunk in client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

More examples are in [`examples/`](examples/), and the endpoint contract is documented in [`docs/API.md`](docs/API.md).

## Operations

```bash
# Upgrade image and service
./scripts/update.sh

# Create a consistent SQLite + environment backup
./scripts/backup.sh

# Restore a backup
./scripts/restore.sh backups/dgx-manager-YYYYMMDDTHHMMSSZ.tar.gz

# Stop and remove only the manager container
./scripts/uninstall.sh

# Also remove the manager database/audit history (models remain)
./scripts/uninstall.sh --purge
```

## Development

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export DGX_SECRET_KEY='development-secret-key-change-before-production'
export DGX_ADMIN_PASSWORD='Development-password-1234'
pytest backend/tests -q
ruff check backend/app backend/tests
uvicorn app.main:app --app-dir backend --reload --port 3000
```

Frontend:

```bash
cd frontend
corepack enable
pnpm install
pnpm dev
pnpm test
pnpm build
pnpm lint
```

The Vite dev server proxies `/api` and `/v1` to `127.0.0.1:3000`.

## Security Model

- Provider and Hugging Face tokens are encrypted with a key derived from `DGX_SECRET_KEY`.
- Gateway keys are returned once and stored only as SHA-256 hashes.
- Browser mutations require a signed administrator session and matching CSRF token.
- Provider URLs reject credentials, loopback, private, link-local, reserved, and metadata-network addresses.
- Model paths must remain inside configured roots. Deployment images and runtime arguments are allowlisted.
- Provider text is never executed directly. Shell repairs are persisted as exact plans, require
  administrator approval, and are digest-checked before the local Host Agent executes them.
- Authorization headers, API keys, and Hugging Face tokens are redacted from diagnostic context and logs.
- Model cards are untrusted data. Recommendation requests redact credentials and paths, bound all
  context, accept only requested typed fields, and still require administrator review plus server
  preflight.
- `trust_remote_code` executes repository-supplied Python inside the inference container. Leave it
  disabled unless the pinned model revision has been reviewed.
- Gateway keys and provider/Hugging Face tokens are different credentials. Do not place their raw
  values in model cards, AI prompts, deployment names, logs, screenshots, or support transcripts.
  Revoke a gateway key if its one-time value is lost or exposed.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Compatibility](docs/COMPATIBILITY.md)
- [Product context](PRODUCT.md)
- [Visual system](DESIGN.md)

## License

MIT
