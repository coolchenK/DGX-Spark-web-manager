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
- Validated SGLang and vLLM deployment adapters with image and argument allowlists.
- Deployment preview, edit, clone, health-gated replacement, automatic rollback, and retained task history.
- Context, concurrency, batch-token, memory, quantization, route alias, and remote-code controls.
- OpenAI-compatible `/v1/models`, chat completions, completions, embeddings (when supported), and SSE streaming.
- Hashed gateway API keys, encrypted provider/Hugging Face secrets, administrator sessions, CSRF protection, and audit history.
- Third-party OpenAI-compatible providers for diagnosis. AI plans require human approval and can only invoke enumerated operations.
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

Do not expose port 3000 to the public internet without TLS and an authenticated reverse proxy. Set `DGX_COOKIE_SECURE=true` when the panel is served over HTTPS.

## Existing Services

On startup the manager scans Docker without restarting or recreating existing containers. Discovered SGLang/vLLM services are registered as unmanaged deployments. They can be observed and controlled, but the manager refuses to delete them. Only containers created by this project carry the `com.dgx-spark-manager.managed=true` label and can be removed from the panel.

## OpenAI API

Create a key under **API 网关**, then point the official SDK to the manager:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://dgx-spark.local:3000/v1",
    api_key="dgx_your_key",
)

for chunk in client.chat.completions.create(
    model="qwen3.8-27b",
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
- AI responses never become shell commands. The executor accepts only start, stop, restart, and inventory rescan operations.
- Authorization headers, API keys, and Hugging Face tokens are redacted from diagnostic context and logs.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Compatibility](docs/COMPATIBILITY.md)
- [Product context](PRODUCT.md)
- [Visual system](DESIGN.md)

## License

MIT
