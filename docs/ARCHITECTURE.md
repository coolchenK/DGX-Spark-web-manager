# Architecture

## Runtime Topology

The manager is one multi-stage ARM64 container. Node builds the React SPA, then the Python image runs FastAPI and serves both the API and compiled assets. Docker Compose uses host networking so the gateway can reach existing inference endpoints at `127.0.0.1:<port>` without rewriting container routes.

Mounted resources:

- `/var/run/docker.sock`: Docker discovery and validated lifecycle actions.
- `/hf-cache`: the host Hugging Face cache.
- `/models`: the host model directory.
- `/app/data`: SQLite, task state, audit records, and encrypted secrets.
- `/host/etc/os-release`: host operating-system identity.

The manager container does not replace or wrap inference containers. Runtime adapters create normal NVIDIA containers using the same host GPU and model caches.

## Backend Domains

| Domain | Responsibility |
| --- | --- |
| `auth` / `security` | Administrator session, CSRF, API-key hashing, Fernet encryption |
| `system` | Host/GPU resource collection with explicit unsupported values |
| `discovery` | Idempotent Docker and model-cache inventory |
| `tasks` | Durable single-host queue, progress, pause/resume/cancel, restart recovery |
| `runtime` | SGLang and vLLM command validation and preview |
| `deployments` | Container creation, health wait, lifecycle, logs, rollback |
| `gateway` | OpenAI model routing, JSON/SSE proxying, usage metrics |
| `providers` | Encrypted external OpenAI-compatible API configuration and testing |
| `diagnostics` | Real-context prompt construction and structured plan validation |
| `operations` | Human-approved, enum-only action execution |
| `audit` | Actor, action, resource, outcome, IP, and redacted details |

## State And Recovery

SQLite runs in WAL-compatible single-host mode through SQLAlchemy. Alembic applies schema migrations before Uvicorn starts. Background tasks use:

```text
queued -> running -> succeeded | failed | paused | cancelled
paused -> queued | cancelled
failed | cancelled -> queued
```

An interrupted `running` task returns to `queued` during startup and appends a recovery log entry. Download tasks use the Hugging Face cache, so a resumed task reuses completed blobs.

## Trust Boundaries

The browser can request only typed REST operations. It cannot submit Docker commands or runtime argument arrays. The backend validates model paths, images, ports, context sizes, memory ratios, and runtime-specific flags before creating a task.

AI providers receive a bounded snapshot of real system values and tail logs after secret redaction. Returned JSON is reduced to known fields. Unknown operations remain visible with `executable=false`; the executor never evaluates text or invokes a shell.

## Frontend

React Router provides ten authenticated surfaces. TanStack Query owns server state and polling, Zustand keeps only session/CSRF and theme preferences, and Ant Design supplies accessible product controls. Large pages are code-split. Tables switch to summary lists on narrow viewports, while destructive operations keep explicit confirmation text.
