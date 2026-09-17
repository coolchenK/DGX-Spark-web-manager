# DGX Spark Web Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy an ARM64-native management plane for DGX Spark models, inference containers, OpenAI-compatible routing, Hugging Face downloads, and approved AI diagnostics.

**Architecture:** A FastAPI service owns persistence, discovery, task execution, adapters, and the OpenAI proxy. A React/Ant Design SPA is compiled into the same multi-arch container and communicates through typed REST/SSE APIs. Runtime operations use Docker and Hugging Face SDK/CLI boundaries; AI plans never execute arbitrary shell.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, Pydantic, httpx, Docker SDK, Hugging Face Hub, React 19, TypeScript, Vite, Ant Design 5, TanStack Query, Zustand, Vitest, Playwright, Docker Compose.

---

### Task 1: Project foundation and contracts

**Files:** `pyproject.toml`, `backend/app/main.py`, `backend/app/config.py`, `backend/app/db.py`, `frontend/package.json`, `frontend/vite.config.ts`, `compose.yaml`, `.env.example`

- [ ] Add backend configuration and health contract tests in `backend/tests/test_health.py`.
- [ ] Run `pytest backend/tests/test_health.py -v` and confirm the missing app failure.
- [ ] Implement the FastAPI app factory, settings validation, SQLite engine and `/api/health`.
- [ ] Scaffold Vite/React/Ant Design and proxy `/api` plus `/v1` during development.
- [ ] Run backend tests, frontend typecheck and an ARM64-compatible dependency audit.

### Task 2: Persistence, authentication and audit

**Files:** `backend/app/models.py`, `backend/app/security.py`, `backend/app/auth.py`, `backend/app/audit.py`, `backend/tests/test_auth.py`, `backend/tests/test_security.py`

- [ ] Write failing tests for login, HttpOnly session, CSRF rejection, API-key hashing, secret encryption and audit redaction.
- [ ] Implement SQLAlchemy entities, session signing, CSRF validation, Argon2 password verification, Fernet secret storage and audit middleware.
- [ ] Add login/logout/me, API-key create/list/revoke and audit-list endpoints.
- [ ] Run targeted tests and Alembic upgrade against a temporary database.

### Task 3: Real system and Docker discovery

**Files:** `backend/app/services/system.py`, `backend/app/services/discovery.py`, `backend/app/runtime/base.py`, `backend/app/runtime/docker.py`, `backend/app/api/system.py`, `backend/tests/test_system.py`, `backend/tests/test_discovery.py`

- [ ] Write failing parsers for DGX Spark `nvidia-smi`, unified-memory fallback, Docker container commands and Hugging Face cache names.
- [ ] Implement real psutil/NVIDIA metrics with explicit unsupported values.
- [ ] Implement idempotent discovery of Docker OpenAI endpoints, HF cache repositories and configured model roots.
- [ ] Add system summary, metrics, model scan and deployment scan endpoints.
- [ ] Verify against fake clients, then run read-only discovery on the DGX Spark.

### Task 4: Persistent task engine and Hugging Face

**Files:** `backend/app/tasks/engine.py`, `backend/app/tasks/huggingface.py`, `backend/app/api/tasks.py`, `backend/app/api/huggingface.py`, `backend/tests/test_tasks.py`, `backend/tests/test_huggingface.py`

- [ ] Write failing state-machine, idempotency, restart recovery, repository validation and path-containment tests.
- [ ] Implement a single-host worker with durable task state and cooperative process cancellation.
- [ ] Add HF search/model-info endpoints and download tasks with byte monitoring, pause, resume and cancel.
- [ ] Trigger inventory rescan after verified completion and write audit events for every transition.
- [ ] Run tests using a temporary fake Hub repository, then a small real public-model download on DGX Spark.

### Task 5: Deployment adapters and lifecycle

**Files:** `backend/app/runtime/sglang.py`, `backend/app/runtime/vllm.py`, `backend/app/services/deployments.py`, `backend/app/api/deployments.py`, `backend/tests/test_deployments.py`

- [ ] Write failing tests for runtime compatibility, deterministic container names, allowed images/arguments, port allocation and rollback.
- [ ] Implement SGLang and vLLM adapters with preview, create, start, stop, restart, logs and health methods.
- [ ] Add deployment preview/create/update/clone/delete and lifecycle endpoints with idempotency keys.
- [ ] Preserve discovered containers and prohibit deletion unless the deployment is manager-owned.
- [ ] Verify existing `qwen38-dspark` and `nemotron-dspark` are imported without restart.

### Task 6: OpenAI-compatible gateway

**Files:** `backend/app/gateway/router.py`, `backend/app/gateway/proxy.py`, `backend/app/api/gateway.py`, `backend/tests/test_gateway.py`, `examples/openai_client.py`, `examples/openai_client.ts`

- [ ] Write failing tests for `/v1/models`, aliases, auth, OpenAI errors, non-streaming proxy, SSE passthrough, cancellation and metrics.
- [ ] Implement deployment selection and httpx streaming proxy for chat, completions and supported embeddings.
- [ ] Record request count, latency, status and returned usage without logging prompts or credentials.
- [ ] Verify Python OpenAI SDK non-streaming and streaming calls against both existing services.

### Task 7: Providers and approved diagnostics

**Files:** `backend/app/services/providers.py`, `backend/app/services/diagnostics.py`, `backend/app/operations/executor.py`, `backend/app/api/providers.py`, `backend/app/api/diagnostics.py`, `backend/tests/test_providers.py`, `backend/tests/test_diagnostics.py`

- [ ] Write failing SSRF, encrypted-secret, connection-test, JSON-schema, operation-whitelist and approval tests.
- [ ] Implement provider CRUD with masked responses and OpenAI-compatible connectivity checks.
- [ ] Gather real system/log context and request structured diagnostic plans.
- [ ] Implement approve/reject and whitelisted start/stop/restart/rescan execution with per-step audit.
- [ ] Verify with a mock provider and one configured real provider when credentials are available.

### Task 8: Product shell and dashboard

**Files:** `frontend/src/app/App.tsx`, `frontend/src/app/theme.ts`, `frontend/src/styles.css`, `frontend/src/pages/LoginPage.tsx`, `frontend/src/pages/DashboardPage.tsx`, `frontend/src/components/AppShell.tsx`, `frontend/src/components/MetricStrip.tsx`, `frontend/src/test/dashboard.test.tsx`

- [ ] Write failing component tests for authentication, theme persistence, real metric labels and mobile navigation.
- [ ] Implement the Ant Design shell, light/dark/system theme and responsive sidebar/Drawer.
- [ ] Build the dashboard from real system, deployment, request and task queries with loading/error/unsupported states.
- [ ] Run Vitest, axe checks and screenshot the dashboard at desktop and mobile widths.

### Task 9: Management workflows

**Files:** `frontend/src/pages/ModelsPage.tsx`, `HuggingFacePage.tsx`, `DeploymentsPage.tsx`, `GatewayPage.tsx`, `ProvidersPage.tsx`, `DiagnosticsPage.tsx`, `TasksPage.tsx`, `AuditPage.tsx`, `SettingsPage.tsx`, `frontend/src/components/ResponsiveDataView.tsx`, `TaskProgress.tsx`, `ApprovalPanel.tsx`, `LogViewer.tsx`

- [ ] Write failing tests for desktop/mobile data views, download creation, deployment confirmation, provider masking and plan approval.
- [ ] Implement all ten required navigation surfaces using shared typed query/mutation hooks.
- [ ] Add real-time task polling/SSE, deployment log viewing, API examples and destructive confirmations.
- [ ] Verify every visible command invokes a backend action and all mutations expose loading/success/error states.
- [ ] Run responsive, keyboard, dark-mode and longest-label visual checks.

### Task 10: Packaging, operations and documentation

**Files:** `Dockerfile`, `compose.yaml`, `scripts/install.sh`, `scripts/update.sh`, `scripts/uninstall.sh`, `scripts/backup.sh`, `scripts/restore.sh`, `README.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/TROUBLESHOOTING.md`

- [ ] Write configuration and image-architecture validation tests.
- [ ] Build the multi-stage image with multi-arch Node/Python bases and non-root application user.
- [ ] Add installation preview/manual/automatic modes, generated secrets, health wait and rollback instructions.
- [ ] Document setup, API clients, backup/restore, upgrade/uninstall and DGX-specific troubleshooting.
- [ ] Run secret scans and confirm no device credential, token or absolute user path is tracked.

### Task 11: DGX Spark deployment and end-to-end acceptance

**Files:** `tests/e2e/manager.spec.ts`, `scripts/verify_openai.py`

- [ ] Deploy under `/opt/dgx-spark-manager` without stopping existing inference containers.
- [ ] Run migrations, initial discovery and health checks; confirm both current models are managed.
- [ ] Exercise login, HF search/download, task persistence, lifecycle preview, Provider test and diagnostic approval.
- [ ] Run OpenAI SDK stream/non-stream calls through the gateway.
- [ ] Capture desktop/mobile light/dark screenshots and inspect for blank content, overflow and overlap.
- [ ] Restart the manager and verify models, deployments, tasks and audit history persist.

### Task 12: GitHub delivery

**Files:** repository metadata and all tracked project files

- [ ] Run `git status`, secret scan and the complete backend/frontend/E2E verification suite.
- [ ] Commit the intended project files on `main` with a concise initial commit.
- [ ] Create the public or private GitHub repository `DGX-Spark-web-manager` under the authenticated account.
- [ ] Push `main`, verify the remote tree and add the deployed URL to the repository description or README.
