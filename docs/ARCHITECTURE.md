# Architecture

## Runtime Topology

The manager is one multi-stage ARM64 container. Node builds the React SPA, then the Python image runs FastAPI and serves both the API and compiled assets. Docker Compose uses host networking so the gateway can reach existing inference endpoints at `127.0.0.1:<port>` without rewriting container routes.

Mounted resources:

- `/var/run/docker.sock`: Docker discovery and validated lifecycle actions.
- `/hf-cache`: the host Hugging Face cache.
- `/models`: the host model directory.
- `/app/data`: SQLite, task state, audit records, and encrypted secrets.
- `/host/etc/os-release`: host operating-system identity.

`DGX_MODEL_ROOT_MAPPINGS` translates the manager container paths back to their host bind sources
before the Docker API creates a runtime container. This prevents container-local paths such as
`/hf-cache/hub` from being interpreted as nonexistent host directories.

The manager container does not replace or wrap inference containers. Runtime adapters create normal NVIDIA containers using the same host GPU and model caches.

## Backend Domains

| Domain | Responsibility |
| --- | --- |
| `auth` / `security` | Administrator session, CSRF, API-key hashing, Fernet encryption |
| `system` | Host/GPU resource collection with explicit unsupported values |
| `discovery` | Idempotent Docker and model-cache inventory |
| `tasks` | Durable single-host queue, progress, pause/resume/cancel, restart recovery |
| `runtime` | SGLang/vLLM environment and model checks, configuration, lifecycle, health, logs, metrics, uninstall, and OpenAI capabilities |
| `deployment_recommendations` | Model evidence, deterministic settings, bounded AI fallback, resource estimates, and Draft candidates |
| `deployments` | Idempotent container creation, health wait, preview/edit/clone workflows, lifecycle, logs, and rollback |
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

## Recommendation Pipeline

A recommendation is owned by the complete `(model_id, runtime, image, provider_id)` tuple. The
backend performs these stages in order:

1. Resolve an available local model asset and load bounded evidence from `README.md`, `config.json`,
   `generation_config.json`, tokenizer files, and model metadata. When the local card is absent and
   the asset has a repository ID, a bounded Hugging Face model-card fetch can overlay the local
   evidence. Evidence is hashed for provenance.
2. Resolve runtime capabilities for the selected allowlisted image. A network-disabled, no-volume
   help probe detects speculative transport and methods. Probe failure returns a conservative
   manifest with warnings instead of silently claiming probe evidence.
3. Select deterministic settings. Model-card values take precedence, followed by local config and
   conservative runtime defaults. Device rules map or clamp quantization, context, concurrency, and
   batch limits against capability and memory constraints.
4. If allowlisted fields remain unresolved or have low-confidence numeric defaults, an enabled
   third-party OpenAI-compatible provider whose latest test is not failed can fill only those fields.
   Complete high-confidence deterministic recommendations do not call AI.
5. Recompute the current unified-memory estimate and classify local Draft Models as `compatible`,
   `review`, or `incompatible` using method, target pairing, tokenizer evidence, and combined model
   memory.

The AI cache defaults to 900 seconds and is keyed by model revision, evidence hash, runtime, image
digest, provider ID, and recommendation schema version. `refresh_ai=true` bypasses a cache hit. A
fresh system/deployment snapshot and resource estimate are still collected on every recommendation
request. Runtime capability results are cached separately by `(runtime, image_digest)`, so two tags
for the same immutable image reuse probe evidence while the response preserves the selected tag.

Each value retains `source`, confidence, reason, and an optional warning. When the administrator
edits an applied value, the wizard records its dotted field path in `modified_fields`; `sources`
contains only retained recommended fields. Provenance identifies when and from which evidence and
provider the starting recommendation was generated.

## Preview and Execution Boundaries

The browser debounces recommendation changes and ignores late responses that do not own its current
tuple. Changing the model, runtime, image, or provider clears tuple-bound values, Draft selection,
acknowledgements, and preview state. The final submit uses an immutable copy of the exact form payload
that produced the displayed preview; editing after preview requires another preview.

Preview is not a cached authorization decision. The backend resolves the current model and image,
rechecks the image allowlist and probed/manifest capabilities, resolves Draft paths and method
mappings, recomputes base-plus-Draft memory, enforces acknowledgements, checks model compatibility,
and validates route consistency. Create and edit call the same preflight again before queueing work,
and the deployment worker resolves and validates the stored spec before container creation.

`route_alias` groups healthy instances behind one public model name. All deployments sharing an
effective route must have identical normalized generation defaults. The gateway round-robins route
members, advertises the intersection of their OpenAI capabilities, and applies only defaults present
in the selected deployment's saved capability snapshot.

## Trust Boundaries

The browser can request only typed REST operations. It cannot submit Docker commands or runtime
argument arrays. The backend validates model paths, images, ports, context sizes, memory ratios,
quantization, capability mappings, and runtime-specific flags before creating a task.

Deployment edits use a health-gated replacement. The old container is stopped and renamed, the
replacement is created from the validated spec, and the existing database record is updated only
after `/v1/models` responds. A failed replacement is removed and the old container name/state is
restored. Discovery merges live container observations into manager-owned configuration instead of
discarding the saved spec.

Model cards and structured metadata are treated as untrusted input. Before recommendation data is
sent to a provider, credentials and paths are redacted, structured fields are allowlisted, strings
and deployment context are bounded, and the prompt places data in a JSON envelope with an explicit
instruction not to follow embedded model-card instructions. Redirects, compressed responses,
oversized bodies, non-JSON output, unknown fields, unsupported values, and values outside strict
ranges are rejected or dropped.

Local evidence files must be regular files. The only accepted evidence-file symlinks are canonical
Hugging Face cache entries under `models--<owner>--<model>/snapshots/<revision>` whose exact relative
target is `../../blobs/<40-or-64-character-hex-hash>`. The evidence reader opens the snapshot,
repository, blob directory, and blob through directory-relative no-follow handles on POSIX. On
Windows it checks the original repository, snapshots, and revision directories for reparse points
before resolving, then verifies their identities and the blob path again before reading. Arbitrary
links, linked blob directories or files, and non-regular blob targets remain unreadable.

AI can suggest only a bounded set of deployment and generation values. It cannot choose images,
quantization, paths, compatibility status, runtime operations, or shell commands. Suggestions remain
visible for human review and are never deployment authorization. The server revalidates the final
administrator-approved spec against current evidence, capabilities, and resources.

Diagnostic AI providers receive a bounded snapshot of real system values and tail logs after secret
redaction. Returned JSON is reduced to known fields. Unknown operations remain visible with
`executable=false`; the executor never evaluates text or invokes a shell.

## Frontend

React Router provides ten authenticated surfaces. TanStack Query owns server state and polling, Zustand keeps only session/CSRF and theme preferences, and Ant Design supplies accessible product controls. Large pages are code-split. Tables switch to summary lists on narrow viewports, while destructive operations keep explicit confirmation text.
