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

## Host Operations Agent

The non-root manager container does not implement the privileged host command execution path.
Privileged diagnostics and approved repairs cross a separate Host Operations Agent trust boundary:

```text
browser -> authenticated manager API -> signed Unix socket request -> root Host Agent
```

The manager receives the Agent key through the read-only
`/run/secrets/ops-agent.key` mount and connects to
`/run/dgx-spark-manager/ops-agent.sock`. The socket is local to the host and is never exposed as a
TCP listener. The container runs with its normal non-root UID plus the numeric `OPS_AGENT_GID`,
which must match the host `dgx-spark-ops` group. The root Agent is the sole executor for the Host
shell and privileged diagnostic or repair command path. The manager still uses its existing Docker
socket mount directly for validated inference-container discovery and lifecycle actions.

Protocol version 1 uses bounded length-prefixed JSON and HMAC-SHA256 authentication. Every request
has a UUID `request_id`, timestamp, and random nonce. The Agent verifies the signature and timestamp
before atomically consuming the nonce, so expired, future, replayed, malformed, or oversized
requests fail before dispatch. Every response is signed and bound to the original `request_id`; the
manager rejects stale, mismatched, malformed, or unsigned responses.

The Agent policy exposes exactly eleven structured read tools: `host.memory`, `host.disk`,
`host.gpu`, `host.ports`, `host.processes`, `docker.list`, `docker.inspect`, `docker.logs`,
`docker.stats`, `systemd.status`, and `systemd.journal`. They use fixed absolute executables and
validated selectors and can run automatically during diagnosis. `shell.execute` is a separate
action and is rejected unless the request carries approval metadata from an administrator-approved
`OperationPlan`. Shell execution uses `/bin/bash --noprofile --norc -c` with a fixed safe `PATH`, a
small environment allowlist, bounded command/cwd/timeout values, and no inherited manager or Agent
secrets.

Commands execute as durable asynchronous Agent jobs. Output is UTF-8 normalized, streamed through
secret redaction, retained within a byte limit, and addressed by absolute `output_offset` and
`truncated_before` values so polling does not duplicate output. Timeout and cancellation terminate
the command process group with a bounded TERM/KILL sequence. Atomic root-only metadata records
process identity and a private recovery token; on restart the Agent revalidates `/proc` identity and
uses pidfds before terminating an interrupted job. The systemd service uses
`KillMode=control-group` as the final service-restart containment boundary.

The Agent is installed as a root systemd socket-activated service. The socket is
`root:dgx-spark-ops 0660`, the shared key is `root:dgx-spark-ops 0640`, and the initialized job
directory is `root:root 0700`. The service runs as root with `NoNewPrivileges=true`, a private
temporary directory, and `UMask=0027`.

`scripts/install-ops-agent.sh` is read-only unless passed `--apply`. Apply accepts only ARM64,
preserves a valid existing key and job history, stages package and unit replacements, activates the
socket, and completes a signed `agent.health` probe. A failed transaction restores the previous
package, units, and systemd state. `scripts/uninstall-ops-agent.sh` also previews by default; apply
removes the package and units but preserves the key and jobs. Physical deletion requires the
explicit `--purge-key` or `--purge-jobs` flag and the exact interactive confirmation shown by the
uninstaller.

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
| `providers` | Encrypted external OpenAI-compatible API configuration, structured connection/default-model readiness probes, and bounded response repair |
| `diagnostics` | Persistent operations sessions, ordered messages/tool runs, bounded read-only orchestration, and linked plans |
| `operations` | Digest-bound administrator approval and asynchronous execution of the exact displayed Shell plan through the Host Agent |
| `audit` | Actor, action, resource, outcome, IP, and redacted details |

## State And Recovery

SQLite runs in WAL-compatible single-host mode through SQLAlchemy. Alembic applies schema migrations before Uvicorn starts. Background tasks use:

```text
queued -> running -> succeeded | failed | paused | cancelled
paused -> queued | cancelled
failed | cancelled -> queued
```

An interrupted `running` task returns to `queued` during startup and appends a recovery log entry. Download tasks use the Hugging Face cache, so a resumed task reuses completed blobs.

## AI Operations Lifecycle

AI operations is a persistent, task-backed workflow rather than a synchronous chat request:

1. An administrator creates a session and queues a user message as an `ops.respond` task.
2. The configured Provider runs a bounded tool loop. The manager validates every tool name and
   argument before dispatch and limits iterations, output, and retained context.
3. The eleven structured Host Agent read tools can run without per-call approval. Their outputs are
   bounded, redacted, persisted as ordered tool runs, and fed back to the Provider for diagnosis.
4. A proposed repair is persisted as a pending `OperationPlan`. Its exact command, working
   directory, timeout, impact, and rollback are displayed in the panel and covered by a canonical
   digest.
5. Administrator approval queues an `operation.execute` task. The worker recomputes the digest
   before calling `shell.execute`; changed or stale plan data fails closed. Redacted output, task
   state, plan state, and audit records remain visible while the command runs.
6. Rejection leaves the plan immutable and returns the session to `active`, allowing a revised
   message without losing the preceding evidence.

The Settings history-clear action is a separate confirmation-gated maintenance transaction. It
refuses to run while AI response or operation tasks/plans are active, physically deletes failed task
history plus operation sessions/messages/tool runs/plans and their related audit rows, then retains
one aggregate `maintenance.history.clear` audit event. Model, deployment, Provider, API-key,
Hugging Face secret, gateway metric, and successful task records are preserved.

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
repository, blob directory, and blob through directory-relative no-follow handles on POSIX. For a
Hugging Face snapshot, POSIX traversal starts from an open handle for the repository parent and opens
the lexical repository, `snapshots`, and revision names one level at a time; regular evidence and
canonical blobs reuse those handles without resolving ancestor links. On Windows it checks the
original repository, snapshots, and revision directories for reparse points before resolving, then
verifies their identities and the blob path again before reading. Arbitrary links, linked blob
directories or files, and non-regular blob targets remain unreadable.

Deployment-recommendation AI can suggest only a bounded set of deployment and generation values. It cannot choose images,
quantization, paths, compatibility status, runtime operations, or shell commands. Suggestions remain
visible for human review and are never deployment authorization. The server revalidates the final
administrator-approved spec against current evidence, capabilities, and resources.

Diagnostic AI providers receive bounded, redacted system and tool results. Read-only Host Agent
tools use validated structured arguments. Shell text is never executed from a Provider response:
it first becomes a persisted plan, is shown verbatim to an administrator, and may run only after
approval metadata and the canonical plan digest are revalidated by both manager and Agent.

The administrator-only `GET /api/ops-agent/health` endpoint reports `ok`, `unavailable`, or `error`.
A healthy response includes the protocol version; unavailable and error responses include only a
fixed redacted `detail`. It does not return socket paths, key material, raw remote errors, commands,
or job output.

## DGX Spark Acceptance

Acceptance on 2026-08-17 used an `aarch64` DGX Spark host. The Linux installer suite passed all 53
tests; preview made no writes, and apply completed a signed protocol-v1 health check. Runtime modes
were verified as socket `0660`, key `0640`, and jobs `0700`. Structured reads, approved Shell,
streaming redaction, cancellation, and restart recovery passed. The ARM64 manager image built
natively, and its non-root Compose container reached the Agent health endpoint through the local
socket.

## Frontend

React Router provides ten authenticated surfaces. TanStack Query owns server state and polling, Zustand keeps only session/CSRF and theme preferences, and Ant Design supplies accessible product controls. Large pages are code-split. Tables switch to summary lists on narrow viewports, while destructive operations keep explicit confirmation text.
