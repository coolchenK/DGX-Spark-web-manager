# Host Operations Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install an ARM64-native root systemd Agent that safely exposes structured read-only diagnostics and approved asynchronous Shell jobs to the manager over an authenticated Unix Socket.

**Architecture:** The Agent is a dependency-light Python package using a length-prefixed JSON protocol over a Unix domain socket. Every request is HMAC-signed, timestamped, and nonce-protected; policy validation occurs in the Agent, not in the AI layer. The FastAPI manager uses a typed client and never exposes the Agent socket to browsers or Provider APIs.

**Tech Stack:** Python 3.12 standard library, Unix sockets, systemd socket activation, HMAC-SHA256, subprocess process groups, pytest, Docker Compose, Bash.

---

## File Map

- Create `host_agent/dgx_ops_agent/__init__.py`: package version and protocol version.
- Create `host_agent/dgx_ops_agent/protocol.py`: canonical JSON, signing, verification, framing, and nonce cache.
- Create `host_agent/dgx_ops_agent/policy.py`: structured tool schemas and approval enforcement.
- Create `host_agent/dgx_ops_agent/redaction.py`: bounded credential redaction shared by Agent output.
- Create `host_agent/dgx_ops_agent/runner.py`: subprocess execution, process groups, output limits, job state.
- Create `host_agent/dgx_ops_agent/server.py`: Unix socket request dispatcher and CLI entry point.
- Create `backend/app/services/ops_agent.py`: authenticated manager-side Agent client.
- Modify `backend/app/config.py`: Agent socket, key, timeouts, and output limits.
- Modify `backend/app/main.py`: construct and expose the client.
- Create `backend/app/api/ops_agent.py`: authenticated health endpoint.
- Modify `backend/app/main.py`: include the Agent health router.
- Create `backend/tests/test_ops_agent_protocol.py`: protocol, policy, replay, and runner tests.
- Create `backend/tests/test_ops_agent_client.py`: client transport and API health tests.
- Create `deploy/systemd/dgx-spark-ops-agent.service`: root service unit.
- Create `deploy/systemd/dgx-spark-ops-agent.socket`: Unix Socket unit.
- Create `scripts/install-ops-agent.sh`: idempotent preview/apply installer.
- Create `scripts/uninstall-ops-agent.sh`: service removal that preserves job logs by default.
- Modify `scripts/install.sh`: install and verify Agent before starting the manager.
- Modify `compose.yaml`: socket/key mounts and Agent group membership.
- Modify `.env.example`: Agent paths and group ID.
- Modify `docs/ARCHITECTURE.md`: trust boundary and protocol.
- Modify `docs/TROUBLESHOOTING.md`: Agent health and recovery.

### Task 1: Signed Protocol and Replay Protection

**Files:**
- Create: `host_agent/dgx_ops_agent/__init__.py`
- Create: `host_agent/dgx_ops_agent/protocol.py`
- Create: `backend/tests/test_ops_agent_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_signed_request_round_trip():
    secret = b"x" * 32
    request = new_request("host.memory", {}, now=1000, nonce="nonce-1")
    signed = sign_message(request, secret)
    verified = verify_message(signed, secret, now=1005, nonces=NonceCache())
    assert verified["action"] == "host.memory"


def test_rejects_tampering_expiration_and_replay():
    secret = b"x" * 32
    cache = NonceCache()
    signed = sign_message(new_request("host.memory", {}, now=1000, nonce="n"), secret)
    verify_message(signed, secret, now=1001, nonces=cache)
    with pytest.raises(ProtocolError, match="replayed"):
        verify_message(signed, secret, now=1002, nonces=cache)
    with pytest.raises(ProtocolError, match="expired"):
        verify_message(sign_message(new_request("host.memory", {}, now=1, nonce="old"), secret), secret, now=1000, nonces=NonceCache())
    signed["parameters"] = {"changed": True}
    with pytest.raises(ProtocolError, match="signature"):
        verify_message(signed, secret, now=1001, nonces=NonceCache())
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k "signed or tampering" -v`

Expected: FAIL because `host_agent.dgx_ops_agent.protocol` does not exist.

- [ ] **Step 3: Implement canonical signing and bounded framing**

```python
PROTOCOL_VERSION = 1


def canonical_bytes(message: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in message.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sign_message(message: dict[str, Any], secret: bytes) -> dict[str, Any]:
    result = dict(message)
    result["signature"] = hmac.new(secret, canonical_bytes(result), hashlib.sha256).hexdigest()
    return result


def verify_message(message: dict[str, Any], secret: bytes, *, now: int,
                   nonces: NonceCache, max_age: int = 30) -> dict[str, Any]:
    signature = str(message.get("signature") or "")
    expected = hmac.new(secret, canonical_bytes(message), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ProtocolError("invalid signature")
    timestamp = int(message.get("timestamp", 0))
    if abs(now - timestamp) > max_age:
        raise ProtocolError("expired request")
    nonce = str(message.get("nonce") or "")
    nonces.consume(nonce, timestamp)
    return {key: value for key, value in message.items() if key != "signature"}
```

Use a four-byte big-endian frame length and reject frames over 1 MiB before allocating the body.

- [ ] **Step 4: Run protocol tests**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k "signed or tampering or frame" -v`

Expected: PASS.

- [ ] **Step 5: Commit the protocol**

```bash
git add host_agent/dgx_ops_agent backend/tests/test_ops_agent_protocol.py
git commit -m "feat: add authenticated ops agent protocol"
```

### Task 2: Agent-Side Read-Only Policy

**Files:**
- Create: `host_agent/dgx_ops_agent/policy.py`
- Modify: `backend/tests/test_ops_agent_protocol.py`

- [ ] **Step 1: Write failing policy tests**

```python
@pytest.mark.parametrize("action", [
    "host.memory", "host.disk", "host.gpu", "host.ports", "host.processes",
    "docker.list", "docker.inspect", "docker.logs", "docker.stats",
    "systemd.status", "systemd.journal",
])
def test_read_only_actions_never_require_approval(action):
    request = validate_action(action, valid_parameters(action), approval=None)
    assert request.read_only is True


def test_shell_always_requires_complete_approval():
    with pytest.raises(PolicyError, match="approval"):
        validate_action("shell.execute", {"command": "id", "cwd": "/"}, approval=None)
    approved = validate_action("shell.execute", {"command": "id", "cwd": "/", "timeout": 30}, approval={
        "plan_id": "plan-1", "step_id": "step-1", "approved_by": "admin",
        "approved_at": "2026-08-17T00:00:00Z",
    })
    assert approved.read_only is False
```

- [ ] **Step 2: Run policy tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k "read_only or shell_always" -v`

Expected: FAIL because policy validation is missing.

- [ ] **Step 3: Implement fixed tool specifications**

```python
READ_ONLY_TOOLS: dict[str, ToolSpec] = {
    "host.memory": ToolSpec(["/usr/bin/free", "--bytes"], set(), 10),
    "host.disk": ToolSpec(["/usr/bin/df", "--block-size=1", "--output=source,target,size,used,avail,pcent"], set(), 10),
    "host.gpu": ToolSpec(["/usr/bin/nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits"], set(), 15),
    "host.ports": ToolSpec(["/usr/bin/ss", "-lntupH"], set(), 10),
    "host.processes": ToolSpec(["/usr/bin/ps", "-eo", "pid,ppid,user,stat,%cpu,%mem,etimes,comm", "--sort=-%cpu"], set(), 10),
}


def validate_action(action: str, parameters: dict[str, Any], approval: dict[str, Any] | None) -> ValidatedAction:
    if action in READ_ONLY_TOOLS:
        return READ_ONLY_TOOLS[action].bind(parameters)
    if action == "shell.execute":
        command, cwd, timeout = validate_shell_parameters(parameters)
        validate_approval(approval)
        return ValidatedAction(action, ["/bin/bash", "-lc", command], cwd,
                               timeout, read_only=False)
    raise PolicyError("unknown action")
```

`validate_shell_parameters` and `validate_approval` use standard-library type/range checks and dataclasses; the host package must not import Pydantic or any manager dependency. Docker and systemd tools must bind only validated container IDs/names, service names matching `[A-Za-z0-9_.@-]+`, integer tail limits from 1 to 5000, and fixed executable paths. They must never concatenate parameters into a Shell string.

- [ ] **Step 4: Run all policy tests**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k "policy or action or approval" -v`

Expected: PASS, including rejection of metacharacters in service/container selectors.

- [ ] **Step 5: Commit policy**

```bash
git add host_agent/dgx_ops_agent/policy.py backend/tests/test_ops_agent_protocol.py
git commit -m "feat: enforce host operations policy"
```

### Task 3: Bounded Redacted Job Runner

**Files:**
- Create: `host_agent/dgx_ops_agent/redaction.py`
- Create: `host_agent/dgx_ops_agent/runner.py`
- Modify: `backend/tests/test_ops_agent_protocol.py`

- [ ] **Step 1: Write failing runner tests**

```python
def test_runner_redacts_secrets_and_bounds_output(tmp_path):
    runner = JobRunner(tmp_path, output_limit=120)
    job = runner.start([sys.executable, "-c", "print('Authorization: Bearer secret-token-' + 'x'*300)"], cwd=tmp_path, timeout=10)
    result = wait_for_terminal(runner, job.id)
    assert "secret-token" not in result.output
    assert "[REDACTED]" in result.output
    assert len(result.output.encode()) <= 120


def test_cancel_terminates_the_process_group(tmp_path):
    runner = JobRunner(tmp_path)
    job = runner.start([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, timeout=120)
    runner.cancel(job.id)
    assert wait_for_terminal(runner, job.id).status == "cancelled"
```

- [ ] **Step 2: Run runner tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k "runner or cancel" -v`

Expected: FAIL because runner and redaction modules are missing.

- [ ] **Step 3: Implement process-group jobs and redaction**

```python
process = subprocess.Popen(
    argv, cwd=cwd, env=safe_environment(env), stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=False, start_new_session=True,
)
```

Read output on a worker thread, redact each decoded chunk plus a carry buffer for split secrets, and retain only the final `output_limit` bytes. Track monotonically increasing `output_offset` and `truncated_before`; `job.get` returns these with the retained output so the manager can append deltas without duplication. On timeout or cancel, send `SIGTERM` to `os.killpg(process.pid, signal.SIGTERM)`, wait five seconds, then use `SIGKILL`. Persist job metadata atomically as JSON without storing the HMAC key or full environment.

- [ ] **Step 4: Run runner tests**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k "runner or cancel or timeout or redact" -v`

Expected: PASS with no leaked test token.

- [ ] **Step 5: Commit runner**

```bash
git add host_agent/dgx_ops_agent/redaction.py host_agent/dgx_ops_agent/runner.py backend/tests/test_ops_agent_protocol.py
git commit -m "feat: execute bounded redacted host jobs"
```

### Task 4: Unix Socket Agent Server

**Files:**
- Create: `host_agent/dgx_ops_agent/server.py`
- Modify: `backend/tests/test_ops_agent_protocol.py`

- [ ] **Step 1: Write failing server dispatch tests**

```python
def test_server_dispatches_health_read_and_job_lifecycle(agent_server, signed_client):
    assert signed_client.call("agent.health", {})["protocol_version"] == 1
    result = signed_client.call("host.memory", {})
    assert result["status"] == "succeeded"
    created = signed_client.call("shell.execute", {"command": "printf ok", "cwd": "/", "timeout": 10}, approval=approval())
    final = poll_agent_job(signed_client, created["job_id"])
    assert final["exit_code"] == 0
    assert final["output"] == "ok"
```

- [ ] **Step 2: Run server tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -k server -v`

Expected: FAIL because the server entry point is missing.

- [ ] **Step 3: Implement authenticated dispatch**

The server must support exactly `agent.health`, policy-defined read tools, `shell.execute`, `job.get`, and `job.cancel`. It verifies the signed request before dispatch, signs every response, maps protocol/policy errors to stable codes, and never returns a Python traceback.

```python
DISPATCH = {
    "agent.health": self.health,
    "job.get": self.get_job,
    "job.cancel": self.cancel_job,
}


def handle(self, message: dict[str, Any]) -> dict[str, Any]:
    request = verify_message(message, self.secret, now=int(time.time()), nonces=self.nonces)
    action = request["action"]
    if action in DISPATCH:
        result = DISPATCH[action](request)
    else:
        validated = validate_action(action, request.get("parameters", {}), request.get("approval"))
        result = self.run(validated)
    return sign_message(response_for(request, result), self.secret)
```

- [ ] **Step 4: Run complete Agent unit tests**

Run: `python -m pytest backend/tests/test_ops_agent_protocol.py -v`

Expected: PASS.

- [ ] **Step 5: Commit server**

```bash
git add host_agent/dgx_ops_agent/server.py backend/tests/test_ops_agent_protocol.py
git commit -m "feat: serve host operations over unix socket"
```

### Task 5: Manager Agent Client and Health API

**Files:**
- Create: `backend/app/services/ops_agent.py`
- Create: `backend/app/api/ops_agent.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_ops_agent_client.py`

- [ ] **Step 1: Write failing client and health API tests**

```python
def test_client_signs_request_and_verifies_response(fake_socket, tmp_path):
    client = OpsAgentClient(fake_socket.path, fake_socket.key_path)
    response = client.call("agent.health", {})
    assert response.protocol_version == 1
    assert fake_socket.last_request["signature"]


def test_health_api_reports_unavailable_without_failing_manager(authenticated_client, monkeypatch):
    monkeypatch.setattr(authenticated_client.app.state.ops_agent_client, "health",
                        lambda: (_ for _ in ()).throw(OpsAgentUnavailable("missing socket")))
    response = authenticated_client.get("/api/ops-agent/health")
    assert response.status_code == 200
    assert response.json() == {"status": "unavailable", "detail": "missing socket"}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_agent_client.py -v`

Expected: FAIL because the client and route do not exist.

- [ ] **Step 3: Implement typed socket client and settings**

Add settings with non-secret defaults:

```python
ops_agent_socket: Path = Path("/run/dgx-spark-manager/ops-agent.sock")
ops_agent_key_file: Path = Path("/run/secrets/ops-agent.key")
ops_agent_connect_timeout_seconds: float = Field(default=3, ge=0.5, le=30)
ops_agent_output_limit_bytes: int = Field(default=1_000_000, ge=10_000, le=10_000_000)
```

The client reads the key for each new process lifetime, opens `AF_UNIX/SOCK_STREAM`, writes one bounded frame, reads one bounded frame, validates response request ID and HMAC, and raises `OpsAgentUnavailable`, `OpsAgentProtocolError`, or `OpsAgentRemoteError` without exposing signatures.

- [ ] **Step 4: Run client/API tests and backend lint**

Run: `python -m pytest backend/tests/test_ops_agent_client.py -v && python -m ruff check backend host_agent`

Expected: PASS.

- [ ] **Step 5: Commit manager integration**

```bash
git add backend/app/services/ops_agent.py backend/app/api/ops_agent.py backend/app/config.py backend/app/main.py backend/tests/test_ops_agent_client.py
git commit -m "feat: connect manager to host operations agent"
```

### Task 6: systemd Units and Idempotent Installer

**Files:**
- Create: `deploy/systemd/dgx-spark-ops-agent.service`
- Create: `deploy/systemd/dgx-spark-ops-agent.socket`
- Create: `scripts/install-ops-agent.sh`
- Create: `scripts/uninstall-ops-agent.sh`
- Modify: `scripts/install.sh`
- Modify: `.env.example`
- Modify: `compose.yaml`

- [ ] **Step 1: Add a shell-level installer contract test**

Create `backend/tests/test_ops_agent_install.py` that reads the scripts/units and Compose text and asserts: preview mode performs no privileged writes, apply requires ARM64, key generation uses mode 0640, the socket mode is 0660, and Compose mounts the socket directory and key read-only. The test uses only the standard library and does not add a YAML parser dependency.

```python
def test_install_contract():
    script = Path("scripts/install-ops-agent.sh").read_text()
    socket = Path("deploy/systemd/dgx-spark-ops-agent.socket").read_text()
    compose = Path("compose.yaml").read_text()
    assert '[[ "${1:-}" == "--apply" ]]' in script
    assert "SocketMode=0660" in socket
    assert "- /run/dgx-spark-manager:/run/dgx-spark-manager" in compose
    assert "- /etc/dgx-spark-manager/ops-agent.key:/run/secrets/ops-agent.key:ro" in compose
```

- [ ] **Step 2: Run the contract test and verify RED**

Run: `python -m pytest backend/tests/test_ops_agent_install.py -v`

Expected: FAIL because units and scripts do not exist.

- [ ] **Step 3: Implement units and installer**

The socket unit must use:

```ini
[Socket]
ListenStream=/run/dgx-spark-manager/ops-agent.sock
SocketUser=root
SocketGroup=dgx-spark-ops
SocketMode=0660
RemoveOnStop=true
```

The service uses `/usr/bin/python3 -m dgx_ops_agent.server --systemd`, sets `Environment=PYTHONPATH=/usr/local/lib/dgx-spark-ops-agent`, consumes the systemd-provided socket from file descriptor 3, runs as root, sets `UMask=0027`, `NoNewPrivileges=true`, and `PrivateTmp=true`. Do not set `ProtectSystem=strict`, because approved repair jobs must be able to modify the host.

The installer previews changes unless passed `--apply`; on apply it verifies `aarch64|arm64`, creates the group, copies the package to `/usr/local/lib/dgx-spark-ops-agent`, generates a 32-byte key with mode 0640 and group `dgx-spark-ops`, installs units, runs `daemon-reload`, enables the socket, and verifies `agent.health` through a local signed probe. `scripts/install.sh` reads the new group GID with `getent group dgx-spark-ops`, writes `OPS_AGENT_GID` to `.env` without changing existing secrets, and only then starts Compose. `compose.yaml` adds that GID to `group_add`.

- [ ] **Step 4: Validate scripts and Compose**

Run: `bash -n scripts/install-ops-agent.sh scripts/uninstall-ops-agent.sh scripts/install.sh && docker compose config >/dev/null && python -m pytest backend/tests/test_ops_agent_install.py -v`

Expected: syntax, Compose expansion with test environment variables, and contract tests pass.

- [ ] **Step 5: Commit deployment artifacts**

```bash
git add deploy/systemd scripts/install-ops-agent.sh scripts/uninstall-ops-agent.sh scripts/install.sh .env.example compose.yaml backend/tests/test_ops_agent_install.py
git commit -m "feat: install ARM64 host operations agent"
```

### Task 7: Agent Documentation and DGX Acceptance

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/TROUBLESHOOTING.md`

- [ ] **Step 1: Run all local Agent and backend tests**

Run: `python -m ruff check backend host_agent && python -m pytest backend/tests/test_ops_agent_protocol.py backend/tests/test_ops_agent_client.py backend/tests/test_ops_agent_install.py -v`

Expected: all checks pass.

- [ ] **Step 2: Preview installation on DGX Spark**

Run on DGX: `cd ~/dgx-spark-web-manager && ./scripts/install-ops-agent.sh`

Expected: prints architecture, files, group, socket, key path, changes, rollback command, and exits without changing systemd state.

- [ ] **Step 3: Apply and verify installation**

Run on DGX: `sudo ./scripts/install-ops-agent.sh --apply && systemctl status dgx-spark-ops-agent.socket --no-pager`

Expected: socket is active, `/run/dgx-spark-manager/ops-agent.sock` is `root:dgx-spark-ops 0660`, and the signed health probe reports protocol version 1.

- [ ] **Step 4: Verify policy boundaries**

Use the manager client to execute `host.memory` without approval and verify success. Submit `shell.execute` without approval and verify `approval_required`; then submit `printf agent-ok` with a test approval and verify output, exit code, audit metadata, timeout, cancel, and redaction behavior.

- [ ] **Step 5: Document operations and commit**

```bash
git add docs/ARCHITECTURE.md docs/TROUBLESHOOTING.md
git commit -m "docs: document host operations agent"
```
