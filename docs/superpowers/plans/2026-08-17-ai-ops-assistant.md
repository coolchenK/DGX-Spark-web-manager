# AI Operations Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle one-shot diagnostic flow with persistent AI operations sessions that can run approved host commands, automatically use structured read-only tools, and reliably handle reasoning-model Provider responses.

**Architecture:** Persistent sessions and messages are added through an Alembic migration while existing `OperationPlan` records remain compatible. `OpsProviderClient` owns OpenAI-compatible response repair, `OpsOrchestrator` owns the bounded tool loop, and `OperationExecutor` delegates approved Shell steps to the Host Operations Agent. The React diagnostics page becomes a responsive conversation and execution timeline.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/Alembic, Pydantic, HTTPX, Host Operations Agent protocol, pytest/respx, React 19, TypeScript, Ant Design, TanStack Query, Vitest.

---

## File Map

- Modify `backend/app/models.py`: `OpsSession`, `OpsMessage`, `OpsToolRun`, Provider probe result.
- Create `backend/migrations/versions/20260817_0002_ops_sessions.py`: additive tables and Provider probe column.
- Create `backend/app/services/ops_provider.py`: Provider request/parse/retry compatibility.
- Create `backend/app/services/ops_tools.py`: manager and Agent read-only tool registry.
- Create `backend/app/services/ops_orchestrator.py`: bounded session response loop and plan creation.
- Modify `backend/app/services/providers.py`: real default-model chat probe.
- Modify `backend/app/services/diagnostics.py`: legacy compatibility and new plan sanitization.
- Modify `backend/app/operations/executor.py`: approved Shell Agent job execution and task log updates.
- Modify `backend/app/api/providers.py`: return structured connection/model probe result.
- Rewrite `backend/app/api/diagnostics.py`: sessions, messages, history, approve/reject compatibility.
- Modify `backend/app/main.py`: construct/register orchestrator and `ops.respond` task.
- Create `backend/tests/test_ops_provider.py`: reasoning, truncation, JSON repair, response-format fallback.
- Create `backend/tests/test_ops_orchestrator.py`: tool loop, limits, session persistence, plan creation.
- Modify `backend/tests/test_diagnostics.py`: Shell approval and legacy history coverage.
- Modify `backend/tests/test_providers.py`: real model probe result coverage.
- Modify `frontend/src/api/types.ts`: session, message, tool run, richer plan/provider types.
- Rewrite `frontend/src/pages/DiagnosticsPage.tsx`: session UI and task-driven replies.
- Create `frontend/src/components/OpsConversation.tsx`: message and tool timeline.
- Create `frontend/src/components/OpsExecutionOutput.tsx`: stable terminal output area.
- Modify `frontend/src/components/ApprovalPanel.tsx`: Shell plan details and execution result.
- Modify `frontend/src/pages/ProvidersPage.tsx`: connection/default-model probe display.
- Create `frontend/src/pages/DiagnosticsPage.test.tsx`: conversation, auto-tool, approval, mobile flows.
- Create `frontend/src/pages/ProvidersPage.test.tsx`: probe detail behavior.
- Modify `frontend/src/styles.css`: responsive operations layout and dark terminal tokens.
- Modify `docs/API.md`: session and Provider probe contracts.
- Modify `docs/ARCHITECTURE.md`: orchestrator and approval data flow.

### Task 1: Persistent Operations Session Schema

**Files:**
- Modify: `backend/app/models.py`
- Create: `backend/migrations/versions/20260817_0002_ops_sessions.py`
- Create: `backend/tests/test_ops_orchestrator.py`

- [ ] **Step 1: Write a failing migration/model test**

```python
def test_ops_session_persists_messages_and_tool_runs(database):
    database.create_schema()
    with database.session_factory() as db:
        session = OpsSession(title="Repair gateway", provider_id=None, requested_by="admin")
        db.add(session); db.flush()
        db.add(OpsMessage(session_id=session.id, role="user", content="Check gateway"))
        db.add(OpsToolRun(session_id=session.id, tool_name="host.memory", risk="read_only",
                          status="succeeded", arguments_json={}, result_json={"available": 1}))
        db.commit(); session_id = session.id
    with database.session_factory() as db:
        assert len(db.get(OpsSession, session_id).messages) == 1
        assert len(db.get(OpsSession, session_id).tool_runs) == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest backend/tests/test_ops_orchestrator.py -k persists -v`

Expected: FAIL because the three models do not exist.

- [ ] **Step 3: Add additive models and migration**

```python
class OpsSession(TimestampMixin, Base):
    __tablename__ = "ops_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255))
    provider_id: Mapped[str | None] = mapped_column(ForeignKey("providers.id"))
    deployment_id: Mapped[str | None] = mapped_column(ForeignKey("deployments.id"))
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    requested_by: Mapped[str] = mapped_column(String(255))
    messages: Mapped[list["OpsMessage"]] = relationship(cascade="all, delete-orphan")
    tool_runs: Mapped[list["OpsToolRun"]] = relationship(cascade="all, delete-orphan")
```

`OpsMessage` stores `role`, bounded `content`, `metadata_json`, and optional `operation_plan_id`. `OpsToolRun` stores tool name, risk, sanitized arguments/result, status, Agent job ID, error, and timestamps. Add `Provider.last_test_result: JSON` with default `{}`. The migration creates only new tables/indexes and the nullable Provider column; downgrade removes only those additions.

- [ ] **Step 4: Run migration/model tests**

Run: `python -m pytest backend/tests/test_ops_orchestrator.py -k persists -v && python -m alembic -c alembic.ini upgrade head`

Expected: PASS and Alembic advances to `20260817_0002` on a temporary database URL.

- [ ] **Step 5: Commit schema**

```bash
git add backend/app/models.py backend/migrations/versions/20260817_0002_ops_sessions.py backend/tests/test_ops_orchestrator.py
git commit -m "feat: persist AI operations sessions"
```

### Task 2: Provider Chat Probe and Reasoning-Model Repair

**Files:**
- Create: `backend/app/services/ops_provider.py`
- Modify: `backend/app/services/providers.py`
- Modify: `backend/app/api/providers.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_ops_provider.py`
- Modify: `backend/tests/test_providers.py`

- [ ] **Step 1: Write failing response compatibility tests**

```python
def test_retries_when_reasoning_consumes_first_output_budget(respx_mock, provider_client):
    route = respx_mock.post("https://provider.invalid/v1/chat/completions")
    route.side_effect = [
        httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {
            "role": "assistant", "content": "", "reasoning_content": "analysis" * 100,
        }}]}),
        httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"action":"answer","summary":"healthy"}',
        }}]}),
    ]
    result = provider_client.complete(provider(), request_messages())
    assert result.action == "answer"
    assert route.calls[0].request.content != route.calls[1].request.content


def test_retries_without_response_format_when_provider_rejects_it(respx_mock, provider_client):
    route = respx_mock.post("https://provider.invalid/v1/chat/completions")
    route.side_effect = [
        httpx.Response(400, json={"error": {"message": "response_format unsupported"}}),
        httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"action":"answer","summary":"ok"}',
        }}]}),
    ]
    assert provider_client.complete(provider(), request_messages()).summary == "ok"
    assert "response_format" not in json.loads(route.calls[1].request.content)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_provider.py -v`

Expected: FAIL because `OpsProviderClient` is missing.

- [ ] **Step 3: Implement strict parse and bounded retry states**

```python
class AssistantTurn(BaseModel):
    action: Literal["tool", "plan", "answer", "question"]
    summary: str = Field(min_length=1, max_length=4000)
    tool: ReadOnlyToolRequest | None = None
    steps: list[ChangeStep] = Field(default_factory=list, max_length=20)


def _extract_content(payload: dict[str, Any]) -> tuple[str, str | None]:
    choice = payload["choices"][0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise IncompleteProviderResponse("Provider returned no final content")
    return content, choice.get("finish_reason")
```

Use an initial `max_tokens=2048`, then one compact repair request with `max_tokens=4096` when content is empty, `finish_reason=length`, JSON parsing fails, or schema validation fails. Never parse `reasoning_content` as a command. Retry without `response_format` only for a 4xx response that names that parameter. Limit provider body errors to 500 sanitized characters.

- [ ] **Step 4: Extend Provider test to verify the selected model**

Construct one `OpsProviderClient` in `create_app` and inject it into `ProviderService` and the later orchestrator. `ProviderService.test` must return and persist:

```python
{
    "status": "healthy",
    "connection": {"status": "healthy", "models_seen": 12},
    "default_model": {"status": "healthy", "model": provider.default_model},
}
```

If `/models` works but chat fails, overall status is `failed`, connection remains `healthy`, and default model contains a bounded actionable error. Run:

`python -m pytest backend/tests/test_ops_provider.py backend/tests/test_providers.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Provider compatibility**

```bash
git add backend/app/services/ops_provider.py backend/app/services/providers.py backend/app/api/providers.py backend/tests/test_ops_provider.py backend/tests/test_providers.py
git commit -m "fix: validate and repair AI provider responses"
```

### Task 3: Structured Read-Only Tool Registry

**Files:**
- Create: `backend/app/services/ops_tools.py`
- Modify: `backend/tests/test_ops_orchestrator.py`

- [ ] **Step 1: Write failing tool registry tests**

```python
def test_registry_executes_only_declared_read_only_tools(agent_client, database):
    registry = OpsToolRegistry(agent_client, database.session_factory)
    result = registry.execute(ReadOnlyToolRequest(name="host.memory", arguments={}))
    assert result.risk == "read_only"
    assert agent_client.calls == [("host.memory", {})]
    with pytest.raises(ValueError, match="not an automatic read-only tool"):
        registry.execute(ReadOnlyToolRequest(name="shell.execute", arguments={"command": "id"}))


def test_database_summary_excludes_secrets(registry, provider_with_secret):
    result = registry.execute(ReadOnlyToolRequest(name="manager.summary", arguments={}))
    dumped = json.dumps(result.output)
    assert provider_with_secret.encrypted_api_key not in dumped
    assert "Authorization" not in dumped
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_orchestrator.py -k registry -v`

Expected: FAIL because the registry is missing.

- [ ] **Step 3: Implement an explicit registry**

```python
AGENT_READ_TOOLS = {
    "host.memory", "host.disk", "host.gpu", "host.ports", "host.processes",
    "docker.list", "docker.inspect", "docker.logs", "docker.stats",
    "systemd.status", "systemd.journal",
}
MANAGER_READ_TOOLS = {"manager.summary", "manager.tasks", "manager.gateway"}


def execute(self, request: ReadOnlyToolRequest) -> ToolResult:
    if request.name in AGENT_READ_TOOLS:
        output = self.agent.call(request.name, request.arguments)
    elif request.name in MANAGER_READ_TOOLS:
        output = self._manager_tool(request)
    else:
        raise ValueError(f"{request.name} is not an automatic read-only tool")
    return ToolResult(risk="read_only", output=sanitize_and_bound(output, 30_000))
```

Arguments are Pydantic discriminated unions, not free-form dictionaries after parsing. Manager summaries select explicit non-secret columns only.

- [ ] **Step 4: Run registry tests**

Run: `python -m pytest backend/tests/test_ops_orchestrator.py -k "registry or summary" -v`

Expected: PASS.

- [ ] **Step 5: Commit tools**

```bash
git add backend/app/services/ops_tools.py backend/tests/test_ops_orchestrator.py
git commit -m "feat: add AI operations read-only tools"
```

### Task 4: Bounded Operations Orchestrator

**Files:**
- Create: `backend/app/services/ops_orchestrator.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_ops_orchestrator.py`

- [ ] **Step 1: Write failing loop and plan tests**

```python
def test_orchestrator_runs_read_tool_then_answers(orchestrator, scripted_provider):
    scripted_provider.turns = [
        AssistantTurn(action="tool", summary="Inspect memory", tool={"name": "host.memory", "arguments": {}}),
        AssistantTurn(action="answer", summary="Memory is healthy"),
    ]
    result = orchestrator.respond(session_id="session-1", prompt="Check memory", actor="admin")
    assert result.status == "answered"
    assert result.tool_runs == 1


def test_orchestrator_creates_pending_plan_without_executing(orchestrator, scripted_provider, agent):
    scripted_provider.turns = [AssistantTurn(
        action="plan", summary="Repair service",
        steps=[{"operation": "shell", "command": "systemctl restart demo", "cwd": "/",
                "timeout": 60, "reason": "Recover", "impact": "Brief outage",
                "rollback": "systemctl restart demo"}],
    )]
    result = orchestrator.respond(session_id="session-1", prompt="Repair demo", actor="admin")
    assert result.status == "approval_required"
    assert result.plan.status == "pending"
    assert agent.calls == []


def test_plan_rejects_known_secret_embedded_in_shell(orchestrator, scripted_provider):
    scripted_provider.turns = [AssistantTurn(
        action="plan", summary="Unsafe",
        steps=[{"operation": "shell", "command": "curl -H 'Authorization: Bearer known-secret' example.invalid",
                "cwd": "/", "timeout": 30, "reason": "test", "impact": "network",
                "rollback": "none"}],
    )]
    with pytest.raises(ValueError, match="secret material"):
        orchestrator.respond(session_id="session-1", prompt="test", actor="admin")


def test_automatic_tool_call_is_audited(orchestrator, scripted_provider, database):
    scripted_provider.turns = [
        AssistantTurn(action="tool", summary="Inspect", tool={"name": "host.memory", "arguments": {}}),
        AssistantTurn(action="answer", summary="Done"),
    ]
    orchestrator.respond(session_id="session-1", prompt="inspect", actor="admin")
    with database.session_factory() as db:
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "ops.tool.execute"))
        assert event.details["tool_name"] == "host.memory"
        assert "output" not in event.details
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_ops_orchestrator.py -k orchestrator -v`

Expected: FAIL because `OpsOrchestrator` is missing.

- [ ] **Step 3: Implement the bounded loop**

```python
def respond(self, *, session_id: str, prompt: str, actor: str) -> OpsResponseResult:
    self._append_message(session_id, "user", prompt)
    started = time.monotonic()
    for turn_index in range(self.max_tool_turns + 1):
        if time.monotonic() - started > self.max_total_seconds:
            return self._stop_with_limit(session_id, "time_limit")
        turn = self.provider.complete(self._provider(session_id), self._messages(session_id))
        if turn.action == "tool":
            if turn_index == self.max_tool_turns:
                return self._stop_with_limit(session_id, "tool_turn_limit")
            result = self.tools.execute(turn.tool)
            self._save_tool_result(session_id, turn.tool, result)
            continue
        if turn.action == "plan":
            return self._create_pending_plan(session_id, actor, turn)
        return self._save_answer(session_id, turn)
```

Set defaults to 6 tool turns, 180 total seconds, 30,000 characters per tool result, and 120,000 cumulative characters. Persist every turn before the next Provider request. A failed tool is persisted and returned to the Provider once; repeating the same failed tool/arguments stops the loop. Before storing a change plan, compare commands against decrypted configured secrets in memory; reject any command containing known secret material and never include the matching value in the error. Record `ops.tool.execute`, `ops.plan.create`, `ops.answer.create`, and limit/failure outcomes with sanitized argument/result summaries.

- [ ] **Step 4: Register the async response task**

Register `ops.respond` in `create_app`. The handler loads session/prompt IDs, calls the orchestrator, and returns session ID, message ID or plan ID, tool count, and status. Run:

`python -m pytest backend/tests/test_ops_orchestrator.py -v`

Expected: PASS, including loop, duplicate-call, output, and time limits.

- [ ] **Step 5: Commit orchestrator**

```bash
git add backend/app/services/ops_orchestrator.py backend/app/main.py backend/tests/test_ops_orchestrator.py
git commit -m "feat: orchestrate bounded AI operations sessions"
```

### Task 5: Session API and Legacy Diagnostic Compatibility

**Files:**
- Rewrite: `backend/app/api/diagnostics.py`
- Modify: `backend/app/services/diagnostics.py`
- Modify: `backend/tests/test_diagnostics.py`
- Modify: `docs/API.md`

- [ ] **Step 1: Write failing session API tests**

```python
def test_create_session_and_queue_message(authenticated_client, healthy_provider):
    created = authenticated_client.post("/api/diagnostics/sessions", json={
        "provider_id": healthy_provider.id, "title": "Repair gateway", "deployment_id": None,
    })
    assert created.status_code == 201
    session_id = created.json()["id"]
    response = authenticated_client.post(f"/api/diagnostics/sessions/{session_id}/messages", json={
        "content": "Inspect the gateway and repair it if required",
    })
    assert response.status_code == 202
    assert response.json()["type"] == "ops.respond"


def test_session_history_contains_tools_and_plan(authenticated_client, populated_ops_session):
    response = authenticated_client.get(f"/api/diagnostics/sessions/{populated_ops_session.id}")
    assert response.status_code == 200
    assert response.json()["messages"]
    assert response.json()["tool_runs"]
    assert response.json()["plans"][0]["status"] == "pending"
```

- [ ] **Step 2: Run API tests and verify RED**

Run: `python -m pytest backend/tests/test_diagnostics.py -k session -v`

Expected: FAIL with 404 because session routes are absent.

- [ ] **Step 3: Implement CRUD and message task endpoints**

Add:

- `GET /api/diagnostics/sessions`
- `POST /api/diagnostics/sessions`
- `GET /api/diagnostics/sessions/{session_id}`
- `POST /api/diagnostics/sessions/{session_id}/messages`
- `POST /api/diagnostics/{plan_id}/approve`
- `POST /api/diagnostics/{plan_id}/reject`

Session creation requires an enabled Provider. A healthy default-model probe makes it immediately eligible; for an upgraded legacy Provider with `last_test_status="healthy"` and no structured probe result, the first message performs the real chat probe before orchestration and persists the result. A failed structured probe blocks messages with an actionable 409 response. Message content is 1 to 10,000 characters. Creating a message task records an audit event but does not call the Provider in the request thread.

Keep `GET /api/diagnostics` returning legacy plans for Dashboard compatibility. Keep the old `POST /api/diagnostics` as a compatibility adapter that creates a session and queues its first message, returning `202` rather than performing a synchronous Provider call.

- [ ] **Step 4: Run diagnostic API tests**

Run: `python -m pytest backend/tests/test_diagnostics.py backend/tests/test_ops_orchestrator.py -v`

Expected: PASS, including legacy history and CSRF checks.

- [ ] **Step 5: Document and commit API**

```bash
git add backend/app/api/diagnostics.py backend/app/services/diagnostics.py backend/tests/test_diagnostics.py docs/API.md
git commit -m "feat: expose persistent AI operations sessions"
```

### Task 6: Execute Approved Shell Plans Through the Agent

**Files:**
- Modify: `backend/app/operations/executor.py`
- Modify: `backend/app/services/diagnostics.py`
- Modify: `backend/tests/test_diagnostics.py`

- [ ] **Step 1: Write failing approval/execution tests**

```python
def test_shell_step_is_never_executed_before_approval(operation_executor, pending_shell_plan, agent):
    with pytest.raises(ValueError, match="not approved"):
        operation_executor.handler(FakeContext(), {"plan_id": pending_shell_plan.id})
    assert agent.calls == []


def test_approved_shell_step_polls_agent_and_records_result(operation_executor, approved_shell_plan, agent):
    agent.script_job(output="fixed\n", exit_code=0)
    result = operation_executor.handler(FakeContext(), {"plan_id": approved_shell_plan.id})
    assert result["steps"][0]["status"] == "succeeded"
    action, parameters, approval = agent.calls[0]
    assert action == "shell.execute"
    assert parameters["command"] == "systemctl restart demo"
    assert approval["plan_id"] == approved_shell_plan.id
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_diagnostics.py -k "shell_step or approved_shell" -v`

Expected: FAIL because Shell steps are currently sanitized to non-executable.

- [ ] **Step 3: Add immutable approved Shell execution**

At plan creation, validate Shell steps into this JSON shape and compute a SHA-256 digest of canonical steps:

```python
{
    "id": "step-uuid", "operation": "shell", "command": "systemctl restart demo",
    "cwd": "/", "timeout": 60, "reason": "Recover", "impact": "Brief outage",
    "rollback": "systemctl restart demo", "executable": True,
}
```

Store the digest in `result_json["approval_digest"]` when approving. The executor recomputes it before each step and rejects mutation. For Shell, call Agent `shell.execute` with plan/step/approver metadata, poll `job.get`, use `output_offset` and `truncated_before` to append only new redacted output to `TaskContext.update`, honor cancellation through `job.cancel`, and stop the plan at the first nonzero exit code. Record a separate audit event for every started, cancelled, succeeded, or failed Shell step; audit details contain the command digest and bounded summary, not raw output or credentials.

- [ ] **Step 4: Run executor and task tests**

Run: `python -m pytest backend/tests/test_diagnostics.py backend/tests/test_tasks.py -v`

Expected: PASS; cancellation stops the Agent job, failed steps leave the plan `failed`, and later steps remain unexecuted.

- [ ] **Step 5: Commit approved execution**

```bash
git add backend/app/operations/executor.py backend/app/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: execute approved AI shell plans"
```

### Task 7: Operations Conversation Frontend

**Files:**
- Modify: `frontend/src/api/types.ts`
- Rewrite: `frontend/src/pages/DiagnosticsPage.tsx`
- Create: `frontend/src/components/OpsConversation.tsx`
- Create: `frontend/src/components/OpsExecutionOutput.tsx`
- Modify: `frontend/src/components/ApprovalPanel.tsx`
- Create: `frontend/src/pages/DiagnosticsPage.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Write failing conversation tests**

```tsx
it('creates a session and sends a task-backed message', async () => {
  renderDiagnosticsPage()
  await user.selectOptions(screen.getByLabelText('在线 AI 服务'), 'provider-1')
  await user.type(screen.getByLabelText('运维请求'), '检查网关异常')
  await user.click(screen.getByRole('button', { name: '发送' }))
  expect(api.post).toHaveBeenNthCalledWith(1, '/api/diagnostics/sessions', expect.any(Object))
  expect(api.post).toHaveBeenNthCalledWith(2, expect.stringMatching(/\/messages$/), {
    content: '检查网关异常',
  })
})

it('shows automatic read tools and requires approval for shell', async () => {
  renderDiagnosticsPage({ session: sessionWithToolAndPlan })
  expect(screen.getByText('host.memory')).toBeInTheDocument()
  expect(screen.getByText('自动执行 · 只读')).toBeInTheDocument()
  expect(screen.getByText('systemctl restart demo')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '批准执行' })).toBeInTheDocument()
})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pnpm --dir frontend test -- DiagnosticsPage.test.tsx`

Expected: FAIL because the current page has only a one-shot form and plan list.

- [ ] **Step 3: Implement responsive conversation and timeline**

Define typed `OpsSession`, `OpsMessage`, and `OpsToolRun`. The page keeps a compact Provider/deployment selector, session list, message composer, and polling query for the selected session while an `ops.respond` task is active. `OpsConversation` renders user/assistant messages, collapsible tool invocations, bounded results, and linked approval panels in time order.

`OpsExecutionOutput` uses a fixed min/max height and scroll area:

```tsx
<pre className="ops-terminal" aria-label="执行输出">
  <code>{output || '等待执行输出...'}</code>
</pre>
```

`ApprovalPanel` renders exact command, working directory, timeout, impact, validation, and rollback. It never hides the command behind an unlabeled icon. Mobile uses one column and a drawer for session history; desktop uses a restrained two-column grid without nested cards.

- [ ] **Step 4: Run frontend checks**

Run: `pnpm --dir frontend test -- DiagnosticsPage.test.tsx && pnpm --dir frontend lint && pnpm --dir frontend build`

Expected: PASS with no TypeScript errors.

- [ ] **Step 5: Commit operations UI**

```bash
git add frontend/src/api/types.ts frontend/src/pages/DiagnosticsPage.tsx frontend/src/components/OpsConversation.tsx frontend/src/components/OpsExecutionOutput.tsx frontend/src/components/ApprovalPanel.tsx frontend/src/pages/DiagnosticsPage.test.tsx frontend/src/styles.css
git commit -m "feat: add conversational AI operations UI"
```

### Task 8: Provider Probe UI

**Files:**
- Modify: `frontend/src/pages/ProvidersPage.tsx`
- Create: `frontend/src/pages/ProvidersPage.test.tsx`
- Modify: `frontend/src/api/types.ts`

- [ ] **Step 1: Write a failing partial-health test**

```tsx
it('distinguishes connection health from default model failure', async () => {
  mockProviderTest({
    status: 'failed',
    connection: { status: 'healthy', models_seen: 8 },
    default_model: { status: 'failed', model: 'missing', detail: 'model not found' },
  })
  renderProvidersPage()
  await user.click(screen.getByRole('button', { name: '测试连接' }))
  expect(await screen.findByText('API 连接正常')).toBeInTheDocument()
  expect(screen.getByText('默认模型不可用')).toBeInTheDocument()
  expect(screen.getByText('model not found')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run test and verify RED**

Run: `pnpm --dir frontend test -- ProvidersPage.test.tsx`

Expected: FAIL because the page only shows one status.

- [ ] **Step 3: Render structured probe results**

Add quiet status rows for API connection and default model. Use `Alert` only for the failed part, keep bounded technical detail collapsible, and invalidate both providers and diagnostics session eligibility after a successful retest.

- [ ] **Step 4: Run frontend Provider tests**

Run: `pnpm --dir frontend test -- ProvidersPage.test.tsx && pnpm --dir frontend lint && pnpm --dir frontend build`

Expected: PASS.

- [ ] **Step 5: Commit Provider UI**

```bash
git add frontend/src/pages/ProvidersPage.tsx frontend/src/pages/ProvidersPage.test.tsx frontend/src/api/types.ts
git commit -m "feat: show AI provider model readiness"
```

### Task 9: Physically Clear Alerts and Diagnostics History

**Files:**
- Modify: `backend/app/api/settings.py`
- Modify: `backend/tests/test_settings.py`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/pages/SettingsPage.tsx`
- Create: `frontend/src/pages/SettingsPage.test.tsx`

- [ ] **Step 1: Write failing transactional cleanup tests**

Cover exact confirmation, CSRF, active `ops.respond`/`operation.execute` task conflicts, approved-plan conflicts, physical deletion counts, cascaded session data, related audit removal, unaffected domain records, and rollback on failure.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest backend/tests/test_settings.py -k history -v`

Expected: FAIL because the cleanup endpoint does not exist.

- [ ] **Step 3: Implement one-transaction physical deletion**

Add `DELETE /api/settings/alerts-diagnostics-history` with body `{ "confirmation": "清除历史记录" }`. Delete failed tasks, operation plans, operations sessions/messages/tool runs and related diagnostic/operations audit details with SQLAlchemy delete statements in dependency-safe order. Reject the entire request while an operations task or approved unfinished plan is active. Insert only one `maintenance.history.clear` audit event containing aggregate counts, then commit once.

- [ ] **Step 4: Add the Settings danger action**

Render an unframed danger section with a destructive button and Ant Design modal. Require the exact confirmation phrase, explain which records are and are not deleted, and invalidate `tasks`, `diagnostics`, `ops-sessions`, `audit`, and dashboard-related queries after success. Keep the control usable at mobile widths and under dark mode.

- [ ] **Step 5: Run backend and frontend checks and commit**

Run: `python -m pytest backend/tests/test_settings.py -v && pnpm --dir frontend test -- SettingsPage.test.tsx && pnpm --dir frontend lint && pnpm --dir frontend build`

```bash
git add backend/app/api/settings.py backend/tests/test_settings.py frontend/src/api/types.ts frontend/src/pages/SettingsPage.tsx frontend/src/pages/SettingsPage.test.tsx
git commit -m "feat: physically clear operations history"
```

### Task 10: Full Verification, Visual Review, and DGX Deployment

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/TROUBLESHOOTING.md`

- [ ] **Step 1: Run complete backend checks**

Run: `python -m ruff check backend host_agent && python -m pytest backend/tests -q`

Expected: lint succeeds and the complete backend suite passes.

- [ ] **Step 2: Run complete frontend checks**

Run: `pnpm --dir frontend test && pnpm --dir frontend lint && pnpm --dir frontend build`

Expected: all frontend tests, lint, and production build pass.

- [ ] **Step 3: Perform Playwright visual checks**

Start the local app with fixture data, then capture Diagnostics and Providers pages at 1440x900 and 390x844 in light and dark modes. Verify the command, buttons, terminal output, session selector, and composer do not overlap or overflow; verify the terminal has stable dimensions while output changes.

- [ ] **Step 4: Deploy to DGX with a database and Compose backup**

Back up `data/manager.db`, `.env`, current Compose config, and current image ID. Install/verify the Agent first, build the ARM64 image, run Alembic upgrade, recreate only the manager container, and verify `/api/health`, `/api/ops-agent/health`, `/v1/models`, and an existing chat completion before testing AI operations.

- [ ] **Step 5: Verify the real configured Provider and approval boundary**

Run a read-only request that triggers memory, GPU, Docker, and service tools and verify they execute automatically. Ask the assistant to create a harmless approved Shell plan such as writing and then removing a file under a dedicated test directory. Confirm no file exists before approval, execute after approval, inspect live output/audit, and remove the test artifact through a separately approved step.

- [ ] **Step 6: Document rollback and commit**

Document Provider truncation symptoms, Agent unavailable behavior, stuck jobs, rejected approvals, and rollback order.

```bash
git add docs/ARCHITECTURE.md docs/TROUBLESHOOTING.md
git commit -m "docs: document conversational AI operations"
```
