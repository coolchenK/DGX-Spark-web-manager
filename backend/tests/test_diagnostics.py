import json
from datetime import UTC, datetime, timedelta

import pytest
from app.db import Database
from app.models import (
    AuditEvent,
    Deployment,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
    TaskRecord,
)
from app.operations.executor import OperationExecutor
from app.services import deployments as deployment_service
from app.services import diagnostics
from app.services.diagnostics import parse_diagnostic_content, sanitize_steps
from app.services.ops_provider import AssistantTurn
from sqlalchemy import select


class ExecutorContext:
    def __init__(self):
        self.updates = []

    def check_control(self):
        return None

    def update(self, **values):
        self.updates.append(values)


class NoopDiscoveryService:
    def scan_all(self, _db):
        return {}


def add_executor_plan(database, *, operation, deployment_id):
    from app.services.diagnostics import operation_plan_digest

    steps = [
        {
            "operation": operation,
            "deployment_id": deployment_id,
            "reason": "Test lifecycle action",
            "executable": True,
        }
    ]
    with database.session_factory() as db:
        plan = OperationPlan(
            summary="Lifecycle action",
            diagnosis="Exercise the deployment",
            risk="low",
            status="approved",
            steps=steps,
            approved_by="admin",
            approved_at=datetime.now(UTC),
            result_json={"approval_digest": operation_plan_digest(steps)},
        )
        db.add(plan)
        db.commit()
        return plan.id


def add_executor_deployment(database, *, container_id="7a" * 32):
    with database.session_factory() as db:
        deployment = Deployment(
            name="managed",
            runtime="vllm",
            container_id=container_id,
            container_name="shared-name",
            endpoint_url="http://127.0.0.1:8100",
            api_model_name="managed",
            status="running",
            health="healthy",
            managed=True,
        )
        db.add(deployment)
        db.commit()
        return deployment.id


@pytest.mark.parametrize(
    ("operation", "action"),
    [
        ("start_deployment", "start"),
        ("stop_deployment", "stop"),
        ("restart_deployment", "restart"),
    ],
)
def test_operation_executor_passes_container_snapshot(tmp_path, operation, action):
    database = Database(f"sqlite:///{tmp_path / f'{action}.db'}")
    database.create_schema()
    deployment_id = add_executor_deployment(database)
    plan_id = add_executor_plan(
        database,
        operation=operation,
        deployment_id=deployment_id,
    )
    calls = []

    class DeploymentService:
        def action_handler(self, _context, payload):
            calls.append(payload)
            return {"status": "ok"}

    executor = OperationExecutor(
        session_factory=database.session_factory,
        deployment_service=DeploymentService(),
        discovery_service=NoopDiscoveryService(),
    )

    executor.handler(ExecutorContext(), {"plan_id": plan_id})

    assert calls == [
        {
            "deployment_id": deployment_id,
            "action": action,
            "expected_container_id": "7a" * 32,
            "expected_container_name": "shared-name",
        }
    ]


def test_operation_executor_rejects_missing_deployment(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'missing.db'}")
    database.create_schema()
    plan_id = add_executor_plan(
        database,
        operation="stop_deployment",
        deployment_id="missing-deployment",
    )

    class DeploymentService:
        def action_handler(self, _context, _payload):
            raise AssertionError("missing deployment must not reach action handler")

    executor = OperationExecutor(
        session_factory=database.session_factory,
        deployment_service=DeploymentService(),
        discovery_service=NoopDiscoveryService(),
    )

    with pytest.raises(ValueError, match="Deployment was not found; retry the operation"):
        executor.handler(ExecutorContext(), {"plan_id": plan_id})


def test_operation_executor_does_not_rebind_snapshot_at_handler_entry(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'rebound.db'}")
    database.create_schema()
    container_a_id = "8a" * 32
    container_b_id = "8b" * 32
    deployment_id = add_executor_deployment(database, container_id=container_a_id)
    plan_id = add_executor_plan(
        database,
        operation="stop_deployment",
        deployment_id=deployment_id,
    )
    captured = {}
    real_service = deployment_service.DeploymentService(
        adapters={},
        session_factory=database.session_factory,
        model_roots=(tmp_path,),
    )

    class DeploymentService:
        def action_handler(self, context, payload):
            captured.update(payload)
            with database.session_factory() as db:
                deployment = db.get(Deployment, deployment_id)
                deployment.container_id = container_b_id
                db.commit()
            return real_service.action_handler(context, payload)

    executor = OperationExecutor(
        session_factory=database.session_factory,
        deployment_service=DeploymentService(),
        discovery_service=NoopDiscoveryService(),
    )

    with pytest.raises(ValueError, match="identity changed concurrently"):
        executor.handler(ExecutorContext(), {"plan_id": plan_id})

    assert captured["expected_container_id"] == container_a_id
    assert captured["expected_container_name"] == "shared-name"
    with database.session_factory() as db:
        deployment = db.get(Deployment, deployment_id)
        assert deployment.container_id == container_b_id
        assert deployment.status == "running"
        assert deployment.health == "healthy"


def test_diagnostic_parser_accepts_json_fence():
    result = parse_diagnostic_content(
        '```json\n{"summary":"High memory", "diagnosis":"Two models", '
        '"risk":"low", "steps":[]}\n```'
    )

    assert result["summary"] == "High memory"


def test_unknown_and_shell_operations_are_not_executable():
    steps = sanitize_steps(
        [
            {
                "operation": "restart_deployment",
                "deployment_id": "dep-1",
                "reason": "Recover health",
            },
            {"operation": "shell", "command": "rm -rf /", "reason": "unsafe"},
        ],
        known_deployment_ids={"dep-1"},
    )

    assert steps[0]["executable"] is True
    assert steps[1]["executable"] is False
    assert "command" not in steps[1]


def test_compact_string_steps_are_safely_normalized():
    steps = sanitize_steps(
        ["Rescan Inventory", "run arbitrary shell"],
        known_deployment_ids=set(),
    )

    assert steps[0]["operation"] == "rescan_inventory"
    assert steps[0]["executable"] is True
    assert steps[1]["operation"] == "explain_only"
    assert steps[1]["executable"] is False


def test_diagnostic_provider_request_limits_generated_output():
    build = getattr(diagnostics, "build_diagnostic_request", None)
    assert callable(build)

    payload = build(
        model="ops-model",
        prompt="Inspect the service",
        actual_context={
            "system": {},
            "deployments": [{"id": "dep-1", "logs": "a" * 3000}],
        },
    )

    assert payload["max_tokens"] == 512
    assert payload["response_format"] == {"type": "json_object"}
    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload["observed_context"]["deployments"][0]["logs"] == "a" * 2500


def test_plan_must_be_approved_before_execution(authenticated_client):
    from app.models import OperationPlan

    with authenticated_client.app.state.database.session_factory() as db:
        plan = OperationPlan(
            summary="Rescan inventory",
            diagnosis="Inventory may be stale",
            risk="low",
            steps=[{"operation": "rescan_inventory", "reason": "Refresh", "executable": True}],
        )
        db.add(plan)
        db.commit()
        plan_id = plan.id

    response = authenticated_client.post(f"/api/diagnostics/{plan_id}/approve")

    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "running", "succeeded"}
    with authenticated_client.app.state.database.session_factory() as db:
        approved = db.get(OperationPlan, plan_id)
        assert approved.status in {"approved", "executing", "completed"}
        assert approved.approved_by == "admin"
        assert len(approved.result_json["approval_digest"]) == 64


def _add_ops_provider(
    authenticated_client,
    *,
    last_test_status="healthy",
    last_test_result=None,
    enabled=True,
):
    if last_test_result is None:
        last_test_result = {
            "status": "healthy",
            "connection": {"status": "healthy", "models_seen": 1},
            "default_model": {"status": "healthy", "model": "ops-model"},
        }
    with authenticated_client.app.state.database.session_factory() as db:
        provider = Provider(
            name=f"ops-provider-{last_test_status}-{len(last_test_result)}",
            base_url="https://provider.example/v1",
            default_model="ops-model",
            encrypted_api_key=authenticated_client.app.state.secret_box.encrypt("provider-key"),
            enabled=enabled,
            last_test_status=last_test_status,
            last_test_result=last_test_result,
        )
        db.add(provider)
        db.commit()
        return provider.id


def test_create_session_and_queue_message_without_calling_provider_in_request(
    authenticated_client, monkeypatch
):
    authenticated_client.app.state.task_engine.stop()
    provider_id = _add_ops_provider(authenticated_client)

    def unexpected_complete(*_args, **_kwargs):
        raise AssertionError("message endpoint must not call the Provider")

    monkeypatch.setattr(
        authenticated_client.app.state.ops_provider_client,
        "complete",
        unexpected_complete,
    )

    created = authenticated_client.post(
        "/api/diagnostics/sessions",
        json={
            "provider_id": provider_id,
            "title": "Repair gateway",
            "deployment_id": None,
        },
    )

    assert created.status_code == 201
    session_id = created.json()["id"]
    response = authenticated_client.post(
        f"/api/diagnostics/sessions/{session_id}/messages",
        json={"content": "Inspect the gateway and repair it if required"},
    )

    assert response.status_code == 202
    assert response.json()["type"] == "ops.respond"
    assert response.json()["status"] == "queued"
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        assert task.input_json == {
            "session_id": session_id,
            "prompt": "Inspect the gateway and repair it if required",
            "actor": "admin",
        }
        audit = db.scalar(
            select(AuditEvent).where(AuditEvent.action == "ops.respond.queue")
        )
        assert audit.resource_id == session_id
        assert audit.details["task_id"] == task.id
        assert "Inspect the gateway" not in str(audit.details)


def test_session_history_contains_messages_tools_and_linked_plan(authenticated_client):
    provider_id = _add_ops_provider(authenticated_client)
    started_at = datetime(2026, 8, 18, tzinfo=UTC)
    with authenticated_client.app.state.database.session_factory() as db:
        session = OpsSession(
            title="Repair gateway",
            provider_id=provider_id,
            requested_by="admin",
        )
        plan = OperationPlan(
            provider_id=provider_id,
            summary="Restart gateway",
            diagnosis="Gateway is unhealthy",
            risk="medium",
            steps=[{"operation": "shell", "command": "systemctl restart gateway"}],
        )
        db.add_all([session, plan])
        db.flush()
        db.add_all(
            [
                OpsMessage(
                    session_id=session.id,
                    role="user",
                    content="Inspect gateway",
                    created_at=started_at,
                ),
                OpsMessage(
                    session_id=session.id,
                    role="assistant",
                    content="Approval is required",
                    operation_plan_id=plan.id,
                    metadata_json={"action": "plan", "plan_id": plan.id},
                    created_at=started_at + timedelta(seconds=1),
                ),
                OpsToolRun(
                    session_id=session.id,
                    tool_name="systemd.status",
                    status="succeeded",
                    arguments_json={"unit": "gateway.service"},
                    result_json={"active": False},
                ),
            ]
        )
        db.commit()
        session_id = session.id

    response = authenticated_client.get(f"/api/diagnostics/sessions/{session_id}")

    assert response.status_code == 200
    body = response.json()
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["tool_runs"][0]["tool_name"] == "systemd.status"
    assert body["tool_runs"][0]["result"] == {"active": False}
    assert body["plans"][0]["id"] == plan.id
    assert body["plans"][0]["status"] == "pending"


def test_failed_structured_provider_probe_blocks_session_message(authenticated_client):
    authenticated_client.app.state.task_engine.stop()
    provider_id = _add_ops_provider(
        authenticated_client,
        last_test_status="failed",
        last_test_result={
            "status": "failed",
            "connection": {"status": "healthy", "models_seen": 1},
            "default_model": {
                "status": "failed",
                "model": "ops-model",
                "error": "model not found provider-key",
            },
        },
    )
    created = authenticated_client.post(
        "/api/diagnostics/sessions",
        json={"provider_id": provider_id, "title": "Blocked provider"},
    )

    response = authenticated_client.post(
        f"/api/diagnostics/sessions/{created.json()['id']}/messages",
        json={"content": "Inspect"},
    )

    assert response.status_code == 409
    assert "test" in response.json()["detail"].lower()
    assert "provider-key" not in response.text
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.scalars(select(TaskRecord).where(TaskRecord.type == "ops.respond")).all() == []


def test_legacy_healthy_provider_is_reprobed_before_first_message(authenticated_client):
    authenticated_client.app.state.task_engine.stop()
    provider_id = _add_ops_provider(
        authenticated_client,
        last_test_status="healthy",
        last_test_result={},
    )
    calls = []

    class ProbeClient:
        def list_models(self, provider):
            calls.append(("models", provider.id))
            return ["ops-model"]

        def complete(self, provider, messages):
            calls.append(("complete", provider.id, messages[-1]["content"]))
            return AssistantTurn(action="answer", summary="probe ok")

    authenticated_client.app.state.provider_service.ops_provider_client = ProbeClient()
    created = authenticated_client.post(
        "/api/diagnostics/sessions",
        json={"provider_id": provider_id, "title": "Legacy provider"},
    )
    response = authenticated_client.post(
        f"/api/diagnostics/sessions/{created.json()['id']}/messages",
        json={"content": "Inspect"},
    )

    assert response.status_code == 202
    assert [call[0] for call in calls] == ["models", "complete"]
    with authenticated_client.app.state.database.session_factory() as db:
        provider = db.get(Provider, provider_id)
        assert provider.last_test_result["default_model"]["status"] == "healthy"


def test_legacy_diagnostic_create_queues_first_session_message(authenticated_client):
    authenticated_client.app.state.task_engine.stop()
    provider_id = _add_ops_provider(authenticated_client)

    response = authenticated_client.post(
        "/api/diagnostics",
        json={
            "provider_id": provider_id,
            "deployment_id": None,
            "prompt": "Inspect memory pressure",
        },
    )

    assert response.status_code == 202
    assert response.json()["type"] == "ops.respond"
    with authenticated_client.app.state.database.session_factory() as db:
        task = db.get(TaskRecord, response.json()["id"])
        session = db.get(OpsSession, task.input_json["session_id"])
        assert session.title == "Inspect memory pressure"


def test_session_mutation_requires_csrf_and_enabled_provider(authenticated_client):
    provider_id = _add_ops_provider(authenticated_client, enabled=False)
    csrf = authenticated_client.headers.pop("X-CSRF-Token")
    try:
        forbidden = authenticated_client.post(
            "/api/diagnostics/sessions",
            json={"provider_id": provider_id, "title": "Forbidden"},
        )
    finally:
        authenticated_client.headers["X-CSRF-Token"] = csrf

    assert forbidden.status_code == 403
    disabled = authenticated_client.post(
        "/api/diagnostics/sessions",
        json={"provider_id": provider_id, "title": "Disabled"},
    )
    assert disabled.status_code == 404


@pytest.mark.parametrize("content", ["   ", "x" * 10_001])
def test_session_message_enforces_content_bounds(authenticated_client, content):
    authenticated_client.app.state.task_engine.stop()
    provider_id = _add_ops_provider(authenticated_client)
    created = authenticated_client.post(
        "/api/diagnostics/sessions",
        json={"provider_id": provider_id, "title": "Bounds"},
    )

    response = authenticated_client.post(
        f"/api/diagnostics/sessions/{created.json()['id']}/messages",
        json={"content": content},
    )

    assert response.status_code == 422


def test_session_rejects_second_active_response_task(authenticated_client):
    authenticated_client.app.state.task_engine.stop()
    provider_id = _add_ops_provider(authenticated_client)
    created = authenticated_client.post(
        "/api/diagnostics/sessions",
        json={"provider_id": provider_id, "title": "One at a time"},
    )
    path = f"/api/diagnostics/sessions/{created.json()['id']}/messages"

    first = authenticated_client.post(path, json={"content": "First"})
    second = authenticated_client.post(path, json={"content": "Second"})

    assert first.status_code == 202
    assert second.status_code == 409
    assert "active response task" in second.json()["detail"]


def test_list_sessions_is_newest_first(authenticated_client):
    provider_id = _add_ops_provider(authenticated_client)
    for title in ("First", "Second"):
        response = authenticated_client.post(
            "/api/diagnostics/sessions",
            json={"provider_id": provider_id, "title": title},
        )
        assert response.status_code == 201

    response = authenticated_client.get("/api/diagnostics/sessions")

    assert response.status_code == 200
    assert [item["title"] for item in response.json()[:2]] == ["Second", "First"]


def test_reject_plan_releases_linked_operations_session(authenticated_client):
    provider_id = _add_ops_provider(authenticated_client)
    with authenticated_client.app.state.database.session_factory() as db:
        session = OpsSession(
            title="Reject repair",
            provider_id=provider_id,
            status="approval_required",
        )
        plan = OperationPlan(
            summary="Risky repair",
            diagnosis="Needs approval",
            risk="high",
            steps=[],
            status="pending",
        )
        db.add_all([session, plan])
        db.flush()
        db.add(
            OpsMessage(
                session_id=session.id,
                role="assistant",
                content="Approval required",
                operation_plan_id=plan.id,
            )
        )
        db.commit()
        session_id, plan_id = session.id, plan.id

    response = authenticated_client.post(f"/api/diagnostics/{plan_id}/reject")

    assert response.status_code == 200
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.get(OpsSession, session_id).status == "active"


def _job(
    *,
    status,
    output="",
    output_offset=0,
    truncated_before=0,
    exit_code=None,
    error=None,
):
    return {
        "job_id": "agent-job-1",
        "status": status,
        "output": output,
        "output_offset": output_offset,
        "truncated_before": truncated_before,
        "exit_code": exit_code,
        "started": 1.0,
        "finished": 2.0 if status not in {"queued", "running"} else None,
        "error": error,
    }


class ScriptedAgent:
    def __init__(self, polls):
        self.polls = list(polls)
        self.calls = []

    def call(self, action, parameters, *, approval=None, timeout_seconds=None):
        self.calls.append((action, parameters, approval, timeout_seconds))
        if action == "shell.execute":
            return _job(status="queued")
        if action == "job.get":
            return self.polls.pop(0)
        if action == "job.cancel":
            return _job(status="cancelled")
        raise AssertionError(f"unexpected Agent action: {action}")


def _add_shell_plan(database, *, status="approved", commands=None):
    from app.services.diagnostics import operation_plan_digest

    commands = commands or ["systemctl restart demo"]
    steps = [
        {
            "id": f"step-{index}",
            "operation": "shell",
            "command": command,
            "cwd": "/",
            "timeout": 60,
            "reason": "Recover service",
            "impact": "Brief outage",
            "rollback": "systemctl restart demo",
            "executable": True,
        }
        for index, command in enumerate(commands, start=1)
    ]
    with database.session_factory() as db:
        plan = OperationPlan(
            summary="Repair service",
            diagnosis="Service is unhealthy",
            risk="high",
            status=status,
            steps=steps,
            approved_by="admin" if status == "approved" else None,
            approved_at=datetime.now(UTC) if status == "approved" else None,
            result_json=(
                {"approval_digest": operation_plan_digest(steps)}
                if status == "approved"
                else {}
            ),
        )
        db.add(plan)
        db.commit()
        return plan.id


def _shell_executor(database, agent, *, secret_box=None):
    return OperationExecutor(
        session_factory=database.session_factory,
        deployment_service=None,
        discovery_service=None,
        agent_client=agent,
        secret_box=secret_box,
        poll_interval_seconds=0,
    )


def test_shell_step_is_never_executed_before_approval(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'pending-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(database, status="pending")
    agent = ScriptedAgent([])

    with pytest.raises(ValueError, match="not approved"):
        _shell_executor(database, agent).handler(
            ExecutorContext(),
            {"plan_id": plan_id},
        )

    assert agent.calls == []


def test_approved_shell_step_polls_agent_and_records_result(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'approved-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(database)
    with database.session_factory() as db:
        session = OpsSession(title="Approved shell", status="approval_required")
        db.add(session)
        db.flush()
        db.add(
            OpsMessage(
                session_id=session.id,
                role="assistant",
                content="Approval required",
                operation_plan_id=plan_id,
            )
        )
        db.commit()
        session_id = session.id
    output = "fixed\n"
    agent = ScriptedAgent(
        [
            _job(
                status="succeeded",
                output=output,
                output_offset=len(output.encode()),
                exit_code=0,
            )
        ]
    )
    context = ExecutorContext()

    result = _shell_executor(database, agent).handler(
        context,
        {"plan_id": plan_id},
    )

    assert result["steps"][0]["status"] == "succeeded"
    assert result["steps"][0]["output"] == output
    action, parameters, approval, _timeout = agent.calls[0]
    assert action == "shell.execute"
    assert parameters == {
        "command": "systemctl restart demo",
        "cwd": "/",
        "timeout": 60,
    }
    assert approval["plan_id"] == plan_id
    assert approval["step_id"] == "step-1"
    assert approval["approved_by"] == "admin"
    assert output.strip() in str(context.updates)
    with database.session_factory() as db:
        plan = db.get(OperationPlan, plan_id)
        assert plan.status == "completed"
        assert db.get(OpsSession, session_id).status == "active"
        events = list(
            db.scalars(
                select(AuditEvent)
                .where(AuditEvent.resource_id == plan_id)
                .order_by(AuditEvent.created_at)
            )
        )
        assert [event.action for event in events] == [
            "ops.shell.start",
            "ops.shell.succeed",
        ]
        assert "systemctl restart demo" not in str([event.details for event in events])
        assert output.strip() not in str([event.details for event in events])


def test_approved_shell_plan_rejects_step_mutation_before_agent_call(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'mutated-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(database)
    with database.session_factory() as db:
        plan = db.get(OperationPlan, plan_id)
        changed = list(plan.steps)
        changed[0] = {**changed[0], "command": "systemctl stop demo"}
        plan.steps = changed
        db.commit()
    agent = ScriptedAgent([])

    with pytest.raises(ValueError, match="changed after approval"):
        _shell_executor(database, agent).handler(
            ExecutorContext(),
            {"plan_id": plan_id},
        )

    assert agent.calls == []
    with database.session_factory() as db:
        assert db.get(OperationPlan, plan_id).status == "failed"


def test_failed_shell_step_stops_later_steps(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'failed-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(
        database,
        commands=["false", "systemctl restart should-not-run"],
    )
    agent = ScriptedAgent(
        [
            _job(
                status="failed",
                output="failed\n",
                output_offset=len(b"failed\n"),
                exit_code=1,
                error="command failed",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="Shell step failed"):
        _shell_executor(database, agent).handler(
            ExecutorContext(),
            {"plan_id": plan_id},
        )

    assert [call[0] for call in agent.calls].count("shell.execute") == 1
    with database.session_factory() as db:
        plan = db.get(OperationPlan, plan_id)
        assert plan.status == "failed"
        assert len(plan.result_json["steps"]) == 1
        assert plan.result_json["steps"][0]["status"] == "failed"


def test_shell_task_cancellation_cancels_agent_job(tmp_path):
    from app.tasks.engine import TaskCancelled

    database = Database(f"sqlite:///{tmp_path / 'cancel-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(database)
    agent = ScriptedAgent([_job(status="running")])

    class CancellingContext(ExecutorContext):
        def __init__(self):
            super().__init__()
            self.checks = 0

        def check_control(self):
            self.checks += 1
            if self.checks >= 2:
                raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        _shell_executor(database, agent).handler(
            CancellingContext(),
            {"plan_id": plan_id},
        )

    assert [call[0] for call in agent.calls] == ["shell.execute", "job.cancel"]
    with database.session_factory() as db:
        plan = db.get(OperationPlan, plan_id)
        assert plan.status == "failed"
        event = db.scalar(
            select(AuditEvent).where(
                AuditEvent.resource_id == plan_id,
                AuditEvent.action == "ops.shell.cancel",
            )
        )
        assert event is not None


def test_shell_polling_uses_offsets_and_redacts_secrets_split_across_chunks(tmp_path):
    from app.security import SecretBox

    database = Database(f"sqlite:///{tmp_path / 'stream-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(database)
    secret_box = SecretBox("test-secret-key-with-at-least-32-characters")
    with database.session_factory() as db:
        db.add(
            Provider(
                name="secret-provider",
                base_url="https://provider.example/v1",
                default_model="ops-model",
                encrypted_api_key=secret_box.encrypt("provider-key"),
            )
        )
        db.commit()
    first = "prefix provider-"
    second = "key suffix\n"
    agent = ScriptedAgent(
        [
            _job(
                status="running",
                output=first,
                output_offset=len(first.encode()),
            ),
            _job(
                status="succeeded",
                output=second,
                truncated_before=len(first.encode()),
                output_offset=len((first + second).encode()),
                exit_code=0,
            ),
        ]
    )
    context = ExecutorContext()

    result = _shell_executor(
        database,
        agent,
        secret_box=secret_box,
    ).handler(context, {"plan_id": plan_id})

    rendered = result["steps"][0]["output"] + str(context.updates)
    assert "provider-key" not in rendered
    assert "[REDACTED]" in rendered
    get_calls = [call for call in agent.calls if call[0] == "job.get"]
    assert [call[1]["offset"] for call in get_calls] == [0, len(first.encode())]


def test_shell_polling_marks_agent_output_retention_gaps(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'truncated-shell.db'}")
    database.create_schema()
    plan_id = _add_shell_plan(database)
    agent = ScriptedAgent(
        [
            _job(status="running", output="abc", output_offset=3),
            _job(
                status="succeeded",
                output="fg",
                truncated_before=5,
                output_offset=7,
                exit_code=0,
            ),
        ]
    )

    result = _shell_executor(database, agent).handler(
        ExecutorContext(),
        {"plan_id": plan_id},
    )

    output = result["steps"][0]["output"]
    assert output.startswith("abc[Earlier Agent output was truncated]")
    assert output.endswith("fg")
