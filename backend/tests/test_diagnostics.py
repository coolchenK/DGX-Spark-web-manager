import json

import pytest
from app.db import Database
from app.models import Deployment, OperationPlan
from app.operations.executor import OperationExecutor
from app.services import deployments as deployment_service
from app.services import diagnostics
from app.services.diagnostics import parse_diagnostic_content, sanitize_steps


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
    with database.session_factory() as db:
        plan = OperationPlan(
            summary="Lifecycle action",
            diagnosis="Exercise the deployment",
            risk="low",
            status="approved",
            steps=[
                {
                    "operation": operation,
                    "deployment_id": deployment_id,
                    "reason": "Test lifecycle action",
                    "executable": True,
                }
            ],
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
