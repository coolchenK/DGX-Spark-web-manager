import pytest
from app.models import (
    ApiKey,
    AuditEvent,
    Deployment,
    ModelAsset,
    OperationPlan,
    OpsMessage,
    OpsSession,
    OpsToolRun,
    Provider,
    RequestMetric,
    SecretSetting,
    TaskRecord,
)
from sqlalchemy import func, select


def test_huggingface_token_is_encrypted_and_applied(authenticated_client):
    initial = authenticated_client.get("/api/settings")
    assert initial.status_code == 200
    assert initial.json()["huggingface"]["token_configured"] is False
    assert "token" not in initial.json()["huggingface"]

    updated = authenticated_client.patch(
        "/api/settings/huggingface",
        json={"token": "hf_test-secret-token-123456"},
    )

    assert updated.status_code == 200
    assert updated.json()["token_configured"] is True
    assert authenticated_client.app.state.huggingface_service.token == "hf_test-secret-token-123456"
    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import SecretSetting

        setting = db.get(SecretSetting, "huggingface_token")
        assert "hf_test-secret-token-123456" not in setting.encrypted_value


def test_spa_is_served_for_client_routes(settings, tmp_path):
    from app.main import create_app
    from fastapi.testclient import TestClient

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>DGX Manager SPA</body></html>")
    settings.static_dir = static_dir

    with TestClient(create_app(settings)) as client:
        response = client.get("/deployments")

    assert response.status_code == 200
    assert "DGX Manager SPA" in response.text


def _seed_history(authenticated_client):
    with authenticated_client.app.state.database.session_factory() as db:
        model = ModelAsset(name="keep-model", local_path="/models/keep")
        provider = Provider(
            name="keep-provider",
            base_url="https://provider.example/v1",
            default_model="ops-model",
            encrypted_api_key="encrypted-provider-key",
        )
        db.add_all([model, provider])
        db.flush()
        deployment = Deployment(
            name="keep-deployment",
            model_id=model.id,
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8000/v1",
            api_model_name="keep-api-model",
        )
        plan = OperationPlan(
            provider_id=provider.id,
            summary="remove plan",
            diagnosis="sensitive diagnosis",
            status="rejected",
        )
        session = OpsSession(
            title="remove session",
            provider_id=provider.id,
            status="answered",
        )
        failed_task = TaskRecord(
            type="model.download",
            status="failed",
            title="remove failed task",
            error="sensitive failure",
        )
        successful_task = TaskRecord(
            type="model.download",
            status="succeeded",
            title="keep successful task",
        )
        db.add_all([deployment, plan, session, failed_task, successful_task])
        db.flush()
        message = OpsMessage(
            session_id=session.id,
            role="assistant",
            content="sensitive response",
            operation_plan_id=plan.id,
        )
        tool = OpsToolRun(
            session_id=session.id,
            tool_name="docker.logs",
            status="succeeded",
            result_json={"output": "sensitive output"},
        )
        db.add_all(
            [
                message,
                tool,
                ApiKey(name="keep-key", prefix="dgx_keep", key_hash="a" * 64),
                SecretSetting(key="keep-secret", encrypted_value="encrypted"),
                RequestMetric(
                    model="keep-api-model",
                    endpoint="/v1/chat/completions",
                    status_code=200,
                    latency_ms=10,
                ),
                AuditEvent(
                    actor="admin",
                    action="ops.tool.execute",
                    resource_type="ops_tool_run",
                    resource_id=tool.id,
                    details={"secret": "remove"},
                ),
                AuditEvent(
                    actor="admin",
                    action="operation_plan.reject",
                    resource_type="operation_plan",
                    resource_id=plan.id,
                ),
                AuditEvent(
                    actor="admin",
                    action="model.download.queue",
                    resource_type="task",
                    resource_id=failed_task.id,
                ),
                AuditEvent(
                    actor="admin",
                    action="model.scan",
                    resource_type="model",
                    resource_id=model.id,
                ),
            ]
        )
        db.commit()
        return {
            "model": model.id,
            "provider": provider.id,
            "deployment": deployment.id,
            "plan": plan.id,
            "session": session.id,
            "message": message.id,
            "tool": tool.id,
            "failed_task": failed_task.id,
            "successful_task": successful_task.id,
        }


def test_history_clear_requires_csrf_and_exact_confirmation(authenticated_client):
    without_csrf = authenticated_client.headers.pop("X-CSRF-Token")
    try:
        forbidden = authenticated_client.request(
            "DELETE",
            "/api/settings/alerts-diagnostics-history",
            json={"confirmation": "清除历史记录"},
        )
    finally:
        authenticated_client.headers["X-CSRF-Token"] = without_csrf

    wrong = authenticated_client.request(
        "DELETE",
        "/api/settings/alerts-diagnostics-history",
        json={"confirmation": "清除诊断记录"},
    )

    assert forbidden.status_code == 403
    assert wrong.status_code == 422


@pytest.mark.parametrize(
    ("task_type", "task_status", "cancel_requested"),
    [
        ("ops.respond", "queued", False),
        ("ops.respond", "running", True),
        ("operation.execute", "paused", False),
    ],
)
def test_history_clear_rejects_active_operations_tasks(
    authenticated_client,
    task_type,
    task_status,
    cancel_requested,
):
    ids = _seed_history(authenticated_client)
    with authenticated_client.app.state.database.session_factory() as db:
        db.add(
            TaskRecord(
                type=task_type,
                status=task_status,
                title="active operation",
                cancel_requested=cancel_requested,
            )
        )
        db.commit()

    response = authenticated_client.request(
        "DELETE",
        "/api/settings/alerts-diagnostics-history",
        json={"confirmation": "清除历史记录"},
    )

    assert response.status_code == 409
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.get(OpsSession, ids["session"]) is not None
        assert db.get(TaskRecord, ids["failed_task"]) is not None


@pytest.mark.parametrize("plan_status", ["approved", "executing"])
def test_history_clear_rejects_approved_unfinished_plan(
    authenticated_client,
    plan_status,
):
    ids = _seed_history(authenticated_client)
    with authenticated_client.app.state.database.session_factory() as db:
        db.get(OperationPlan, ids["plan"]).status = plan_status
        db.commit()

    response = authenticated_client.request(
        "DELETE",
        "/api/settings/alerts-diagnostics-history",
        json={"confirmation": "清除历史记录"},
    )

    assert response.status_code == 409
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.get(OperationPlan, ids["plan"]) is not None


def test_history_clear_physically_deletes_only_target_history(authenticated_client):
    ids = _seed_history(authenticated_client)

    response = authenticated_client.request(
        "DELETE",
        "/api/settings/alerts-diagnostics-history",
        json={"confirmation": "清除历史记录"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "deleted": {
            "failed_tasks": 1,
            "operation_plans": 1,
            "ops_sessions": 1,
            "ops_messages": 1,
            "ops_tool_runs": 1,
            "audit_events": 3,
        },
    }
    with authenticated_client.app.state.database.session_factory() as db:
        assert db.get(TaskRecord, ids["failed_task"]) is None
        assert db.get(OperationPlan, ids["plan"]) is None
        assert db.get(OpsSession, ids["session"]) is None
        assert db.get(OpsMessage, ids["message"]) is None
        assert db.get(OpsToolRun, ids["tool"]) is None
        assert db.get(TaskRecord, ids["successful_task"]) is not None
        assert db.get(ModelAsset, ids["model"]) is not None
        assert db.get(Provider, ids["provider"]) is not None
        assert db.get(Deployment, ids["deployment"]) is not None
        assert db.scalar(select(func.count()).select_from(ApiKey)) == 1
        assert db.get(SecretSetting, "keep-secret") is not None
        assert db.scalar(select(func.count()).select_from(RequestMetric)) == 1
        events = list(db.scalars(select(AuditEvent).order_by(AuditEvent.created_at)))
        assert [event.action for event in events] == [
            "auth.login",
            "model.scan",
            "maintenance.history.clear",
        ]
        clear_event = events[-1]
        assert clear_event.details == response.json()["deleted"]
        assert "sensitive" not in str(clear_event.details)


def test_history_clear_rolls_back_all_deletes_when_audit_insert_fails(
    authenticated_client,
    monkeypatch,
):
    import app.api.settings as settings_api

    ids = _seed_history(authenticated_client)

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(settings_api, "record_audit", fail_audit)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        authenticated_client.request(
            "DELETE",
            "/api/settings/alerts-diagnostics-history",
            json={"confirmation": "清除历史记录"},
        )

    with authenticated_client.app.state.database.session_factory() as db:
        assert db.get(TaskRecord, ids["failed_task"]) is not None
        assert db.get(OperationPlan, ids["plan"]) is not None
        assert db.get(OpsSession, ids["session"]) is not None
        assert db.get(OpsMessage, ids["message"]) is not None
        assert db.get(OpsToolRun, ids["tool"]) is not None
